"""
05_train_lstm.py
================
Trains a Long Short-Term Memory (LSTM) network to predict flight rerouting
probability by modeling the temporal evolution of weather conditions.

Why LSTM over plain ML:
  Weather conditions change over time. A flight may enter turbulence that
  was building up over the past 30–60 minutes. LSTM captures that sequential
  dependency by processing weather+trajectory as a TIME SERIES per flight,
  not as isolated snapshots. This is the key advantage over Random Forest.

Architecture:
  Input  → [batch, SEQ_LEN, N_FEATURES]  (sliding window of waypoints)
  LSTM 1 → 128 units, return_sequences=True
  Dropout → 0.3
  LSTM 2 → 64 units
  Dropout → 0.3
  Dense  → 32 units, ReLU
  Output → 1 unit, Sigmoid (rerouting probability)

Pipeline:
  1. Load preprocessed dataset
  2. Reconstruct per-flight sequences from waypoints
  3. Build sliding window sequences (SEQ_LEN waypoints → label)
  4. Use SAME train/val/test split as RF (from rf_split_indices.pkl)
  5. Train with early stopping + learning rate scheduler
  6. Evaluate on test set
  7. Save model and metrics

Outputs:
  models/lstm_model.keras
  models/lstm_metrics.json
  models/lstm_history.csv
"""

import json
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, BatchNormalization
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

PROCESSED  = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)

# Sequence window: how many consecutive waypoints feed one prediction.
# 10 waypoints ≈ 10–30 minutes of flight time (waypoints are ~1–3 min apart).
SEQ_LEN      = 20
BATCH_SIZE   = 256
MAX_EPOCHS   = 100
LEARNING_RATE = 1e-4
RANDOM_STATE = 42

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/05_train_lstm.log"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

from tensorflow.keras.regularizers import l2

from sklearn.preprocessing import StandardScaler

def scale_features(X_train, X_val, X_test):
    """
    Fit scaler on train sequences only, then apply to val and test.
    Reshape to 2D for sklearn, then back to 3D for LSTM.
    """
    n_train, seq, feats = X_train.shape

    scaler = StandardScaler()

    # Reshape to (samples * seq_len, features) to fit scaler
    X_train_2d = X_train.reshape(-1, feats)
    X_val_2d   = X_val.reshape(-1, feats)
    X_test_2d  = X_test.reshape(-1, feats)

    X_train_scaled = scaler.fit_transform(X_train_2d).reshape(X_train.shape)
    X_val_scaled   = scaler.transform(X_val_2d).reshape(X_val.shape)
    X_test_scaled  = scaler.transform(X_test_2d).reshape(X_test.shape)

    # Save scaler for inference later
    with open(MODELS_DIR / "lstm_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    log.info("Features scaled with StandardScaler (fit on train only)")
    return X_train_scaled, X_val_scaled, X_test_scaled

# ─────────────────────────────────────────────
# STEP 1 — Load data
# ─────────────────────────────────────────────

def load_data():
    """
    Loads the ML-ready dataset and the full dataset (which has icao24
    for grouping waypoints into per-flight sequences).

    We need the full dataset here because LSTM requires temporal ordering
    within each flight — the ML dataset (post-SMOTE) loses that structure.
    We use the ORIGINAL (pre-SMOTE) data for LSTM and handle imbalance
    via class_weight in the loss function instead.
    """
    df_full = pd.read_csv(PROCESSED / "dataset_full.csv")
    with open(PROCESSED / "feature_cols.json") as f:
        feature_cols = json.load(f)

    # Verify all feature columns are present
    missing = [c for c in feature_cols if c not in df_full.columns]
    if missing:
        raise ValueError(f"Missing feature columns in dataset_full.csv: {missing}")

    log.info(f"Loaded {len(df_full)} waypoints from dataset_full.csv")
    log.info(f"Unique flights (icao24): {df_full['icao24'].nunique()}")
    return df_full, feature_cols


# ─────────────────────────────────────────────
# STEP 2 — Build sliding window sequences
# ─────────────────────────────────────────────

def build_sequences(df: pd.DataFrame, feature_cols: list):
    """
    Converts per-waypoint rows into fixed-length sliding window sequences.

    For each flight (grouped by icao24):
      - Sort waypoints chronologically
      - Slide a window of SEQ_LEN waypoints across the trajectory
      - Label = the rerouting label of the LAST waypoint in each window
        (predicting whether the flight WILL reroute based on current + recent history)

    Returns:
      X_seq  : shape (N_sequences, SEQ_LEN, N_features)
      y_seq  : shape (N_sequences,)
      indices: original row indices of the last waypoint in each window
               (used to align with the RF split indices)
    """
    log.info(f"Building sequences (window={SEQ_LEN})...")

    X_list, y_list, idx_list = [], [], []

    groups = df.groupby("icao24")
    skipped = 0

    for icao24, group in groups:
        group = group.sort_values("wp_datetime").reset_index()
        orig_indices = group["index"].values   # original df row indices

        if len(group) < SEQ_LEN + 1:
            skipped += 1
            continue

        feat_matrix = group[feature_cols].values.astype(np.float32)
        label_array = group["rerouted"].values

        for i in range(len(group) - SEQ_LEN):
            window = feat_matrix[i : i + SEQ_LEN]
            label  = label_array[i + SEQ_LEN - 1]           # predict NEXT step's label
            last_idx = orig_indices[i + SEQ_LEN]

            X_list.append(window)
            y_list.append(label)
            idx_list.append(last_idx)

    X_seq = np.array(X_list, dtype=np.float32)
    y_seq = np.array(y_list, dtype=np.int32)
    indices = np.array(idx_list)

    log.info(f"Sequences built: {X_seq.shape[0]} total "
             f"(skipped {skipped} short flights)")
    log.info(f"Class distribution — 0: {(y_seq==0).sum()}, 1: {(y_seq==1).sum()}")
    return X_seq, y_seq, indices


# ─────────────────────────────────────────────
# STEP 3 — Align with RF split indices
# ─────────────────────────────────────────────

def split_sequences(X_seq, y_seq, indices, df_full):
    
    idx_to_icao  = df_full["icao24"].reset_index(drop=False).set_index("index")["icao24"]
    seq_flights  = np.array([idx_to_icao.get(i, "UNKNOWN") for i in indices])

    unique_flights = np.unique(seq_flights)

    flights_tv, flights_test = train_test_split(
        unique_flights, test_size=0.15, random_state=42)
    flights_train, flights_val = train_test_split(
        flights_tv, test_size=0.176, random_state=42)

    train_mask = np.isin(seq_flights, flights_train)
    val_mask   = np.isin(seq_flights, flights_val)
    test_mask  = np.isin(seq_flights, flights_test)

    X_train, y_train = X_seq[train_mask], y_seq[train_mask]
    X_val,   y_val   = X_seq[val_mask],   y_seq[val_mask]
    X_test,  y_test  = X_seq[test_mask],  y_seq[test_mask]

    # ← log AFTER y_val is defined
    log.info(f"LSTM split — train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}")
    log.info(f"Class dist (val) — 0: {(y_val==0).sum()}, 1: {(y_val==1).sum()}")

    return X_train, X_val, X_test, y_train, y_val, y_test

# ─────────────────────────────────────────────
# STEP 4 — Build LSTM model
# ─────────────────────────────────────────────

def build_model(n_features: int) -> tf.keras.Model:
    """
    Two-layer stacked LSTM with dropout regularization.

    Architecture choices explained:
      - Stacked LSTM (2 layers): first layer captures local temporal patterns
        (e.g., wind speed rising over 5 min); second layer captures longer
        dependencies (e.g., systematic deviation building over 30 min).
      - BatchNormalization: stabilizes training, reduces sensitivity to LR.
      - Dropout (0.3): prevents memorization of training sequences.
      - Sigmoid output: produces probability [0,1] — direct rerouting score.
    """
    
    model = Sequential([
        Input(shape=(SEQ_LEN, n_features)),

        LSTM(64, return_sequences=True, name="lstm_1",
            kernel_regularizer=l2(1e-4),
            recurrent_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.5),                          # was 0.3

        LSTM(32, return_sequences=False, name="lstm_2",
            kernel_regularizer=l2(1e-4),
            recurrent_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.5),                          # was 0.3

        Dense(16, activation="relu", name="dense_1",
            kernel_regularizer=l2(1e-4)),
        Dropout(0.3),                          # was 0.2

        Dense(1, activation="sigmoid", name="output"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
    )

    model.summary(print_fn=log.info)
    return model


# ─────────────────────────────────────────────
# STEP 5 — Train
# ─────────────────────────────────────────────

def train(model, X_train, y_train, X_val, y_val) -> pd.DataFrame:
    """
    Trains the LSTM with:
      - EarlyStopping: stops when val_loss stops improving (patience=10)
      - ReduceLROnPlateau: halves learning rate when stuck (patience=5)
      - ModelCheckpoint: saves the best epoch (by val_auc)
      - class_weight: handles residual class imbalance (alternative to SMOTE
        for sequential data where SMOTE would break temporal order)
    """
    # Compute class weights from training labels
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    weight_for_0 = (1 / n_neg) * (len(y_train) / 2.0)
    weight_for_1 = (1 / n_pos) * (len(y_train) / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    log.info(f"Class weights — 0: {weight_for_0:.3f}, 1: {weight_for_1:.3f}")

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=20,
            mode="min",
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(MODELS_DIR / "lstm_best_checkpoint.keras"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=0
        ),
    ]

    log.info(f"Starting LSTM training (max {MAX_EPOCHS} epochs)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    hist_df = pd.DataFrame(history.history)
    hist_df.index.name = "epoch"
    hist_df.to_csv(MODELS_DIR / "lstm_history.csv")
    log.info(f"Training stopped at epoch {len(hist_df)}")
    log.info(f"Best val_auc: {hist_df['val_auc'].max():.4f}")
    return hist_df


# ─────────────────────────────────────────────
# STEP 6 — Evaluate
# ─────────────────────────────────────────────

def evaluate(model, X_test, y_test, history: pd.DataFrame) -> dict:
    y_proba = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).flatten()
    y_pred  = (y_proba >= 0.5).astype(int)

    metrics = {
        "accuracy":  round(float(accuracy_score(y_test, y_pred)),  4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall":    round(float(recall_score(y_test, y_pred)),    4),
        "f1":        round(float(f1_score(y_test, y_pred)),        4),
        "auc_roc":   round(float(roc_auc_score(y_test, y_proba)),  4),
    }

    log.info("\n" + "="*40)
    log.info("LSTM — TEST SET RESULTS")
    log.info("="*40)
    for k, v in metrics.items():
        log.info(f"  {k:<12}: {v}")
    log.info("\nClassification Report:")
    log.info(classification_report(y_test, y_pred,
                                   target_names=["Not Rerouted", "Rerouted"]))

    # ── Plots ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Training curves
    axes[0].plot(history["auc"],     label="Train AUC")
    axes[0].plot(history["val_auc"], label="Val AUC")
    axes[0].set_title("Training AUC over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    # Confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["Not Rerouted", "Rerouted"],
        cmap="Blues", ax=axes[1]
    )
    axes[1].set_title("Confusion Matrix")

    # ROC curve
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[2])
    axes[2].set_title(f"ROC Curve (AUC = {metrics['auc_roc']:.3f})")
    axes[2].plot([0,1],[0,1],"k--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(MODELS_DIR / "lstm_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved evaluation plots → models/lstm_evaluation.png")

    return metrics


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("LSTM Training — Flight Path Rerouting Project")
    log.info("=" * 60)

    # 1. Load
    df, feature_cols = load_data()

    # 2. Build sequences
    X_seq, y_seq, indices = build_sequences(df, feature_cols)
    n_features = X_seq.shape[2]

    # 3. Split (aligned with RF)
    X_train, X_val, X_test, y_train, y_val, y_test = \
        split_sequences(X_seq, y_seq, indices, df)

    # 3b. Scale
    X_train, X_val, X_test = scale_features(X_train, X_val, X_test)

    # 4. Build model
    model = build_model(n_features)

    # 5. Train
    history = train(model, X_train, y_train, X_val, y_val)

    # 6. Evaluate
    metrics = evaluate(model, X_test, y_test, history)

    # 7. Save
    model.save(MODELS_DIR / "lstm_model.keras")
    log.info("Model saved -> models/lstm_model.keras")

    with open(MODELS_DIR / "lstm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save val/test data for ensemble script
    # We save val data for weight optimization and test data for final comparison
    np.save(MODELS_DIR / "lstm_X_val.npy", X_val)
    np.save(MODELS_DIR / "lstm_y_val.npy", y_val)
    np.save(MODELS_DIR / "lstm_X_test.npy", X_test)
    np.save(MODELS_DIR / "lstm_y_test.npy", y_test)
    
    log.info("Test and Val arrays saved for ensemble -> models/lstm_*.npy")
    log.info("Metrics saved -> models/lstm_metrics.json")
    log.info("Test arrays saved for ensemble -> models/lstm_X_test.npy")
    log.info("\nLSTM training complete.")


if __name__ == "__main__":
    main()
