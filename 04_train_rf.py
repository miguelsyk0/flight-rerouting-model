"""
04_train_rf.py
==============
Trains a Random Forest classifier to predict flight rerouting probability.

Pipeline:
  1. Load preprocessed dataset (data/processed/dataset_ml.csv)
  2. Train/validation/test split (70/15/15)
  3. Hyperparameter tuning via 5-fold cross-validation (RandomizedSearchCV)
  4. Train final model on full train+val set with best hyperparameters
  5. Evaluate on held-out test set
  6. Save model, metrics, and feature importance

Outputs:
  models/rf_model.pkl
  models/rf_metrics.json
  models/rf_feature_importance.csv
  models/rf_split_indices.pkl  ← shared with LSTM so both use same test set
"""

import json
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "2"  # cap loky worker count

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV, StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay
)
from scipy.stats import randint

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

PROCESSED  = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)

RANDOM_STATE   = 42
TEST_SIZE      = 0.15
VAL_SIZE       = 0.15    # fraction of the non-test portion
N_ITER_SEARCH  = 10      # number of hyperparameter combinations to try
CV_FOLDS       = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/04_train_rf.log"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# STEP 1 — Load data
# ─────────────────────────────────────────────

def load_data():
    df = pd.read_csv(PROCESSED / "dataset_ml.csv")
    with open(PROCESSED / "feature_cols.json") as f:
        feature_cols = json.load(f)

    X = df[feature_cols].values
    y = df["rerouted"].values

    log.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    log.info(f"Class distribution — 0: {(y==0).sum()}, 1: {(y==1).sum()}")
    return X, y, feature_cols


# ─────────────────────────────────────────────
# STEP 2 — Train / val / test split
# ─────────────────────────────────────────────

def split_data(X, y):
    """
    Stratified split to preserve class ratio across all three sets.
    Splits: 70% train | 15% val | 15% test
    The test indices are saved so the LSTM uses the exact same test set.
    """
    # First split off test set
    X_trainval, X_test, y_trainval, y_test, idx_trainval, idx_test = \
        train_test_split(
            X, y, np.arange(len(y)),
            test_size=TEST_SIZE,
            stratify=y,
            random_state=RANDOM_STATE
        )

    # Split remainder into train and val
    val_fraction = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val, idx_train, idx_val = \
        train_test_split(
            X_trainval, y_trainval, idx_trainval,
            test_size=val_fraction,
            stratify=y_trainval,
            random_state=RANDOM_STATE
        )

    log.info(f"Split sizes — train: {len(y_train)}, "
             f"val: {len(y_val)}, test: {len(y_test)}")

    # Save split indices — LSTM will reuse these for fair comparison
    with open(MODELS_DIR / "rf_split_indices.pkl", "wb") as f:
        pickle.dump({
            "train": idx_train, "val": idx_val, "test": idx_test
        }, f)

    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────
# STEP 3 — Hyperparameter search
# ─────────────────────────────────────────────

def tune_hyperparameters(X_train, y_train) -> dict:
    """
    Uses RandomizedSearchCV with 5-fold stratified CV.
    Optimizes for F1 score (best for imbalanced classification).

    Search space covers the most impactful RF hyperparameters:
      n_estimators   — number of trees (more = better, diminishing returns)
      max_depth      — tree depth cap (prevents overfitting)
      min_samples_split/leaf — min samples to split a node
      max_features   — features considered at each split ("sqrt" is standard)
      class_weight   — handles any residual class imbalance after SMOTE
    """
    log.info(f"Running hyperparameter search ({N_ITER_SEARCH} iterations, "
             f"{CV_FOLDS}-fold CV)...")

    param_dist = {
        "n_estimators":      randint(100, 300),
        "max_depth":         [None, 10, 20, 30, 40, 50],
        "min_samples_split": randint(2, 20),
        "min_samples_leaf":  randint(1, 10),
        "max_features":      ["sqrt", "log2", 0.3, 0.5],
        "class_weight":      [None, "balanced"],
        "bootstrap":         [True, False],
    }

    base_rf = RandomForestClassifier(random_state=RANDOM_STATE)
    cv      = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                               random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_dist,
        n_iter=N_ITER_SEARCH,
        scoring="f1",
        cv=cv,
        n_jobs=4,
        verbose=1,
        random_state=RANDOM_STATE,
        return_train_score=True,
    )
    search.fit(X_train, y_train)

    best = search.best_params_
    log.info(f"Best hyperparameters: {best}")
    log.info(f"Best CV F1: {search.best_score_:.4f}")
    return best


# ─────────────────────────────────────────────
# STEP 4 — Train final model
# ─────────────────────────────────────────────

def train_final_model(X_train, X_val, y_train, y_val,
                      best_params: dict) -> RandomForestClassifier:
    """
    Trains the final RF on the combined train+val set using the best
    hyperparameters found during search.

    Combining train+val for final training is standard practice —
    it gives the model more data while keeping the test set unseen.
    """
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])

    log.info("Training final Random Forest on train+val set...")
    # OOB score is only available if bootstrap=True
    use_bootstrap = best_params.get("bootstrap", True)
    
    rf = RandomForestClassifier(
        **best_params,
        random_state=RANDOM_STATE,
        n_jobs=4,
        oob_score=use_bootstrap
    )
    rf.fit(X_full, y_full)
    if use_bootstrap:
        log.info(f"OOB Score (internal validation): {rf.oob_score_:.4f}")
    return rf


# ─────────────────────────────────────────────
# STEP 5 — Evaluate on test set
# ─────────────────────────────────────────────

def evaluate(model, X_test, y_test, feature_cols: list) -> dict:
    """
    Full evaluation on the held-out test set.
    Returns a metrics dict and saves plots.
    """
    y_pred      = model.predict(X_test)
    y_proba     = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred),    4),
        "f1":        round(f1_score(y_test, y_pred),        4),
        "auc_roc":   round(roc_auc_score(y_test, y_proba),  4),
        "oob_score": round(model.oob_score_, 4) if getattr(model, "oob_score", False) else None,
    }

    log.info("\n" + "="*40)
    log.info("RANDOM FOREST — TEST SET RESULTS")
    log.info("="*40)
    for k, v in metrics.items():
        log.info(f"  {k:<12}: {v}")
    log.info("\nClassification Report:")
    log.info(classification_report(y_test, y_pred,
                                   target_names=["Not Rerouted", "Rerouted"]))

    # ── Feature importance ─────────────────────────────────────
    importance_df = pd.DataFrame({
        "feature":   feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    importance_df.to_csv(MODELS_DIR / "rf_feature_importance.csv", index=False)
    log.info("\nTop 10 Features:")
    for _, row in importance_df.head(10).iterrows():
        log.info(f"  {row['feature']:<30} {row['importance']:.4f}")

    # ── Plots ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["Not Rerouted", "Rerouted"],
        cmap="Blues", ax=axes[0]
    )
    axes[0].set_title("Confusion Matrix")

    # ROC curve
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[1])
    axes[1].set_title(f"ROC Curve (AUC = {metrics['auc_roc']:.3f})")
    axes[1].plot([0,1],[0,1],"k--", alpha=0.5)

    # Feature importance bar chart (top 15)
    top15 = importance_df.head(15)
    axes[2].barh(top15["feature"][::-1], top15["importance"][::-1])
    axes[2].set_xlabel("Importance")
    axes[2].set_title("Top 15 Feature Importances")
    axes[2].tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    plt.savefig(MODELS_DIR / "rf_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved evaluation plots → models/rf_evaluation.png")

    return metrics


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Random Forest Training — Flight Path Rerouting Project")
    log.info("=" * 60)

    # 1. Load
    X, y, feature_cols = load_data()

    # 2. Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # 3. Tune
    best_params = tune_hyperparameters(X_train, y_train)

    # 4. Train
    rf = train_final_model(X_train, X_val, y_train, y_val, best_params)

    # 5. Evaluate
    metrics = evaluate(rf, X_test, y_test, feature_cols)

    # 6. Save model and metrics
    with open(MODELS_DIR / "rf_model.pkl", "wb") as f:
        pickle.dump(rf, f)
    log.info("Model saved -> models/rf_model.pkl")

    metrics["best_params"] = best_params
    with open(MODELS_DIR / "rf_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics saved -> models/rf_metrics.json")
    log.info("\nRandom Forest training complete.")


if __name__ == "__main__":
    main()
