"""
06_ensemble.py
==============
Combines RF and LSTM predictions into a weighted ensemble and performs
the final comparative evaluation for the study (Table 1 in the paper).

How the ensemble works:
  - RF outputs  P_rf(rerouted)  — probability from predict_proba()
  - LSTM outputs P_lstm(rerouted) — probability from sigmoid output
  - Ensemble:  P_ensemble = w_rf * P_rf + w_lstm * P_lstm
  - Weights (w_rf, w_lstm) are optimized on the VALIDATION set by
    grid-searching the combination that maximizes F1 score.
  - Final classification: P_ensemble >= 0.5 → rerouted

Evaluation:
  - Compares RF, LSTM, and Ensemble side by side (Table 1)
  - DeLong test for statistical significance of AUC differences
  - Saves ensemble model components for use in 07_risk_scoring.py

Outputs:
  models/ensemble_weights.json
  models/ensemble_metrics_comparison.json
  models/ensemble_comparison.png
  models/delong_test_results.json
"""

import json
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product

import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    RocCurveDisplay
)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

PROCESSED  = Path("data/processed")
MODELS_DIR = Path("models")
Path("logs").mkdir(exist_ok=True)

THRESHOLD = 0.5   # classification threshold for ensemble probability

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/06_ensemble.log"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# STEP 1 — Load models and data
# ─────────────────────────────────────────────

def load_rf():
    with open(MODELS_DIR / "rf_model.pkl", "rb") as f:
        rf = pickle.load(f)
    log.info("RF model loaded.")
    return rf


def load_lstm():
    model = tf.keras.models.load_model(MODELS_DIR / "lstm_model.keras")
    log.info("LSTM model loaded.")
    return model


def load_test_data():
    """
    Loads test sets for both models.
    RF uses flat feature vectors; LSTM uses sequences.
    Both share the same underlying flight events (same split).
    """
    with open(MODELS_DIR / "rf_split_indices.pkl", "rb") as f:
        splits = pickle.load(f)

    df = pd.read_csv(PROCESSED / "dataset_ml.csv")
    with open(PROCESSED / "feature_cols.json") as f:
        feature_cols = json.load(f)

    # RF test data
    X_rf_test = df.loc[splits["test"], feature_cols].values
    y_rf_test = df.loc[splits["test"], "rerouted"].values

    # LSTM test data (pre-saved by 05_train_lstm.py)
    X_lstm_test = np.load(MODELS_DIR / "lstm_X_test.npy")
    y_lstm_test = np.load(MODELS_DIR / "lstm_y_test.npy")

    # RF validation data (for weight optimization)
    X_rf_val = df.loc[splits["val"], feature_cols].values
    y_rf_val = df.loc[splits["val"], "rerouted"].values

    log.info(f"RF test set:   {X_rf_test.shape}")
    log.info(f"LSTM test set: {X_lstm_test.shape}")
    return (X_rf_test, y_rf_test, X_rf_val, y_rf_val,
            X_lstm_test, y_lstm_test)


# ─────────────────────────────────────────────
# STEP 2 — Generate individual model probabilities
# ─────────────────────────────────────────────

def get_probabilities(rf, lstm, X_rf, X_lstm) -> tuple:
    """Returns (P_rf, P_lstm) probability arrays for positive class."""
    P_rf   = rf.predict_proba(X_rf)[:, 1]
    P_lstm = lstm.predict(X_lstm, verbose=0).flatten()
    return P_rf, P_lstm


# ─────────────────────────────────────────────
# STEP 3 — Optimize ensemble weights on validation set
# ─────────────────────────────────────────────

def optimize_weights(rf, lstm, X_rf_val, y_rf_val,
                     X_lstm_val, y_lstm_val) -> tuple:
    """
    Grid searches the weight combination (w_rf, w_lstm) that maximizes
    F1 score on the validation set.
    """
    log.info("Optimizing ensemble weights on validation set...")

    P_rf_v, P_lstm_v = get_probabilities(rf, lstm, X_rf_val, X_lstm_val)
    
    # Align validation slices
    n = min(len(P_rf_v), len(P_lstm_v))
    y_val_c    = y_rf_val[:n]
    P_rf_c     = P_rf_v[:n]
    P_lstm_c   = P_lstm_v[:n]

    best_f1, best_w_rf = 0.0, 0.5

    for w_rf in np.arange(0.0, 1.01, 0.05):
        w_lstm = 1.0 - w_rf
        P_ens  = w_rf * P_rf_c + w_lstm * P_lstm_c
        y_pred = (P_ens >= THRESHOLD).astype(int)
        score  = f1_score(y_val_c, y_pred, zero_division=0)

        if score > best_f1:
            best_f1  = score
            best_w_rf = w_rf

    best_w_lstm = 1.0 - best_w_rf
    log.info(f"Best weights -> RF: {best_w_rf:.2f}, LSTM: {best_w_lstm:.2f} "
             f"(Val F1: {best_f1:.4f})")
    return best_w_rf, best_w_lstm


# ─────────────────────────────────────────────
# STEP 4 — DeLong test for AUC comparison
# ─────────────────────────────────────────────

def delong_test(y_true, p1, p2) -> dict:
    """
    DeLong's method for comparing two AUC-ROC values statistically.
    Tests H0: AUC(model_1) == AUC(model_2)

    Implementation follows:
      DeLong, E.R., DeLong, D.M. & Clarke-Pearson, D.L. (1988).
      Biometrics, 44(3), 837–845.

    Returns: {auc1, auc2, z_stat, p_value}
    """
    def compute_midrank(x):
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N, dtype=float)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1)
            i = j
        T2 = np.empty(N, dtype=float)
        T2[J] = T + 1
        return T2

    def fastDeLong(predictions_sorted_transposed, label_1_count):
        m = label_1_count
        n = predictions_sorted_transposed.shape[1] - m
        positive_examples = predictions_sorted_transposed[:, :m]
        negative_examples = predictions_sorted_transposed[:, m:]
        k = predictions_sorted_transposed.shape[0]

        tx = np.empty([k, m], dtype=float)
        ty = np.empty([k, n], dtype=float)
        tz = np.empty([k, m + n], dtype=float)
        for r in range(k):
            tx[r, :] = compute_midrank(positive_examples[r, :])
            ty[r, :] = compute_midrank(negative_examples[r, :])
            tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

        aucs = (tz[:, :m].sum(axis=1) - tx.sum(axis=1)) / (m * n)
        v01  = (tz[:, :m] - tx[:, :]) / n
        v10  = 1.0 - (tz[:, m:] - ty[:, :]) / m
        sx   = np.cov(v01)
        sy   = np.cov(v10)
        delongcov = sx / m + sy / n
        return aucs, delongcov

    order      = (-y_true).argsort()
    label_1_count = int(y_true.sum())
    predictions_sorted = np.vstack([p1, p2])[:, order]
    aucs, cov  = fastDeLong(predictions_sorted, label_1_count)

    auc_diff   = aucs[0] - aucs[1]
    se         = np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    z_stat     = auc_diff / (se + 1e-12)

    from scipy import stats
    p_value = float(2 * stats.norm.sf(abs(z_stat)))

    return {
        "auc_model_1": round(float(aucs[0]), 4),
        "auc_model_2": round(float(aucs[1]), 4),
        "z_statistic": round(float(z_stat),  4),
        "p_value":     round(p_value,         6),
        "significant_at_0.05": p_value < 0.05,
    }


# ─────────────────────────────────────────────
# STEP 5 — Full comparative evaluation
# ─────────────────────────────────────────────

def evaluate_all(rf, lstm, w_rf, w_lstm,
                 X_rf_test, y_rf_test,
                 X_lstm_test, y_lstm_test) -> dict:
    """
    Evaluates RF, LSTM, and Ensemble on the test set.
    Produces Table 1 from the paper.
    """

    def score(y_true, y_prob, name):
        y_pred = (y_prob >= THRESHOLD).astype(int)
        m = {
            "model":     name,
            "accuracy":  round(float(accuracy_score(y_true, y_pred)),  4),
            "precision": round(float(precision_score(y_true, y_pred)), 4),
            "recall":    round(float(recall_score(y_true, y_pred)),    4),
            "f1":        round(float(f1_score(y_true, y_pred)),        4),
            "auc_roc":   round(float(roc_auc_score(y_true, y_prob)),   4),
        }
        log.info(f"\n{'='*40}")
        log.info(f"{name} — TEST RESULTS")
        log.info(f"{'='*40}")
        for k, v in m.items():
            if k != "model":
                log.info(f"  {k:<12}: {v}")
        log.info(classification_report(y_true, y_pred,
                                       target_names=["Not Rerouted", "Rerouted"]))
        return m, y_pred, y_prob

    # Individual model predictions on their respective test data
    P_rf   = rf.predict_proba(X_rf_test)[:, 1]
    P_lstm = lstm.predict(X_lstm_test, verbose=0).flatten()
    # Ensemble — average both probability arrays
    # NOTE: if test sets differ in length due to sequence windowing,
    # we use the minimum length for a fair comparison slice
    n = min(len(P_rf), len(P_lstm))
    y_test_common = y_rf_test[:n]     # use RF labels as ground truth
    P_rf_c   = P_rf[:n]
    P_lstm_c = P_lstm[:n]
    P_ens    = w_rf * P_rf_c + w_lstm * P_lstm_c

    m_rf,   pred_rf,   prob_rf   = score(y_test_common, P_rf_c,   "Random Forest")
    m_lstm, pred_lstm, prob_lstm = score(y_lstm_test,  P_lstm, "LSTM")

    n        = min(len(P_rf), len(P_lstm))
    y_common = y_rf_test[:n]
    P_ens    = w_rf * P_rf[:n] + w_lstm * P_lstm[:n]
    
    m_ens,  pred_ens,  prob_ens  = score(y_test_common, P_ens,    "Ensemble (RF+LSTM)")

    results = {
        "random_forest": m_rf,
        "lstm":          m_lstm,
        "ensemble":      m_ens,
    }

    # ── DeLong tests ───────────────────────────────────────────
    log.info("\n[DeLong Test] Ensemble vs Random Forest:")
    dl_ens_rf   = delong_test(y_common, prob_ens,      P_rf[:n])
    log.info(f"  AUC Ensemble: {dl_ens_rf['auc_model_1']}, "
             f"AUC RF: {dl_ens_rf['auc_model_2']}, "
             f"p={dl_ens_rf['p_value']}, "
             f"Significant: {dl_ens_rf['significant_at_0.05']}")

    log.info("[DeLong Test] Ensemble vs LSTM:")
    dl_ens_lstm = delong_test(y_lstm_test[:n], prob_ens, P_lstm[:n])    
    log.info(f"  AUC Ensemble: {dl_ens_lstm['auc_model_1']}, "
             f"AUC LSTM: {dl_ens_lstm['auc_model_2']}, "
             f"p={dl_ens_lstm['p_value']}, "
             f"Significant: {dl_ens_lstm['significant_at_0.05']}")

    log.info("[DeLong Test] LSTM vs Random Forest:")
    dl_lstm_rf  = delong_test(y_common, P_lstm[:n],    P_rf[:n])
    log.info(f"  AUC LSTM: {dl_lstm_rf['auc_model_1']}, "
             f"AUC RF: {dl_lstm_rf['auc_model_2']}, "
             f"p={dl_lstm_rf['p_value']}, "
             f"Significant: {dl_lstm_rf['significant_at_0.05']}")

    delong_results = {
        "ensemble_vs_rf":   dl_ens_rf,
        "ensemble_vs_lstm": dl_ens_lstm,
        "lstm_vs_rf":       dl_lstm_rf,
    }

    with open(MODELS_DIR / "delong_test_results.json", "w") as f:
        json.dump(delong_results, f, indent=2)
    log.info("DeLong results saved -> models/delong_test_results.json")

    # ── Comparison plot ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Metric comparison bar chart
    metric_names = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    x = np.arange(len(metric_names))
    w = 0.25
    rf_vals   = [m_rf[m]   for m in metric_names]
    lstm_vals = [m_lstm[m] for m in metric_names]
    ens_vals  = [m_ens[m]  for m in metric_names]

    axes[0].bar(x - w, rf_vals,   w, label="Random Forest", color="#4C72B0")
    axes[0].bar(x,     lstm_vals, w, label="LSTM",           color="#DD8452")
    axes[0].bar(x + w, ens_vals,  w, label="Ensemble",       color="#55A868")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m.replace("_", "\n") for m in metric_names])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Model Comparison — Test Set Metrics")
    axes[0].legend()
    axes[0].set_ylabel("Score")

    # ROC curves
    for prob, label, color in [
        (P_rf_c,   f"RF (AUC={m_rf['auc_roc']})",     "#4C72B0"),
        (P_lstm_c, f"LSTM (AUC={m_lstm['auc_roc']})",  "#DD8452"),
        (P_ens,    f"Ensemble (AUC={m_ens['auc_roc']})", "#55A868"),
    ]:
        RocCurveDisplay.from_predictions(
            y_test_common, prob,
            name=label, ax=axes[1], color=color
        )
    axes[1].plot([0,1],[0,1],"k--", alpha=0.4, label="Random")
    axes[1].set_title("ROC Curves — All Models")
    axes[1].legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(MODELS_DIR / "ensemble_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Comparison plot saved -> models/ensemble_comparison.png")

    return results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Ensemble Evaluation — Flight Path Rerouting Project")
    log.info("=" * 60)

    # 1. Load
    rf   = load_rf()
    lstm = load_lstm()
    (X_rf_test, y_rf_test, X_rf_val, y_rf_val,
     X_lstm_test, y_lstm_test) = load_test_data()

    # 2. Optimize weights on validation set
    X_lstm_val = np.load(MODELS_DIR / "lstm_X_val.npy")
    y_lstm_val = np.load(MODELS_DIR / "lstm_y_val.npy")

    w_rf, w_lstm = optimize_weights(
        rf, lstm, X_rf_val, y_rf_val, X_lstm_val, y_lstm_val
    )

    # Save weights
    weights = {"w_rf": round(float(w_rf), 4), "w_lstm": round(float(w_lstm), 4)}
    with open(MODELS_DIR / "ensemble_weights.json", "w") as f:
        json.dump(weights, f, indent=2)
    log.info("Ensemble weights saved -> models/ensemble_weights.json")

    # 3. Evaluate all models
    results = evaluate_all(
        rf, lstm, w_rf, w_lstm,
        X_rf_test, y_rf_test,
        X_lstm_test, y_lstm_test
    )

    with open(MODELS_DIR / "ensemble_metrics_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info("Metrics comparison saved -> models/ensemble_metrics_comparison.json")

    # 4. Print Table 1 summary
    log.info("\n" + "="*60)
    log.info("TABLE 1 - Model Performance on Test Set")
    log.info("="*60)
    header = f"{'Model':<22} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}"
    log.info(header)
    log.info("-"*60)
    for key, m in results.items():
        log.info(
            f"{m['model']:<22} "
            f"{m['accuracy']:>7.4f} "
            f"{m['precision']:>7.4f} "
            f"{m['recall']:>7.4f} "
            f"{m['f1']:>7.4f} "
            f"{m['auc_roc']:>7.4f}"
        )
    log.info("="*60)

    log.info("\nEnsemble evaluation complete.")


if __name__ == "__main__":
    main()
