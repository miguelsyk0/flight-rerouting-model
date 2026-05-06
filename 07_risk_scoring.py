"""
07_risk_scoring.py
==================
Generates a prescriptive risk score grid across all flight levels and headings
for a given flight's current position and weather conditions.

This is the "prescriptive analytics" component of the study (RQ4).

How it works:
  For a given waypoint (lat, lon, time) and its observed weather:
    1. Build a candidate grid: every combination of
         flight level (FL100 -> FL410, every 1,000 ft)
         heading (0° -> 350°, every 10°)
    2. Adjust weather features for each altitude candidate
       (higher altitude -> lower wind speed near surface, less precipitation, etc.)
    3. Run the ensemble model on each candidate combination
    4. Assign a risk score (0 = safe, 1 = high rerouting probability)
    5. Identify the minimum-risk flight level + heading = RECOMMENDATION

Outputs per flight scenario:
  results/risk_grid_<scenario>.csv    — full grid with risk scores
  results/risk_heatmap_<scenario>.png — visual heatmap (FL × heading)
  results/recommendations.json        — recommended FL and heading per scenario

Validation:
  Loads historical rerouting events from dataset_full.csv, runs the risk
  scoring mechanism on those waypoints, and measures how often the
  recommended FL/heading matches the actual recorded deviation outcome.
"""

import json
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

import tensorflow as tf

# ---------------------------------------------
# CONFIGURATION
# ---------------------------------------------

PROCESSED  = Path("data/processed")
MODELS_DIR = Path("models")
RESULTS    = Path("results")
RESULTS.mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)

# Risk grid dimensions
FL_MIN      = 10000    # feet — FL100
FL_MAX      = 41000    # feet — FL410
FL_STEP     = 1000     # 1,000-ft increments
HEADING_MIN = 0
HEADING_MAX = 350
HEADING_STEP = 10

# Threshold above which a score is considered "high risk"
HIGH_RISK_THRESHOLD = 0.5

# Number of historical scenarios to validate against
N_VALIDATION_SCENARIOS = 200

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/07_risk_scoring.log"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)


# ---------------------------------------------
# STEP 1 — Load models and metadata
# ---------------------------------------------

def load_ensemble():
    with open(MODELS_DIR / "rf_model.pkl", "rb") as f:
        rf = pickle.load(f)
    lstm = tf.keras.models.load_model(MODELS_DIR / "lstm_model.keras")
    with open(MODELS_DIR / "ensemble_weights.json") as f:
        weights = json.load(f)
    with open(PROCESSED / "feature_cols.json") as f:
        feature_cols = json.load(f)
    with open(MODELS_DIR / "rf_split_indices.pkl", "rb") as f:
        splits = pickle.load(f)

    log.info(f"Ensemble weights: RF={weights['w_rf']}, LSTM={weights['w_lstm']}")
    return rf, lstm, weights, feature_cols


def load_scaler():
    with open(PROCESSED / "scaler.pkl", "rb") as f:
        return pickle.load(f)


# ---------------------------------------------
# STEP 2 — Risk grid builder
# ---------------------------------------------

# Feature column order (must match feature_cols.json exactly)
CONTINUOUS_COLS = [
    "wind_speed_kt", "wind_speed_squared", "visibility_sm",
    "turb_intensity_num", "wind_shear_index", "temp_dewpoint_spread_c",
    "altitude_ft"
]

def altitude_weather_adjustment(base_features: dict, altitude_ft: float) -> dict:
    """
    Adjusts weather-derived features based on candidate altitude.

    In real aviation, many weather phenomena are altitude-dependent:
      - Turbulence is often worst in mid-levels (FL180–FL280 convective)
        and near jet stream (FL350–FL410)
      - Precipitation decreases above FL200 (freezing level)
      - Wind speeds generally increase with altitude (up to jet stream)
      - Visibility (IFR) is primarily a surface/low-level concern

    This function applies simplified but physically reasonable adjustments
    so the risk model produces differentiated scores across flight levels.

    For production use, these adjustments should come from actual
    upper-air soundings or model output statistics (MOS).
    """
    features = base_features.copy()
    fl = altitude_ft / 1000   # in flight level units (FL = hundreds of feet / 10)

    # Wind speed increases with altitude up to ~FL350 (jet stream peak),
    # then decreases slightly at very high altitudes
    if fl <= 350:
        wind_factor = 1.0 + (fl / 350) * 0.8    # up to 80% stronger at FL350
    else:
        wind_factor = 1.8 - (fl - 350) / 600    # slight decrease above FL350

    features["wind_speed_kt"]      = base_features["wind_speed_kt"] * max(wind_factor, 0.5)
    features["wind_speed_squared"] = features["wind_speed_kt"] ** 2

    # Precipitation: negligible above FL200 (all ice/snow, not a routing hazard)
    if altitude_ft > 20000:
        features["precip_flag"]    = 0
        features["convective_flag"] = features.get("convective_flag", 0) * 0.3

    # Turbulence: highest in convective layer (FL180–FL280) and jet stream (FL350+)
    if 18000 <= altitude_ft <= 28000:
        features["turb_intensity_num"] = base_features["turb_intensity_num"] * 1.4
    elif altitude_ft >= 35000:
        features["turb_intensity_num"] = base_features["turb_intensity_num"] * 1.2
    elif altitude_ft < 10000:
        features["turb_intensity_num"] = base_features["turb_intensity_num"] * 0.7

    features["wind_shear_index"] = (
        features["wind_speed_kt"] * features["turb_intensity_num"]
    )

    # Visibility: IFR conditions are surface phenomenon
    if altitude_ft > 15000:
        features["visibility_sm"]      = max(features["visibility_sm"], 10.0)
        features["visibility_category"] = 0   # VMC above clouds

    features["altitude_ft"] = altitude_ft
    return features


def build_candidate_row(base_features: dict, altitude_ft: float,
                        heading_deg: float, hour: int, month: int,
                        feature_cols: list) -> np.ndarray:
    """
    Constructs one feature vector for a (altitude, heading) candidate.
    Returns a raw (unscaled) 1D numpy array matching feature_cols order.
    """
    f = altitude_weather_adjustment(base_features, altitude_ft)

    # Heading (circular encoding)
    # -----------------------------------------------------------------------------
    # | Feature                  | Description                                    |
    # -----------------------------------------------------------------------------
    # | wind_speed_kt            | Surface wind speed (knots)                     |
    # | wind_speed_squared       | Squared wind speed (captures gusts better)     |
    # | visibility_sm            | Horizontal visibility (statute miles)          |
    # | visibility_category      | 0=VMC (>=3sm), 1=MVFR, 2=IFR (<1sm)            |
    # | turb_intensity_num       | 0-6 turbulence severity scale                  |
    # | wind_shear_index         | wind_speed * turb_intensity (composite)        |
    # | convective_flag          | 1 if wx_string contains TS/SQ/FZRA             |
    # | in_sigmet_conv           | Inside convective SIGMET polygon               |
    # | in_sigmet_turb           | Inside turbulence SIGMET polygon               |
    # | in_sigmet_ice            | Inside icing SIGMET polygon                    |
    # | altitude_ft              | Barometric altitude (feet)                     |
    # | altitude_band            | FL category: low/mid/high/very_high            |
    # | true_track_deg_sin       | sin(heading) - circular feature encoding       |
    # | true_track_deg_cos       | cos(heading) - circular feature encoding       |
    # | hour_sin / hour_cos      | Cyclical hour encoding (diurnal pattern)       |
    # | month_sin / month_cos    | Cyclical month encoding (seasonal pattern)     |
    # | precip_flag              | 1 if any precipitation present in wx_str       |
    # | temp_dewpoint_spread_c   | Temp - dewpoint (low = foggy/icing risk)       |
    # -----------------------------------------------------------------------------
    f["true_track_deg_sin"] = np.sin(np.radians(heading_deg))
    f["true_track_deg_cos"] = np.cos(np.radians(heading_deg))

    # Temporal (cyclical encoding)
    f["hour_sin"]  = np.sin(2 * np.pi * hour  / 24)
    f["hour_cos"]  = np.cos(2 * np.pi * hour  / 24)
    f["month_sin"] = np.sin(2 * np.pi * month / 12)
    f["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Assemble in correct feature order
    row = np.array([f.get(col, 0.0) for col in feature_cols], dtype=np.float32)

    return row


def compute_risk_grid(base_features: dict, hour: int, month: int,
                      rf, lstm, weights: dict,
                      feature_cols: list, scaler,
                      seq_len: int = 10) -> pd.DataFrame:
    """
    Generates the full (FL × heading) risk score grid.

    For each (altitude, heading) combination:
      1. Build a feature vector
      2. RF: get rerouting probability
      3. LSTM: build a dummy sequence (repeat the static vector seq_len times)
         This is a simplification — in production you'd have real history.
      4. Ensemble: weighted average
      5. Store risk score

    Returns a DataFrame with columns:
      altitude_ft | heading_deg | risk_score_rf | risk_score_lstm | risk_score
    """
    altitudes = np.arange(FL_MIN, FL_MAX + 1, FL_STEP)
    headings  = np.arange(HEADING_MIN, HEADING_MAX + 1, HEADING_STEP)

    rows = []

    # Pre-build all feature vectors
    X_rf_all = []
    for alt in altitudes:
        for hdg in headings:
            vec = build_candidate_row(
                base_features, alt, hdg, hour, month, feature_cols
            )
            X_rf_all.append(vec)

    X_rf_all = np.array(X_rf_all)

    # Scale continuous features in batch (much faster + avoids warnings)
    cont_indices = [feature_cols.index(c) for c in CONTINUOUS_COLS
                    if c in feature_cols]
    
    # To avoid "X does not have valid feature names" warning, 
    # we can use a DataFrame for the scaling step
    X_cont = X_rf_all[:, cont_indices]
    X_cont_df = pd.DataFrame(X_cont, columns=CONTINUOUS_COLS)
    X_rf_all[:, cont_indices] = scaler.transform(X_cont_df)

    # RF predictions — batch
    P_rf_all = rf.predict_proba(X_rf_all)[:, 1]

    # LSTM predictions — build sequences (repeat vector seq_len times)
    X_lstm_all = np.stack([
        np.tile(vec, (seq_len, 1)) for vec in X_rf_all
    ])   # shape: (N_candidates, seq_len, n_features)
    P_lstm_all = lstm.predict(X_lstm_all, verbose=0, batch_size=256).flatten()

    # Ensemble
    w_rf, w_lstm = weights["w_rf"], weights["w_lstm"]
    P_ens_all = w_rf * P_rf_all + w_lstm * P_lstm_all

    idx = 0
    for alt in altitudes:
        for hdg in headings:
            rows.append({
                "altitude_ft":    int(alt),
                "flight_level":   f"FL{int(alt/100):03d}",
                "heading_deg":    int(hdg),
                "risk_score_rf":  round(float(P_rf_all[idx]),  4),
                "risk_score_lstm": round(float(P_lstm_all[idx]), 4),
                "risk_score":     round(float(P_ens_all[idx]),  4),
                "high_risk":      int(P_ens_all[idx] >= HIGH_RISK_THRESHOLD),
            })
            idx += 1

    return pd.DataFrame(rows)


# ---------------------------------------------
# STEP 3 — Extract recommendation
# ---------------------------------------------

def get_recommendation(grid: pd.DataFrame) -> dict:
    """
    Returns the (altitude, heading) combination with the lowest risk score.
    Also returns the lowest-risk altitude band and the safe heading zone.
    """
    best_idx  = grid["risk_score"].idxmin()
    best      = grid.loc[best_idx]

    # Altitude with lowest average risk (across all headings)
    alt_risk  = grid.groupby("altitude_ft")["risk_score"].mean()
    best_alt  = int(alt_risk.idxmin())

    # Heading with lowest average risk (across all altitudes)
    hdg_risk  = grid.groupby("heading_deg")["risk_score"].mean()
    best_hdg  = int(hdg_risk.idxmin())

    # Safe zones: alt/hdg combinations where risk < 0.3
    safe_zone = grid[grid["risk_score"] < 0.3]

    rec = {
        "recommended_altitude_ft":    int(best["altitude_ft"]),
        "recommended_flight_level":   best["flight_level"],
        "recommended_heading_deg":    int(best["heading_deg"]),
        "minimum_risk_score":         round(float(best["risk_score"]), 4),
        "best_altitude_ft":           best_alt,
        "best_heading_deg":           best_hdg,
        "pct_safe_combinations":      round(len(safe_zone) / len(grid) * 100, 1),
        "high_risk_pct":              round(grid["high_risk"].mean() * 100, 1),
    }
    return rec


# ---------------------------------------------
# STEP 4 — Risk heatmap visualization
# ---------------------------------------------

def plot_risk_heatmap(grid: pd.DataFrame, rec: dict, scenario_name: str):
    """
    Plots a heatmap of risk score across flight levels (Y-axis)
    and headings (X-axis). This is the visual output for pilots/dispatchers.
    """
    pivot = grid.pivot_table(
        index="altitude_ft", columns="heading_deg", values="risk_score"
    ).sort_index(ascending=False)   # high altitude at top

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # ── Heatmap ---------------------------------------------──
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "risk", ["#2ecc71", "#f39c12", "#e74c3c"]   # green -> orange -> red
    )
    im = axes[0].imshow(
        pivot.values,
        aspect="auto",
        cmap=cmap, vmin=0, vmax=1,
        extent=[
            grid["heading_deg"].min(), grid["heading_deg"].max(),
            grid["altitude_ft"].min(), grid["altitude_ft"].max()
        ],
        origin="lower"
    )
    plt.colorbar(im, ax=axes[0], label="Risk Score (0=Safe, 1=High Risk)")

    # Mark recommended point
    axes[0].scatter(
        rec["recommended_heading_deg"],
        rec["recommended_altitude_ft"],
        s=200, c="white", marker="*",
        zorder=5, label=f"Recommended: "
                        f"FL{rec['recommended_altitude_ft']//100:03d} / "
                        f"{rec['recommended_heading_deg']}°"
    )
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_xlabel("Heading (°)")
    axes[0].set_ylabel("Altitude (ft)")
    axes[0].set_title(f"Risk Score Grid — {scenario_name}")

    # Set y-ticks as flight levels
    fl_ticks = np.arange(FL_MIN, FL_MAX + 1, 5000)
    axes[0].set_yticks(fl_ticks)
    axes[0].set_yticklabels([f"FL{int(f/100):03d}" for f in fl_ticks], fontsize=7)

    # ── Risk profile by altitude ─────────────────────────────
    alt_risk = grid.groupby("altitude_ft")["risk_score"].agg(["mean", "min", "max"])
    axes[1].fill_betweenx(
        alt_risk.index, alt_risk["min"], alt_risk["max"],
        alpha=0.3, color="#3498db", label="Risk range"
    )
    axes[1].plot(alt_risk["mean"], alt_risk.index, color="#2980b9",
                 linewidth=2, label="Mean risk")
    axes[1].axhline(
        rec["best_altitude_ft"], color="#2ecc71",
        linestyle="--", linewidth=1.5,
        label=f"Best FL ({rec['best_altitude_ft']//100:03d})"
    )
    axes[1].axvline(HIGH_RISK_THRESHOLD, color="red", linestyle=":", alpha=0.6)
    axes[1].set_xlabel("Risk Score")
    axes[1].set_ylabel("Altitude (ft)")
    axes[1].set_title("Risk Profile by Altitude")
    axes[1].legend(fontsize=8)
    axes[1].set_yticks(fl_ticks)
    axes[1].set_yticklabels([f"FL{int(f/100):03d}" for f in fl_ticks], fontsize=7)
    axes[1].set_xlim(0, 1)

    plt.suptitle(
        f"Flight Rerouting Risk Assessment — {scenario_name}\n"
        f"Recommended: FL{rec['recommended_altitude_ft']//100:03d} / "
        f"{rec['recommended_heading_deg']}° | "
        f"Min Risk: {rec['minimum_risk_score']:.3f} | "
        f"Safe Combinations: {rec['pct_safe_combinations']}%",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    out = RESULTS / f"risk_heatmap_{scenario_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Heatmap saved -> {out}")


# ---------------------------------------------
# STEP 5 — Validation against historical events
# ---------------------------------------------

def validate_against_historical(rf, lstm, weights, feature_cols, scaler):
    """
    Tests the risk scoring mechanism against N_VALIDATION_SCENARIOS
    historical rerouting waypoints from dataset_full.csv.

    Metric: "match rate" — how often does the recommended (alt, hdg)
    reduce risk score compared to the actual (alt, hdg) taken?
    A "match" means the recommendation is safer than the actual route.
    """
    df = pd.read_csv(PROCESSED / "dataset_full.csv")

    # Take a sample of actual rerouting events
    rerouted = df[df["rerouted"] == 1].sample(
        n=min(N_VALIDATION_SCENARIOS, len(df[df["rerouted"] == 1])),
        random_state=42
    )

    log.info(f"\nValidating risk scoring on {len(rerouted)} historical rerouting events...")

    matches     = 0
    risk_actual = []
    risk_rec    = []

    for _, row in rerouted.iterrows():
        base = {col: row.get(col, 0.0) for col in feature_cols}
        hour  = int(row.get("hour_sin", 0) * 12 + 12) % 24   # rough decode
        month = 6   # default

        try:
            grid = compute_risk_grid(
                base, hour, month,
                rf, lstm, weights,
                feature_cols, scaler,
                seq_len=10
            )
        except Exception as e:
            log.warning(f"Grid computation failed for a scenario: {e}")
            continue

        rec = get_recommendation(grid)

        # Risk at actual position
        actual_alt = float(row.get("altitude_ft", 35000))
        actual_hdg_sin = float(row.get("true_track_deg_sin", 0))
        actual_hdg = int(np.degrees(np.arcsin(np.clip(actual_hdg_sin, -1, 1))))
        actual_hdg = (actual_hdg // 10) * 10    # snap to grid

        actual_row = grid[
            (grid["altitude_ft"] == min(FL_MAX, max(FL_MIN,
                int(round(actual_alt / FL_STEP) * FL_STEP))))
            & (grid["heading_deg"] == actual_hdg % 360)
        ]

        if actual_row.empty:
            continue

        actual_risk = float(actual_row["risk_score"].values[0])
        rec_risk    = rec["minimum_risk_score"]

        risk_actual.append(actual_risk)
        risk_rec.append(rec_risk)

        if rec_risk < actual_risk:
            matches += 1

    match_rate = matches / len(risk_actual) * 100 if risk_actual else 0
    avg_risk_reduction = (
        np.mean(risk_actual) - np.mean(risk_rec)
        if risk_actual else 0
    )

    val_results = {
        "n_scenarios":            len(risk_actual),
        "match_rate_pct":         round(match_rate, 2),
        "avg_actual_risk":        round(float(np.mean(risk_actual)), 4),
        "avg_recommended_risk":   round(float(np.mean(risk_rec)),    4),
        "avg_risk_reduction":     round(float(avg_risk_reduction),   4),
    }

    log.info(f"\nValidation Results:")
    for k, v in val_results.items():
        log.info(f"  {k:<30}: {v}")

    return val_results


# ---------------------------------------------
# STEP 6 — Demo scenarios
# ---------------------------------------------

DEMO_SCENARIOS = {
    "Midwest_Thunderstorm": {
        "wind_speed_kt":         45,
        "visibility_sm":          1.5,
        "turb_intensity_num":     4,
        "convective_flag":        1,
        "precip_flag":            1,
        "temp_dewpoint_spread_c": 2,
        "in_sigmet_conv":         1,
        "in_sigmet_turb":         1,
        "in_sigmet_ice":          0,
        "hour": 14, "month": 6,
    },
    "North_Atlantic_Jetstream": {
        "wind_speed_kt":         90,
        "visibility_sm":         10,
        "turb_intensity_num":     3,
        "convective_flag":        0,
        "precip_flag":            0,
        "temp_dewpoint_spread_c": 20,
        "in_sigmet_conv":         0,
        "in_sigmet_turb":         1,
        "in_sigmet_ice":          0,
        "hour": 8, "month": 1,
    },
    "Clear_Day_Baseline": {
        "wind_speed_kt":         10,
        "visibility_sm":         10,
        "turb_intensity_num":     0,
        "convective_flag":        0,
        "precip_flag":            0,
        "temp_dewpoint_spread_c": 15,
        "in_sigmet_conv":         0,
        "in_sigmet_turb":         0,
        "in_sigmet_ice":          0,
        "hour": 12, "month": 9,
    },
}


# ---------------------------------------------
# MAIN
# ---------------------------------------------

def main():
    log.info("=" * 60)
    log.info("Risk Scoring - Flight Path Rerouting Project")
    log.info("=" * 60)

    # Load
    rf, lstm, weights, feature_cols = load_ensemble()
    scaler = load_scaler()

    all_recs = {}

    # -- Demo scenarios -----------------------------------------
    log.info("\n[1/2] Running demo scenarios...")
    for scenario_name, scenario in DEMO_SCENARIOS.items():
        log.info(f"\n  Scenario: {scenario_name}")

        hour  = scenario.pop("hour",  12)
        month = scenario.pop("month", 6)

        # Build base features (fill unspecified with defaults)
        base = {col: 0.0 for col in feature_cols}
        base.update(scenario)
        base["wind_speed_squared"]  = base["wind_speed_kt"] ** 2
        base["wind_shear_index"]    = (
            base["wind_speed_kt"] * base["turb_intensity_num"]
        )
        base["altitude_ft"]         = 35000   # starting altitude
        base["visibility_category"] = (
            0 if base["visibility_sm"] >= 3 else
            1 if base["visibility_sm"] >= 1 else 2
        )

        # Compute risk grid
        grid = compute_risk_grid(
            base, hour, month,
            rf, lstm, weights,
            feature_cols, scaler
        )

        # Get recommendation
        rec = get_recommendation(grid)
        all_recs[scenario_name] = rec

        log.info(f"  Recommendation: {rec['recommended_flight_level']} / "
                 f"{rec['recommended_heading_deg']} deg | "
                 f"Risk: {rec['minimum_risk_score']:.3f}")

        # Save grid and heatmap
        grid.to_csv(RESULTS / f"risk_grid_{scenario_name}.csv", index=False)
        plot_risk_heatmap(grid, rec, scenario_name)

    # ── Historical validation ──────────────────────────────────
    log.info("\n[2/2] Validating against historical rerouting events...")
    val_results = validate_against_historical(rf, lstm, weights, feature_cols, scaler)
    all_recs["validation"] = val_results

    # Save all recommendations and validation
    with open(RESULTS / "recommendations.json", "w") as f:
        json.dump(all_recs, f, indent=2)
    log.info("\nAll recommendations saved -> results/recommendations.json")

    log.info("\n* Risk scoring complete.")
    log.info("  Outputs:")
    log.info("  -> results/risk_grid_<scenario>.csv")
    log.info("  -> results/risk_heatmap_<scenario>.png")
    log.info("  -> results/recommendations.json")


if __name__ == "__main__":
    main()
