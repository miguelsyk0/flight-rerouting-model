"""
08_app.py
=========
Streamlit web application for real-time flight rerouting risk assessment.

Run with:
    streamlit run 08_app.py

What it does:
  1. User enters a flight ICAO24 address or callsign
  2. App fetches live position from OpenSky /states/all
  3. App fetches current weather from NOAA AWC for that position
  4. Ensemble model (RF + LSTM) scores rerouting risk
  5. App displays risk score, heatmap, and recommended FL + heading

Alternatively, user can enter weather conditions manually (manual mode)
for scenario testing without needing a live flight.
"""

import json
import pickle
import logging
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import requests
from pathlib import Path
from datetime import datetime, timezone, timedelta
from math import radians, sin, cos, sqrt, atan2, degrees, asin

import tensorflow as tf

# ─────────────────────────────────────────────
# PAGE CONFIG — must be first Streamlit call
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Flight Rerouting Risk Assessment",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# PATHS & CONSTANTS
# ─────────────────────────────────────────────

MODELS_DIR = Path("models")
PROCESSED  = Path("data/processed")

BASE_OPENSKY = "https://opensky-network.org/api"
TOKEN_URL    = (
    "https://auth.opensky-network.org/auth/realms/"
    "opensky-network/protocol/openid-connect/token"
)
BASE_AWC     = "https://aviationweather.gov/api/data"

FL_MIN, FL_MAX, FL_STEP       = 10000, 41000, 1000
HEADING_MIN, HEADING_MAX      = 0, 350
HEADING_STEP                  = 10
HIGH_RISK_THRESHOLD           = 0.5

CONTINUOUS_COLS = [
    "wind_speed_kt", "visibility_sm",
    "turb_intensity_num", "temp_dewpoint_spread_c",
    "altitude_ft"
]

AIRPORT_COORDS = {
    "KJFK": (40.6413, -73.7781), "KORD": (41.9742, -87.9073),
    "KATL": (33.6367, -84.4281), "KDFW": (32.8998, -97.0403),
    "KLAX": (33.9425, -118.4081), "EGLL": (51.4775, -0.4614),
    "KEWR": (40.6925, -74.1687), "KLGA": (40.7772, -73.8726),
    "KMDW": (41.7868, -87.7522), "KBOS": (42.3643, -71.0052),
    "KPHL": (39.8719, -75.2411), "KDEN": (39.8561, -104.6737),
    "KSLC": (40.7884, -111.9778), "KSFO": (37.6213, -122.3790),
    "KOAK": (37.7213, -122.2208),
}


# ─────────────────────────────────────────────
# MODEL LOADING (cached — only loads once)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading models...")
def load_models():
    """Loads RF, LSTM, scaler, weights, and feature columns."""
    missing = []
    for p in ["rf_model.pkl", "lstm_model.keras",
              "ensemble_weights.json"]:
        if not (MODELS_DIR / p).exists():
            missing.append(p)
    if not (PROCESSED / "feature_cols.json").exists():
        missing.append("data/processed/feature_cols.json")
    if not (PROCESSED / "scaler.pkl").exists():
        missing.append("data/processed/scaler.pkl")

    if missing:
        return None, None, None, None, None, missing

    with open(MODELS_DIR / "rf_model.pkl", "rb") as f:
        rf = pickle.load(f)
    lstm = tf.keras.models.load_model(MODELS_DIR / "lstm_model.keras")
    with open(MODELS_DIR / "ensemble_weights.json") as f:
        weights = json.load(f)
    with open(PROCESSED / "feature_cols.json") as f:
        feature_cols = json.load(f)
    with open(PROCESSED / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return rf, lstm, weights, feature_cols, scaler, []


# ─────────────────────────────────────────────
# OPENSKY — live flight lookup
# ─────────────────────────────────────────────

def get_opensky_token(client_id: str, client_secret: str) -> str | None:
    try:
        resp = requests.post(
            TOKEN_URL,
            data={
                "grant_type":    "client_credentials",
                "client_id":     client_id,
                "client_secret": client_secret,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("access_token")
    except Exception:
        return None


def fetch_flight_state(query: str, token: str) -> dict | None:
    """
    Looks up a flight by ICAO24 or callsign from /states/all.
    Returns a state dict or None if not found.
    """
    headers = {"Authorization": f"Bearer {token}"}

    # Try ICAO24 first (exact match)
    params = {"icao24": query.lower().strip()}
    try:
        resp = requests.get(
            f"{BASE_OPENSKY}/states/all",
            params=params, headers=headers, timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("states"):
            return _parse_state(data["states"][0])
    except Exception:
        pass

    # Try callsign via global query + filter
    try:
        resp = requests.get(
            f"{BASE_OPENSKY}/states/all",
            headers=headers, timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("states"):
            callsign_query = query.upper().strip()
            for sv in data["states"]:
                cs = (sv[1] or "").strip().upper()
                if cs == callsign_query:
                    return _parse_state(sv)
    except Exception:
        pass

    return None


def _parse_state(sv: list) -> dict:
    """Maps raw OpenSky state vector array to named fields."""
    return {
        "icao24":          sv[0],
        "callsign":        (sv[1] or "").strip(),
        "origin_country":  sv[2],
        "latitude":        sv[6],
        "longitude":       sv[5],
        "baro_altitude_m": sv[7],
        "baro_altitude_ft": round(sv[7] * 3.28084) if sv[7] else None,
        "on_ground":       sv[8],
        "velocity_ms":     sv[9],
        "velocity_kt":     round(sv[9] * 1.94384) if sv[9] else None,
        "true_track_deg":  sv[10],
        "vertical_rate":   sv[11],
        "squawk":          sv[14],
    }


# ─────────────────────────────────────────────
# NOAA AWC — weather for a position
# ─────────────────────────────────────────────

def haversine_nm(lat1, lon1, lat2, lon2) -> float:
    R = 3440.065
    φ1, φ2 = radians(lat1), radians(lat2)
    dφ, dλ = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dφ/2)**2 + cos(φ1)*cos(φ2)*sin(dλ/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


def get_dynamic_nearest_station(lat: float, lon: float) -> str:
    """
    Queries NOAA for stations in a 1.0 degree bbox and returns the closest.
    """
    headers = {"User-Agent": "FlightReroutingResearch/1.0"}
    bbox = f"{lat-0.5},{lon-0.5},{lat+0.5},{lon+0.5}"
    try:
        resp = requests.get(
            f"{BASE_AWC}/metar",
            params={"bbox": bbox, "format": "json", "hours": 1},
            headers=headers, timeout=5
        )
        if resp.ok:
            records = resp.json()
            if records:
                best_id, best_d = None, float("inf")
                for m in records:
                    s_id = m.get("icaoId")
                    s_lat = float(m.get("lat") or 0)
                    s_lon = float(m.get("lon") or 0)
                    d = haversine_nm(lat, lon, s_lat, s_lon)
                    if d < best_d:
                        best_d, best_id = d, s_id
                return best_id
    except Exception:
        pass
    return "KATL" # Fallback


def fetch_weather_for_position(lat: float, lon: float) -> dict:
    """
    Fetches current weather for a lat/lon by querying the nearest
    METAR station and checking for active SIGMETs.

    Returns a flat weather dict ready for feature engineering.
    """
    station = get_dynamic_nearest_station(lat, lon)
    weather = {
        "station":            station,
        "wind_speed_kt":      None,
        "wind_dir_deg":       None,
        "visibility_sm":      None,
        "temp_c":             None,
        "dewpoint_c":         None,
        "wx_string":          "",
        "convective_flag":    0,
        "precip_flag":        0,
        "turb_intensity_num": 0.0,
        "in_sigmet_conv":     0,
        "in_sigmet_turb":     0,
        "in_sigmet_ice":      0,
    }

    headers = {
        "User-Agent": "FlightReroutingResearch/1.0 (academic; saint-louis-university)"
    }

    # ── METAR ───────────────────────────────────────────────────
    try:
        resp = requests.get(
            f"{BASE_AWC}/metar",
            params={"ids": station, "format": "json", "hours": 1},
            headers=headers, timeout=10
        )
        if resp.ok:
            records = resp.json()
            if records:
                m = records[0]
                # Align with current AWC JSON schema
                val_wspd = m.get("wspd")
                if val_wspd is not None: weather["wind_speed_kt"] = float(val_wspd)
                
                val_wdir = m.get("wdir")
                if val_wdir is not None: weather["wind_dir_deg"] = float(val_wdir)
                
                val_vis = m.get("visib")
                if val_vis is not None: weather["visibility_sm"] = float(val_vis)
                
                val_temp = m.get("temp")
                if val_temp is not None: weather["temp_c"] = float(val_temp)
                
                val_dewp = m.get("dewp")
                if val_dewp is not None: weather["dewpoint_c"] = float(val_dewp)
                
                weather["wx_string"] = str(m.get("wxString") or "")
    except Exception:
        pass

    # ── Convective / precip flags from wx_string ────────────────
    wx = weather["wx_string"].upper()
    weather["convective_flag"] = int(any(c in wx for c in ["TS","SQ","FC","GR","VCTS"]))
    weather["precip_flag"]     = int(any(c in wx for c in ["RA","SN","DZ","SH","TS"]))

    # ── PIREP (turbulence near position) ────────────────────────
    try:
        # Use a 2.0 degree bbox for a wider turbulence search
        bbox = f"{lat-1.0},{lon-1.0},{lat+1.0},{lon+1.0}"
        resp = requests.get(
            f"{BASE_AWC}/pirep",
            params={
                "bbox":     bbox,
                "format":   "json",
                "hours":    2,
            },
            headers=headers, timeout=10
        )
        if resp.ok:
            pireps = resp.json() or []
            turb_map = {
                "NEG":0,"SMTH":0,"SMTH-LGT":1,"LGT":1,
                "LGT-MOD":2,"MOD":3,"MOD-SEV":4,"SEV":5,"EXTM":6,"EXTRM":6
            }
            max_turb = 0
            for p in pireps:
                # Align with current AWC PIREP schema (tbInt1)
                ti = str(p.get("tbInt1") or p.get("turbulence_intensity","")).upper().strip()
                max_turb = max(max_turb, turb_map.get(ti, 0))
            weather["turb_intensity_num"] = float(max_turb)
    except Exception:
        pass

    # ── SIGMET check ────────────────────────────────────────────
    try:
        from shapely.geometry import Point, shape
        resp = requests.get(
            f"{BASE_AWC}/sigmet",
            params={"format": "geojson"},
            headers=headers, timeout=10
        )
        if resp.ok:
            features = resp.json().get("features", [])
            pt = Point(lon, lat)
            # Use a 0.2 degree buffer (~12nm) for "vicinity" hazards
            vicinity = pt.buffer(0.2) 
            for feat in features:
                try:
                    geom = shape(feat["geometry"])
                    if geom.intersects(vicinity):
                        h = str(feat.get("properties", {}).get("hazard","")).upper()
                        if "CONV" in h: weather["in_sigmet_conv"] = 1
                        if "TURB" in h: weather["in_sigmet_turb"] = 1
                        if "ICE"  in h: weather["in_sigmet_ice"]  = 1
                except Exception:
                    pass
    except Exception:
        pass

    return weather


# ─────────────────────────────────────────────
# FEATURE BUILDER
# ─────────────────────────────────────────────

def build_features(weather: dict, altitude_ft: float,
                   heading_deg: float, hour: int, month: int,
                   feature_cols: list, scaler) -> np.ndarray:
    """Builds and scales a single feature vector."""
    wind_kt = weather.get("wind_speed_kt")
    if wind_kt is None: wind_kt = 0.0
    
    turb    = weather.get("turb_intensity_num", 0)
    
    vis     = weather.get("visibility_sm")
    if vis is None: vis = 10.0

    vis_cat = 0 if vis >= 3 else 1 if vis >= 1 else 2
    
    # ── Altitude-Aware Temperature Fallback ──────────────────
    temp = weather.get("temp_c")
    if temp is None:
        # Standard lapse rate: 15C at sea level, -2C per 1000ft
        temp = 15.0 - (altitude_ft / 1000) * 1.98
    
    dewp = weather.get("dewpoint_c")
    if dewp is None:
        # Standard spread of 10C for 'clear' fallback
        dewp = temp - 10.0
        
    spread  = max(0, temp - dewp)

    f = {
        "wind_speed_kt":          wind_kt,
        "visibility_sm":          vis,
        "turb_intensity_num":     turb,
        "convective_flag":        weather.get("convective_flag", 0),
        "precip_flag":            weather.get("precip_flag", 0),
        "temp_dewpoint_spread_c": spread,
        "in_sigmet_conv":         weather.get("in_sigmet_conv", 0),
        "in_sigmet_turb":         weather.get("in_sigmet_turb", 0),
        "in_sigmet_ice":          weather.get("in_sigmet_ice",  0),
        "altitude_ft":            altitude_ft,
        "true_track_deg_sin":     np.sin(np.radians(heading_deg)),
        "true_track_deg_cos":     np.cos(np.radians(heading_deg)),
        "hour_sin":               np.sin(2 * np.pi * hour  / 24),
        "hour_cos":               np.cos(2 * np.pi * hour  / 24),
        "month_sin":              np.sin(2 * np.pi * month / 12),
        "month_cos":              np.cos(2 * np.pi * month / 12),
    }

    # Convert to DataFrame to maintain feature names and silence scaler warnings
    df_row = pd.DataFrame([f], columns=feature_cols)
    cont_idx = [c for c in CONTINUOUS_COLS if c in feature_cols]
    df_row[cont_idx] = scaler.transform(df_row[cont_idx])
    
    return df_row.values.flatten()


# ─────────────────────────────────────────────
# RISK GRID
# ─────────────────────────────────────────────

def altitude_adjust(weather: dict, altitude_ft: float) -> dict:
    w = weather.copy()
    wind = weather.get("wind_speed_kt")
    if wind is None: wind = 0.0
    
    turb = weather.get("turb_intensity_num")
    if turb is None: turb = 0.0
    
    fl   = altitude_ft / 1000

    wind_factor = (1.0 + (fl/350)*0.8) if fl <= 350 else (1.8 - (fl-350)/600)
    w["wind_speed_kt"] = wind * max(wind_factor, 0.5)

    # Removed aggressive altitude-based hazard zeroing to respect manual user inputs.
    # The model will judge risk based on the flags actually present.
    w["precip_flag"]     = weather.get("precip_flag", 0)
    w["convective_flag"] = weather.get("convective_flag", 0)

    if 18000 <= altitude_ft <= 28000:
        w["turb_intensity_num"] = turb * 1.4
    elif altitude_ft >= 35000:
        w["turb_intensity_num"] = turb * 1.2
    elif altitude_ft < 10000:
        w["turb_intensity_num"] = turb * 0.7

    if altitude_ft > 15000:
        vis = weather.get("visibility_sm")
        if vis is None: vis = 10.0
        w["visibility_sm"] = max(vis, 10.0)

    return w


@st.cache_data(show_spinner=False, ttl=300)
def compute_risk_grid_cached(
    weather_json: str,
    hour: int, month: int,
    _rf, _lstm, weights_json: str,
    feature_cols_json: str, _scaler
) -> pd.DataFrame:
    """
    Cached wrapper around the risk grid computation.
    TTL=300 means results are cached for 5 minutes — avoids
    rerunning the full grid on every Streamlit re-render.
    """
    weather      = json.loads(weather_json)
    weights      = json.loads(weights_json)
    feature_cols = json.loads(feature_cols_json)
    SEQ_LEN      = 10

    altitudes = np.arange(FL_MIN, FL_MAX + 1, FL_STEP)
    headings  = np.arange(HEADING_MIN, HEADING_MAX + 1, HEADING_STEP)

    X_all = []
    for alt in altitudes:
        for hdg in headings:
            adj = altitude_adjust(weather, alt)
            vec = build_features(adj, alt, hdg, hour, month, feature_cols, _scaler)
            X_all.append(vec)

    X_all = np.array(X_all, dtype=np.float32)

    P_rf   = _rf.predict_proba(X_all)[:, 1]
    X_lstm = np.stack([np.tile(v, (SEQ_LEN, 1)) for v in X_all])
    P_lstm = _lstm.predict(X_lstm, verbose=0, batch_size=256).flatten()
    P_ens  = weights["w_rf"] * P_rf + weights["w_lstm"] * P_lstm

    rows, idx = [], 0
    for alt in altitudes:
        for hdg in headings:
            rows.append({
                "altitude_ft":  int(alt),
                "flight_level": f"FL{int(alt/100):03d}",
                "heading_deg":  int(hdg),
                "risk_rf":      round(float(P_rf[idx]),   4),
                "risk_lstm":    round(float(P_lstm[idx]), 4),
                "risk_score":   round(float(P_ens[idx]),  4),
                "high_risk":    int(P_ens[idx] >= HIGH_RISK_THRESHOLD),
            })
            idx += 1

    return pd.DataFrame(rows)


def get_recommendation(grid: pd.DataFrame,
                       current_alt: float = 35000,
                       current_hdg: float = 0) -> dict:
    best_idx = grid["risk_score"].idxmin()
    best     = grid.loc[best_idx]
    alt_risk = grid.groupby("altitude_ft")["risk_score"].mean()
    hdg_risk = grid.groupby("heading_deg")["risk_score"].mean()
    safe_pct = round(len(grid[grid["risk_score"] < 0.3]) / len(grid) * 100, 1)

    # Risk at the CURRENT position (snap to nearest grid cell)
    snap_alt = min(FL_MAX, max(FL_MIN, int(round(current_alt / FL_STEP) * FL_STEP)))
    snap_hdg = (int(current_hdg) // HEADING_STEP) * HEADING_STEP % 360
    cur_row  = grid[
        (grid["altitude_ft"] == snap_alt) &
        (grid["heading_deg"] == snap_hdg)
    ]
    current_risk = float(cur_row["risk_score"].values[0]) if not cur_row.empty else float(best["risk_score"])

    return {
        "altitude_ft":    int(best["altitude_ft"]),
        "flight_level":   best["flight_level"],
        "heading_deg":    int(best["heading_deg"]),
        "risk_score":     round(float(best["risk_score"]), 4),
        "current_risk":   round(current_risk, 4),
        "best_altitude":  int(alt_risk.idxmin()),
        "best_heading":   int(hdg_risk.idxmin()),
        "safe_pct":       safe_pct,
        "high_risk_pct":  round(grid["high_risk"].mean() * 100, 1),
    }


# ─────────────────────────────────────────────
# HEATMAP PLOT
# ─────────────────────────────────────────────

def plot_heatmap(grid: pd.DataFrame, rec: dict,
                 current_alt: float, current_hdg: float):
    pivot = grid.pivot_table(
        index="altitude_ft", columns="heading_deg", values="risk_score"
    ).sort_index(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                             facecolor="#0e1117")
    for ax in axes:
        ax.set_facecolor("#0e1117")

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "risk", ["#2ecc71", "#f39c12", "#e74c3c"]
    )

    # ── Heatmap ───────────────────────────────────────────────
    im = axes[0].imshow(
        pivot.values, aspect="auto",
        cmap=cmap, vmin=0, vmax=1,
        extent=[0, 350, FL_MIN, FL_MAX],
        origin="lower"
    )
    cbar = plt.colorbar(im, ax=axes[0])
    cbar.set_label("Risk Score", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # Current position marker
    snap_alt = min(FL_MAX, max(FL_MIN,
        int(round(current_alt / FL_STEP) * FL_STEP)))
    axes[0].scatter(
        current_hdg, snap_alt,
        s=180, c="#3498db", marker="o",
        zorder=5, label="Current position", edgecolors="white"
    )
    # Recommended position marker
    axes[0].scatter(
        rec["heading_deg"], rec["altitude_ft"],
        s=220, c="white", marker="*",
        zorder=6, label=f"Recommended: {rec['flight_level']} / {rec['heading_deg']}°"
    )
    axes[0].legend(loc="upper right", fontsize=8,
                   facecolor="#1e2130", labelcolor="white")
    axes[0].set_xlabel("Heading (°)", color="white")
    axes[0].set_ylabel("Altitude (ft)", color="white")
    axes[0].set_title("Risk Grid — FL × Heading", color="white", fontsize=12)
    axes[0].tick_params(colors="white")

    fl_ticks = np.arange(FL_MIN, FL_MAX + 1, 5000)
    axes[0].set_yticks(fl_ticks)
    axes[0].set_yticklabels(
        [f"FL{int(f/100):03d}" for f in fl_ticks], fontsize=7, color="white"
    )
    axes[0].set_xticks(range(0, 360, 30))
    axes[0].tick_params(colors="white")

    # ── Altitude risk profile ────────────────────────────────
    alt_risk = grid.groupby("altitude_ft")["risk_score"].agg(["mean","min","max"])
    axes[1].fill_betweenx(
        alt_risk.index, alt_risk["min"], alt_risk["max"],
        alpha=0.25, color="#3498db"
    )
    axes[1].plot(alt_risk["mean"], alt_risk.index,
                 color="#3498db", linewidth=2, label="Mean risk")
    axes[1].axhline(rec["best_altitude"], color="#2ecc71",
                    linestyle="--", linewidth=1.5,
                    label=f"Safest FL ({rec['best_altitude']//100:03d})")
    axes[1].axhline(snap_alt, color="#3498db",
                    linestyle=":", linewidth=1.5,
                    label=f"Current FL ({int(snap_alt/100):03d})")
    axes[1].axvline(HIGH_RISK_THRESHOLD, color="#e74c3c",
                    linestyle=":", alpha=0.7, label="High risk threshold")
    axes[1].set_xlabel("Risk Score", color="white")
    axes[1].set_ylabel("Altitude (ft)", color="white")
    axes[1].set_title("Risk Profile by Altitude", color="white", fontsize=12)
    axes[1].set_xlim(0, 1)
    axes[1].set_yticks(fl_ticks)
    axes[1].set_yticklabels(
        [f"FL{int(f/100):03d}" for f in fl_ticks], fontsize=7, color="white"
    )
    axes[1].tick_params(colors="white")
    axes[1].set_facecolor("#0e1117")
    axes[1].legend(fontsize=8, facecolor="#1e2130", labelcolor="white")

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# RISK BADGE HELPER
# ─────────────────────────────────────────────

def risk_badge(score: float) -> tuple[str, str]:
    """Returns (label, color) based on risk score."""
    if score < 0.10:
        return "LOW RISK", "#2ecc71"
    elif score < 0.25:
        return "MODERATE RISK", "#f39c12"
    elif score < 0.50:
        return "HIGH RISK", "#e74c3c"
    else:
        return "CRITICAL RISK", "#c0392b"


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────

def main():
    st.title("✈️ Flight Rerouting Risk Assessment")
    st.markdown(
        "Real-time prediction of flight rerouting probability using "
        "Random Forest + LSTM ensemble trained on OpenSky and NOAA data."
    )

    # ── Load models ───────────────────────────────────────────
    rf, lstm, weights, feature_cols, scaler, missing = load_models()

    if missing:
        st.error(
            f"⚠️ Missing model files: `{'`, `'.join(missing)}`\n\n"
            "Run scripts 04 → 07 first to train the models before using the app."
        )
        st.stop()

    # ── Sidebar — credentials + mode ─────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        mode = st.radio(
            "Input mode",
            ["Live Flight Lookup", "Manual Input"],
            help="Live mode fetches real-time flight + weather data. "
                 "Manual mode lets you enter conditions directly."
        )

        st.divider()
        st.subheader("OpenSky Credentials")
        st.caption("Required for Live Flight Lookup. "
                   "Get yours at opensky-network.org → Account → API Clients")
        client_id     = st.text_input("Client ID",     type="default")
        client_secret = st.text_input("Client Secret", type="password")

        st.divider()
        st.caption(
            "**Model:** RF + LSTM Ensemble\n\n"
            f"**Weights:** RF={weights['w_rf']}, LSTM={weights['w_lstm']}\n\n"
            "**Threshold:** 0.5"
        )

    # ═══════════════════════════════════════════════════════════
    # MODE 1 — LIVE FLIGHT LOOKUP
    # ═══════════════════════════════════════════════════════════

    if "Live Flight" in mode:
        st.subheader("Live Flight Lookup")

        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input(
                "Enter ICAO24 address or callsign",
                placeholder="e.g. a0f3d2 or UAL123",
                help="ICAO24 is the hex transponder code. "
                     "Callsign is the flight number (e.g. UAL123, PAL203)."
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            search = st.button("Search Flight", type="primary", width="stretch")

        if search and query:
            if not client_id or not client_secret:
                st.warning("Enter your OpenSky credentials in the sidebar first.")
                st.stop()

            with st.spinner("Authenticating with OpenSky..."):
                token = get_opensky_token(client_id, client_secret)

            if not token:
                st.error("Authentication failed. Check your Client ID and Secret.")
                st.stop()

            with st.spinner(f"Looking up flight: {query.upper()}..."):
                state = fetch_flight_state(query, token)

            if not state:
                st.error(
                    f"Flight `{query.upper()}` not found in current OpenSky data. "
                    "The aircraft may be on the ground, out of coverage, or the "
                    "identifier may be incorrect."
                )
                st.stop()

            # ── Flight info cards ──────────────────────────────
            st.success(f"Flight found: **{state['callsign'] or state['icao24'].upper()}**")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("ICAO24",   state["icao24"].upper())
            c2.metric("Altitude", f"FL{int((state['baro_altitude_ft'] or 0)/100):03d}"
                      if state["baro_altitude_ft"] else "N/A")
            c3.metric("Heading",  f"{state['true_track_deg']:.0f}°"
                      if state["true_track_deg"] else "N/A")
            c4.metric("Speed",    f"{state['velocity_kt']} kt"
                      if state["velocity_kt"] else "N/A")

            lat  = state["latitude"]
            lon  = state["longitude"]
            alt  = state["baro_altitude_ft"] or 35000
            hdg  = state["true_track_deg"]   or 0

            if not lat or not lon:
                st.error("No position data available for this flight.")
                st.stop()

            # ── Fetch weather ──────────────────────────────────
            with st.spinner("Fetching current weather conditions..."):
                weather = fetch_weather_for_position(lat, lon)

            # ── Run model ──────────────────────────────────────
            now   = datetime.now(timezone.utc)
            hour  = now.hour
            month = now.month

            with st.spinner("Running ensemble model..."):
                grid = compute_risk_grid_cached(
                    json.dumps(weather), hour, month,
                    rf, lstm, json.dumps(weights),
                    json.dumps(feature_cols), scaler
                )

            rec   = get_recommendation(grid, alt, hdg)
            score = rec["current_risk"]

            _render_results(
                weather, state, rec, score, grid, alt, hdg, lat, lon
            )

    # ═══════════════════════════════════════════════════════════
    # MODE 2 — MANUAL INPUT
    # ═══════════════════════════════════════════════════════════

    else:
        st.subheader("Manual Weather and Flight Input")
        st.caption(
            "Enter flight conditions directly. "
            "Useful for testing specific scenarios or when no live flight is available."
        )

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("**Flight Parameters**")
            alt = st.slider("Current Altitude (ft)", 10000, 45000, 35000, 1000)
            hdg = st.slider("Current Heading (°)",   0, 359, 90, 10)

            st.markdown("**Weather Conditions**")
            wind_kt  = st.slider("Wind Speed (kt)",         0, 150, 20)
            vis_sm   = st.slider("Visibility (statute mi)", 0.0, 10.0, 10.0, 0.25)
            temp_c   = st.slider("Temperature (°C)",        -60, 40, 5)
            dew_c    = st.slider("Dewpoint (°C)",           -60, 40, -5)

        with col_r:
            st.markdown("**Turbulence**")
            turb_label = st.select_slider(
                "Turbulence Intensity",
                options=["None (0)", "Light (1)", "Light-Moderate (2)",
                         "Moderate (3)", "Moderate-Severe (4)",
                         "Severe (5)", "Extreme (6)"],
                value="None (0)"
            )
            turb_num = int(turb_label.split("(")[1].rstrip(")"))

            st.markdown("**Active Hazards**")
            conv_sigmet = st.checkbox("Inside Convective SIGMET polygon")
            turb_sigmet = st.checkbox("Inside Turbulence SIGMET polygon")
            ice_sigmet  = st.checkbox("Inside Icing SIGMET polygon")
            has_conv    = st.checkbox("Convective weather (thunderstorms)")
            has_precip  = st.checkbox("Precipitation present")

        run = st.button("Run Risk Assessment", type="primary", width="stretch")

        if run:
            weather = {
                "wind_speed_kt":      float(wind_kt),
                "visibility_sm":      float(vis_sm),
                "temp_c":             float(temp_c),
                "dewpoint_c":         float(dew_c),
                "turb_intensity_num": float(turb_num),
                "convective_flag":    int(has_conv),
                "precip_flag":        int(has_precip),
                "in_sigmet_conv":     int(conv_sigmet),
                "in_sigmet_turb":     int(turb_sigmet),
                "in_sigmet_ice":      int(ice_sigmet),
                "wx_string":          "TS" if has_conv else "",
                "station":            "MANUAL",
            }

            now   = datetime.now(timezone.utc)
            hour  = now.hour
            month = now.month

            with st.spinner("Running ensemble model..."):
                grid = compute_risk_grid_cached(
                    json.dumps(weather), hour, month,
                    rf, lstm, json.dumps(weights),
                    json.dumps(feature_cols), scaler
                )

            rec   = get_recommendation(grid, alt, hdg)
            score = rec["current_risk"]

            _render_results(
                weather, None, rec, score, grid, alt, hdg, None, None
            )


# ─────────────────────────────────────────────
# SHARED RESULTS RENDERER
# ─────────────────────────────────────────────

def _render_results(weather, state, rec, score, grid, alt, hdg, lat, lon):
    st.divider()
    st.subheader("Risk Assessment Results")

    c_score, c_rec, c_weather = st.columns([1, 1, 2])

    with c_score:
        label, color = risk_badge(score)
        st.markdown(
            f"""
            <div style='text-align:center;padding:20px;
                        background:{color}22;border-radius:12px;
                        border:1px solid {color}'>
                <div style='font-size:32px;font-weight:700;color:{color}'>
                    {score:.1%}
                </div>
                <div style='font-size:14px;margin-top:4px;color:{color}'>
                    {label}
                </div>
                <div style='font-size:12px;margin-top:8px;color:#aaa'>
                    Risk at current position
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        # Show recommendation score as a secondary callout
        rec_label, rec_color = risk_badge(rec["risk_score"])
        st.markdown(
            f"""
            <div style='text-align:center;padding:10px;margin-top:8px;
                        background:{rec_color}11;border-radius:8px;
                        border:1px dashed {rec_color}'>
                <div style='font-size:13px;color:#aaa'>Best available route</div>
                <div style='font-size:22px;font-weight:600;color:{rec_color}'>
                    {rec["risk_score"]:.1%}
                </div>
                <div style='font-size:11px;color:#aaa'>
                    {rec["flight_level"]} / {rec["heading_deg"]}&deg;
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c_rec:
        st.markdown("**Recommendation**")
        st.metric("Optimal Flight Level", rec["flight_level"])
        st.metric("Optimal Heading",      f"{rec['heading_deg']}°")
        st.metric("Min Risk Score",        f"{rec['risk_score']:.1%}")
        st.caption(f"Safe combinations: {rec['safe_pct']}% of grid")

    with c_weather:
        st.markdown("**Weather at Position**")
        wc1, wc2 = st.columns(2)
        
        # Safe formatting for metrics that might be None
        w_wind = weather.get("wind_speed_kt") or 0.0
        w_vis  = weather.get("visibility_sm") or 10.0
        
        # Use altitude-aware temp for UI if missing
        w_temp = weather.get("temp_c")
        if w_temp is None:
            w_temp = 15.0 - (alt / 1000) * 1.98

        wc1.metric("Wind Speed",  f"{w_wind:.0f} kt")
        wc1.metric("Visibility",  f"{w_vis:.1f} sm")
        wc1.metric("Temperature", f"{w_temp:.1f} °C")
        
        # Turbulence lookup
        turb_idx = int(weather.get("turb_intensity_num") or 0)
        turb_labels = ["None", "Smth-Lgt", "Light", "Lgt-Mod", "Moderate", "Severe", "Extreme"]
        wc2.metric("Turbulence", turb_labels[min(turb_idx, 6)])
        
        wc2.metric("Convective SIGMET", "Active" if weather.get("in_sigmet_conv") else "Clear")
        wc2.metric("Turbulence SIGMET", "Active" if weather.get("in_sigmet_turb") else "Clear")

        if weather.get("wx_string"):
            st.caption(f"WX: `{weather['wx_string']}`")
        if weather.get("station") and weather["station"] != "MANUAL":
            st.caption(f"Nearest station: {weather['station']}")

    st.divider()

    # ── Individual model scores ───────────────────────────────
    st.markdown("**Model Breakdown**")
    current_alt_snap = min(FL_MAX, max(FL_MIN, int(round(alt / FL_STEP) * FL_STEP)))
    current_hdg_snap = (int(hdg) // HEADING_STEP) * HEADING_STEP % 360

    current_row = grid[
        (grid["altitude_ft"] == current_alt_snap) &
        (grid["heading_deg"] == current_hdg_snap)
    ]

    if not current_row.empty:
        cr = current_row.iloc[0]
        m1, m2, m3 = st.columns(3)
        m1.metric("Random Forest",  f"{cr['risk_rf']:.1%}")
        m2.metric("LSTM",           f"{cr['risk_lstm']:.1%}")
        m3.metric("Ensemble",       f"{cr['risk_score']:.1%}",
                  delta=f"at FL{current_alt_snap//100:03d} / {current_hdg_snap}°",
                  delta_color="off")

    # ── Heatmap ───────────────────────────────────────────────
    st.markdown("**Risk Heatmap** - Drag to explore; * = Recommendation, o = Current position")
    fig = plot_heatmap(grid, rec, alt, hdg)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    # ── High-risk zones ───────────────────────────────────────
    if rec["high_risk_pct"] > 20:
        st.warning(
            f"Caution: {rec['high_risk_pct']}% of the FL/heading grid is high-risk under "
            "current weather conditions. Significant deviations from planned route "
            "are likely. Review the heatmap and consider the recommended altitude."
        )

    # ── Raw grid download ─────────────────────────────────────
    with st.expander("Download full risk grid (CSV)"):
        st.dataframe(
            grid.sort_values("risk_score").head(50),
            width="stretch"
        )
        st.download_button(
            "Download full grid CSV",
            data=grid.to_csv(index=False),
            file_name=f"risk_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    if lat and lon:
        with st.expander("Flight Position Map"):
            st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}),
                   zoom=5, use_container_width=True)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    main()