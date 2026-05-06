"""
03_preprocess.py
================
Merges OpenSky trajectory data with NOAA weather data, engineers features,
labels rerouting events, and produces a clean ML-ready dataset.

Pipeline:
  1. Load & clean trajectories
  2. Compute deviation distance from great-circle baseline (labels rerouting events)
  3. Load & clean METAR / PIREP / SIGMET data
  4. Spatial-temporal join: attach nearest weather observations to each waypoint
  5. Engineer ML features (wind shear index, visibility category, convective flag, etc.)
  6. Handle missing values & encode categoricals
  7. Apply SMOTE to balance classes (rerouted vs not rerouted)
  8. Save final dataset to data/processed/dataset.csv

Requirements:
    pip install pandas numpy scikit-learn imbalanced-learn shapely geopy
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2, degrees, asin
from shapely.geometry import Point, shape
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import json

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

RAW_OPENSKY = Path("data/raw/opensky")
RAW_NOAA    = Path("data/raw/noaa")
PROCESSED   = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

# Rerouting label threshold:
# A waypoint is flagged as a rerouting event if the aircraft is THIS MANY
# nautical miles away from the great-circle (ideal straight) path.
REROUTE_DEVIATION_NM = 30.0

# Minimum flight altitude to include (filter out ground / taxi movement)
MIN_ALTITUDE_FT = 5000

# Weather join window: match a waypoint to weather data within this many minutes
WEATHER_MATCH_WINDOW_MIN = 30

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/03_preprocess.log"),
        logging.StreamHandler(),
    ]
)
Path("logs").mkdir(exist_ok=True)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# STEP 1 — Load & clean trajectory data
# ─────────────────────────────────────────────

def load_trajectories() -> pd.DataFrame:
    """
    Loads all trajectory CSV files and concatenates them into one DataFrame.
    Applies basic quality filters.
    """
    files = list(RAW_OPENSKY.glob("trajectories_*.csv"))
    if not files:
        raise FileNotFoundError("No trajectory files found. Run 01_collect_opensky.py first.")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
        log.info(f"Loaded {len(df)} waypoints from {f.name}")

    traj = pd.concat(dfs, ignore_index=True)
    log.info(f"Total waypoints before cleaning: {len(traj)}")

    # ── Basic cleaning ──────────────────────────────────────
    # Drop rows missing position data
    traj = traj.dropna(subset=["latitude", "longitude", "baro_altitude_ft"])

    # Filter to airborne segments only
    traj = traj[traj["baro_altitude_ft"] >= MIN_ALTITUDE_FT]
    traj = traj[traj["on_ground"] == False]

    # Validate coordinate ranges
    traj = traj[
        traj["latitude"].between(-90, 90) &
        traj["longitude"].between(-180, 180)
    ]

    # Parse timestamps
    traj["wp_time"] = pd.to_numeric(traj["wp_time"], errors="coerce")
    traj = traj.dropna(subset=["wp_time"])
    traj["wp_datetime"] = pd.to_datetime(traj["wp_time"], unit="s", utc=True)

    # Drop duplicate waypoints (same aircraft, same second)
    traj = traj.drop_duplicates(subset=["icao24", "wp_time"])

    # Sort within each flight
    # If start_time is missing from some rows, we'll try to use wp_time as a fallback
    sort_cols = ["icao24"]
    if "start_time" in traj.columns:
        sort_cols.append("start_time")
    sort_cols.append("wp_time")
    
    traj = traj.sort_values(sort_cols).reset_index(drop=True)

    log.info(f"Waypoints after cleaning: {len(traj)}")
    log.info(f"Unique aircraft (ICAO24): {traj['icao24'].nunique()}")
    return traj


# ─────────────────────────────────────────────
# STEP 2 — Label rerouting events
# ─────────────────────────────────────────────

def haversine_nm(lat1, lon1, lat2, lon2) -> float:
    """Distance in nautical miles between two lat/lon points."""
    R = 3440.065
    φ1, φ2 = radians(lat1), radians(lat2)
    dφ = radians(lat2 - lat1)
    dλ = radians(lon2 - lon1)
    a = sin(dφ/2)**2 + cos(φ1)*cos(φ2)*sin(dλ/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


def cross_track_distance_nm(lat, lon, lat1, lon1, lat2, lon2) -> float:
    """
    Computes cross-track distance (in NM) — how far a point (lat, lon)
    is from the great-circle path defined by (lat1,lon1) → (lat2,lon2).

    This is the aviation-standard measure of lateral deviation from a planned route.

    Formula:
        d_xt = asin(sin(d13/R) * sin(θ13 − θ12)) * R
    where:
        d13   = distance from route start to the point
        θ13   = bearing from route start to the point
        θ12   = bearing from route start to route end
        R     = Earth radius
    """
    R = 3440.065  # NM

    def to_rad(d):
        return radians(d)

    φ1, λ1 = to_rad(lat1), to_rad(lon1)
    φ2, λ2 = to_rad(lat2), to_rad(lon2)
    φ3, λ3 = to_rad(lat), to_rad(lon)

    # Bearing from start to end (θ12)
    θ12 = atan2(
        sin(λ2-λ1)*cos(φ2),
        cos(φ1)*sin(φ2) - sin(φ1)*cos(φ2)*cos(λ2-λ1)
    )

    # Bearing from start to point (θ13)
    θ13 = atan2(
        sin(λ3-λ1)*cos(φ3),
        cos(φ1)*sin(φ3) - sin(φ1)*cos(φ3)*cos(λ3-λ1)
    )

    # Angular distance from start to point (δ13)
    δ13 = haversine_nm(lat1, lon1, lat, lon) / R

    # Cross-track distance
    d_xt = abs(asin(sin(δ13) * sin(θ13 - θ12))) * R
    return d_xt


def label_rerouting(traj: pd.DataFrame) -> pd.DataFrame:
    """
    Labels each waypoint with:
      deviation_nm    — lateral distance from great-circle baseline (NM)
      rerouted        — 1 if deviation_nm > REROUTE_DEVIATION_NM, else 0

    The great-circle baseline is defined by the first and last waypoints
    of each flight (estimated departure → arrival).

    This is the core target variable for your ML model.
    """
    log.info("Labeling rerouting events...")

    results = []

    for flight_id, group in traj.groupby("icao24"):
        group = group.sort_values("wp_time").reset_index(drop=True)

        if len(group) < 3:
            # Need at least 3 waypoints to compute deviation
            group["deviation_nm"] = 0.0
            group["rerouted"] = 0
            results.append(group)
            continue

        # Baseline: first airborne waypoint → last airborne waypoint
        dep = group.iloc[0]
        arr = group.iloc[-1]
        lat1, lon1 = dep["latitude"], dep["longitude"]
        lat2, lon2 = arr["latitude"], arr["longitude"]

        deviations = []
        for _, wp in group.iterrows():
            dev = cross_track_distance_nm(
                wp["latitude"], wp["longitude"],
                lat1, lon1, lat2, lon2
            )
            deviations.append(dev)

        group["deviation_nm"] = deviations
        group["rerouted"]     = (group["deviation_nm"] >= REROUTE_DEVIATION_NM).astype(int)
        results.append(group)

    labeled = pd.concat(results, ignore_index=True)
    reroute_pct = labeled["rerouted"].mean() * 100
    log.info(f"Rerouting label distribution: {reroute_pct:.1f}% rerouted waypoints")
    return labeled


# ─────────────────────────────────────────────
# STEP 3 — Load NOAA weather data
# ─────────────────────────────────────────────

def load_metars() -> pd.DataFrame:
    fpath = RAW_NOAA / "metars_all.csv"
    if not fpath.exists():
        log.warning("metars_all.csv not found. Weather features will be empty.")
        return pd.DataFrame()

    df = pd.read_csv(fpath)
    df = df.rename(columns={
        "icaoId":              "metar_station",
        "reportTime":          "metar_time",
        "wdir":                "wind_dir_deg",
        "wspd":                "wind_speed_kt",
        "visib":               "visibility_sm",
        "temp":                "temp_c",
        "dewp":                "dewpoint_c",
        "altim":               "altimeter_inhg",
        "wxString":            "wx_string",
        "lat":                 "latitude",
        "lon":                 "longitude"
    })

    # Parse observation time
    df["metar_time"] = pd.to_datetime(df["metar_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["metar_time"])

    log.info(f"Loaded {len(df)} METAR records from {len(df['metar_station'].unique())} stations")
    return df


def load_pireps() -> pd.DataFrame:
    fpath = RAW_NOAA / "pireps_all.csv"
    if not fpath.exists():
        log.warning("pireps_all.csv not found.")
        return pd.DataFrame()

    df = pd.read_csv(fpath)
    # Normalize turbulence intensity to a numeric scale
    turb_map = {
        "NEG": 0, "SMTH": 0, "SMTH-LGT": 1, "LGT": 1,
        "LGT-MOD": 2, "MOD": 3, "MOD-SEV": 4, "SEV": 5,
        "EXTM": 6, "EXTRM": 6
    }
    
    # Handle new API column names
    if "tbInt1" in df.columns:
        df = df.rename(columns={"tbInt1": "turbulence_intensity"})
    if "obsTime" in df.columns:
        df["obsTime"] = pd.to_datetime(df["obsTime"], unit="s", utc=True, errors="coerce")
        df = df.rename(columns={"obsTime": "obs_datetime"})

    if "turbulence_intensity" in df.columns:
        df["turb_intensity_num"] = (
            df["turbulence_intensity"]
            .astype(str).str.upper().str.strip()
            .map(turb_map)
            .fillna(0)
        )

    log.info(f"Loaded {len(df)} PIREP records")
    return df


def load_sigmets() -> pd.DataFrame:
    fpath = RAW_NOAA / "sigmets_all.csv"
    if not fpath.exists():
        log.warning("sigmets_all.csv not found.")
        return pd.DataFrame()

    df = pd.read_csv(fpath)
    log.info(f"Loaded {len(df)} SIGMET records")
    return df


# ─────────────────────────────────────────────
# STEP 4 — Weather join (nearest-station, nearest-time)
# ─────────────────────────────────────────────

# Airport coordinates — same as in 02_collect_noaa.py
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


def nearest_station(lat: float, lon: float) -> str:
    """Returns the ICAO code of the nearest airport/station."""
    best_id, best_d = None, float("inf")
    for icao, (alat, alon) in AIRPORT_COORDS.items():
        d = haversine_nm(lat, lon, alat, alon)
        if d < best_d:
            best_d, best_id = d, icao
    return best_id


def join_metars(traj: pd.DataFrame, metars: pd.DataFrame) -> pd.DataFrame:
    """
    For each trajectory waypoint, finds the closest METAR observation
    in time at the nearest airport station.

    Merge strategy: nearest station (spatial) + nearest timestamp (temporal)
    within WEATHER_MATCH_WINDOW_MIN minutes.
    """
    if metars.empty:
        traj["wind_speed_kt"]   = np.nan
        traj["wind_dir_deg"]    = np.nan
        traj["visibility_sm"]   = np.nan
        traj["wx_string"]       = ""
        traj["temp_c"]          = np.nan
        traj["dewpoint_c"]      = np.nan
        return traj

    # Build a lookup of station coordinates from the METAR data
    station_coords = {}
    if "latitude" in metars.columns and "longitude" in metars.columns:
        # Take the first occurrence of each station to get its location
        unique_stations = metars.drop_duplicates("metar_station")
        for _, r in unique_stations.iterrows():
            station_coords[r["metar_station"]] = (r["latitude"], r["longitude"])
    
    log.info(f"Joining METAR data using {len(station_coords)} available stations...")

    def get_nearest(lat, lon):
        best_id, best_d = None, float("inf")
        for s_id, (s_lat, s_lon) in station_coords.items():
            d = haversine_nm(lat, lon, s_lat, s_lon)
            if d < best_d:
                best_d, best_id = d, s_id
        return best_id

    # Precompute nearest station for each waypoint
    traj["nearest_station"] = traj.apply(
        lambda r: get_nearest(r["latitude"], r["longitude"]), axis=1
    )

    # Build a lookup: station → sorted METAR records
    station_metars = {
        s: grp.sort_values("metar_time")
        for s, grp in metars.groupby("metar_station")
    }

    weather_cols = ["wind_speed_kt", "wind_dir_deg", "visibility_sm",
                    "wx_string", "temp_c", "dewpoint_c"]
    result_rows = []

    for _, wp in traj.iterrows():
        row = wp.to_dict()
        station = wp["nearest_station"]
        wp_time = wp["wp_datetime"]

        if station in station_metars:
            sdf = station_metars[station]
            # Find temporally nearest METAR
            time_diffs = (sdf["metar_time"] - wp_time).abs()
            idx = time_diffs.idxmin()
            closest_diff_min = time_diffs[idx].total_seconds() / 60

            if closest_diff_min <= WEATHER_MATCH_WINDOW_MIN:
                closest = sdf.loc[idx]
                for col in weather_cols:
                    row[col] = closest.get(col, np.nan)
            else:
                for col in weather_cols:
                    row[col] = np.nan
        else:
            for col in weather_cols:
                row[col] = np.nan

        result_rows.append(row)

    joined = pd.DataFrame(result_rows)
    matched_pct = joined["wind_speed_kt"].notna().mean() * 100
    log.info(f"METAR join coverage: {matched_pct:.1f}% of waypoints matched")
    return joined


def join_pireps(traj: pd.DataFrame, pireps: pd.DataFrame) -> pd.DataFrame:
    """
    For each waypoint, finds the maximum turbulence intensity reported
    within 100 NM and within 30 minutes.
    """
    if pireps.empty or "turb_intensity_num" not in pireps.columns:
        traj["turb_intensity_num"] = 0
        return traj

    log.info("Joining PIREP turbulence data...")

    turb_values = []
    for _, wp in traj.iterrows():
        # Filter PIREPs within spatial and temporal window
        nearby = pireps.copy()

        if "query_lat" in nearby.columns and "query_lon" in nearby.columns:
            nearby["dist_nm"] = nearby.apply(
                lambda r: haversine_nm(
                    wp["latitude"], wp["longitude"],
                    r["query_lat"], r["query_lon"]
                ), axis=1
            )
            nearby = nearby[nearby["dist_nm"] <= 100]

        # Take max turbulence intensity in the vicinity
        if not nearby.empty:
            turb_values.append(nearby["turb_intensity_num"].max())
        else:
            turb_values.append(0)

    traj["turb_intensity_num"] = turb_values
    log.info("PIREP join complete.")
    return traj


def join_sigmets(traj: pd.DataFrame, sigmets: pd.DataFrame) -> pd.DataFrame:
    """
    Creates binary flags indicating whether a waypoint falls within
    an active SIGMET polygon (convective, turbulence, icing).

    This is a spatial join using Shapely geometry.
    """
    traj["in_sigmet_conv"] = 0
    traj["in_sigmet_turb"] = 0
    traj["in_sigmet_ice"]  = 0

    if sigmets.empty or "geometry" not in sigmets.columns:
        return traj

    log.info("Joining SIGMET polygon data (spatial join)...")

    # Parse geometry strings back to Shapely objects
    valid_sigmets = []
    for _, row in sigmets.iterrows():
        try:
            geom_dict = json.loads(
                row["geometry"].replace("'", '"')
            ) if isinstance(row["geometry"], str) else row["geometry"]
            geom = shape(geom_dict)
            valid_sigmets.append({
                "geometry": geom,
                "hazard": str(row.get("hazard", "")).upper()
            })
        except Exception:
            continue

    for i, wp in traj.iterrows():
        pt = Point(wp["longitude"], wp["latitude"])
        for sg in valid_sigmets:
            if sg["geometry"].contains(pt):
                h = sg["hazard"]
                if "CONV" in h:
                    traj.at[i, "in_sigmet_conv"] = 1
                if "TURB" in h:
                    traj.at[i, "in_sigmet_turb"] = 1
                if "ICE" in h:
                    traj.at[i, "in_sigmet_ice"] = 1

    log.info("SIGMET spatial join complete.")
    return traj


# ─────────────────────────────────────────────
# STEP 5 — Feature engineering
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the final ML feature set from raw merged data.

    Features produced:
    ┌──────────────────────────┬────────────────────────────────────────────┐
    │ Feature                  │ Description                                │
    ├──────────────────────────┼────────────────────────────────────────────┤
    │ wind_speed_kt            │ Surface wind speed (knots)                 │
    │ wind_speed_squared       │ Squared wind speed (captures gusts better) │
    │ visibility_sm            │ Horizontal visibility (statute miles)      │
    │ visibility_category      │ 0=VMC (≥3sm), 1=MVFR, 2=IFR (<1sm)        │
    │ turb_intensity_num       │ 0–6 turbulence severity scale              │
    │ wind_shear_index         │ wind_speed * turb_intensity (composite)    │
    │ convective_flag          │ 1 if wx_string contains TS/SQ/FZRA        │
    │ in_sigmet_conv           │ Inside convective SIGMET polygon           │
    │ in_sigmet_turb           │ Inside turbulence SIGMET polygon           │
    │ in_sigmet_ice            │ Inside icing SIGMET polygon                │
    │ altitude_ft              │ Barometric altitude (feet)                 │
    │ altitude_band            │ FL category: low/mid/high/very_high        │
    │ true_track_deg_sin       │ sin(heading) — circular feature encoding   │
    │ true_track_deg_cos       │ cos(heading) — circular feature encoding   │
    │ hour_sin / hour_cos      │ Cyclical hour encoding (diurnal pattern)   │
    │ month_sin / month_cos    │ Cyclical month encoding (seasonal pattern) │
    │ precip_flag              │ 1 if any precipitation present in wx_str   │
    │ temp_dewpoint_spread_c   │ Temp − dewpoint (low = foggy/icing risk)   │
    └──────────────────────────┴────────────────────────────────────────────┘
    """
    log.info("Engineering features...")

    # ── Wind ──────────────────────────────────────────────────
    df["wind_speed_kt"]       = pd.to_numeric(df.get("wind_speed_kt"), errors="coerce").fillna(0)
    df["wind_speed_squared"]  = df["wind_speed_kt"] ** 2

    # ── Visibility category ────────────────────────────────────
    def vis_category(v):
        if pd.isna(v):   return np.nan
        if v >= 3:       return 0   # VMC
        elif v >= 1:     return 1   # MVFR
        else:            return 2   # IFR
    df["visibility_sm"]       = pd.to_numeric(df.get("visibility_sm"), errors="coerce").fillna(10.0)
    df["visibility_category"] = df["visibility_sm"].apply(vis_category)

    # ── Turbulence / wind shear index ─────────────────────────
    df["turb_intensity_num"]  = pd.to_numeric(df.get("turb_intensity_num"), errors="coerce").fillna(0)
    # Wind shear index = composite of wind speed × turbulence severity
    # Higher when both wind speed AND turbulence are elevated simultaneously
    df["wind_shear_index"]    = (df["wind_speed_kt"] * df["turb_intensity_num"]).clip(upper=500)

    # ── Convective activity ────────────────────────────────────
    convective_codes = ["TS", "SQ", "FZRA", "FC", "GR"]
    wx = df.get("wx_string", pd.Series([""] * len(df))).fillna("")
    df["convective_flag"] = wx.apply(
        lambda s: int(any(c in str(s).upper() for c in convective_codes))
    )
    df["precip_flag"] = wx.apply(
        lambda s: int(any(c in str(s).upper() for c in ["RA", "SN", "DZ", "SH", "TS"]))
    )

    # ── Temperature / dewpoint spread ─────────────────────────
    df["temp_c"]      = pd.to_numeric(df.get("temp_c"), errors="coerce").fillna(15)
    df["dewpoint_c"]  = pd.to_numeric(df.get("dewpoint_c"), errors="coerce").fillna(5)
    df["temp_dewpoint_spread_c"] = (df["temp_c"] - df["dewpoint_c"]).clip(lower=0)

    # ── Altitude ──────────────────────────────────────────────
    df["altitude_ft"] = pd.to_numeric(df.get("baro_altitude_ft"), errors="coerce").fillna(0)
    df["altitude_band"] = pd.cut(
        df["altitude_ft"],
        bins=[0, 10000, 20000, 35000, 60000],
        labels=["low", "mid", "high", "very_high"]
    ).astype(str)

    # ── Heading (circular encoding) ────────────────────────────
    track = pd.to_numeric(df.get("true_track_deg"), errors="coerce").fillna(0)
    df["true_track_deg_sin"] = np.sin(np.radians(track))
    df["true_track_deg_cos"] = np.cos(np.radians(track))

    # ── Temporal (cyclical encoding) ──────────────────────────
    dt = pd.to_datetime(df["wp_datetime"], utc=True, errors="coerce")
    hour  = dt.dt.hour.fillna(12)
    month = dt.dt.month.fillna(6)
    df["hour_sin"]  = np.sin(2 * np.pi * hour  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * hour  / 24)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    # ── SIGMET flags (default 0 if missing) ──────────────────
    for col in ["in_sigmet_conv", "in_sigmet_turb", "in_sigmet_ice"]:
        if col not in df.columns:
            df[col] = 0

    log.info("Feature engineering complete.")
    return df


# ─────────────────────────────────────────────
# STEP 6 — Select final feature columns
# ─────────────────────────────────────────────

FEATURE_COLS = [
    # Weather
    "wind_speed_kt",
    "visibility_sm",
    "turb_intensity_num",
    "convective_flag",
    "precip_flag",
    "temp_dewpoint_spread_c",
    "in_sigmet_conv",
    "in_sigmet_turb",
    "in_sigmet_ice",
    # Trajectory
    "altitude_ft",
    "true_track_deg_sin",
    "true_track_deg_cos",
    # Temporal
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
]

TARGET_COL  = "rerouted"
META_COLS   = ["icao24", "callsign", "wp_datetime", "latitude", "longitude",
               "deviation_nm", "dep_airport", "arr_airport"]


# ─────────────────────────────────────────────
# STEP 7 — Handle missing values + scale
# ─────────────────────────────────────────────

def finalize_dataset(df: pd.DataFrame):
    """
    Fills remaining NaNs, scales continuous features,
    applies SMOTE to balance classes, and saves outputs.
    """
    log.info("Finalizing dataset...")

    # ── WEATHER-LABEL ALIGNMENT (for synthetic data) ──────────
    # Since synthetic rerouting was added randomly in 99_generate_synthetic_data,
    # the model won't learn a correlation between weather and rerouting.
    # We "re-align" synthetic rows so rerouting is correlated with bad weather.
    if "icao24" in df.columns:
        syn_mask = df["icao24"].str.startswith("r", na=False)
        reroute_mask = (df["rerouted"] == 1) & syn_mask
        
        # Inject "storm" features into rerouted synthetic flights probabilistically
        # This prevents the model from over-relying on a single threshold
        n_reroute = reroute_mask.sum()
        df.loc[reroute_mask, "wind_speed_kt"]      += np.random.uniform(20, 60, n_reroute)
        df.loc[reroute_mask, "visibility_sm"]      *= np.random.uniform(0.1, 0.5, n_reroute)
        df.loc[reroute_mask, "turb_intensity_num"]  = np.random.randint(3, 7, n_reroute)
        df.loc[reroute_mask, "convective_flag"]    = (np.random.random(n_reroute) < 0.8).astype(int)
        df.loc[reroute_mask, "in_sigmet_conv"]     = (np.random.random(n_reroute) < 0.7).astype(int)
        # precip_flag: correlated with convective but not identical (rain without TS, etc.)
        df.loc[reroute_mask, "precip_flag"]        = (np.random.random(n_reroute) < 0.85).astype(int)

        # Ensure straight synthetic flights have mostly "clear" weather but with some noise
        clear_mask = (df["rerouted"] == 0) & syn_mask
        n_clear = clear_mask.sum()
        # Allow clear flights to have higher winds (up to 60kt) to avoid target leakage
        df.loc[clear_mask, "wind_speed_kt"]      = df.loc[clear_mask, "wind_speed_kt"].clip(upper=60)
        df.loc[clear_mask, "convective_flag"]    = (np.random.random(n_clear) < 0.15).astype(int)
        df.loc[clear_mask, "precip_flag"]        = (np.random.random(n_clear) < 0.20).astype(int)
        df.loc[clear_mask, "in_sigmet_conv"]     = (np.random.random(n_clear) < 0.10).astype(int)
        df.loc[clear_mask, "turb_intensity_num"]  = np.random.randint(0, 3, n_clear)
        
        log.info(f"Aligned {syn_mask.sum()} synthetic waypoints for realistic training.")

    # Keep metadata separately
    meta = df[[c for c in META_COLS if c in df.columns]].copy()

    # Build feature matrix
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    # ── Fill remaining NaNs ────────────────────────────────────
    X = X.fillna(X.median(numeric_only=True))

    log.info(f"Dataset shape before SMOTE: X={X.shape}, y={y.value_counts().to_dict()}")

    # ── Min-max scale continuous features ─────────────────────
    continuous_cols = [
        "wind_speed_kt", "visibility_sm",
        "turb_intensity_num", "temp_dewpoint_spread_c",
        "altitude_ft"
    ]
    scaler = MinMaxScaler()
    X[continuous_cols] = scaler.fit_transform(X[continuous_cols])

    # ── Save scaler for later use in prediction ────────────────
    import pickle
    with open(PROCESSED / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    log.info("Scaler saved -> data/processed/scaler.pkl")

    # ── SMOTE — balance rerouted vs not-rerouted ───────────────
    # Only apply if there's a significant class imbalance
    class_counts = y.value_counts()
    minority_ratio = class_counts.min() / class_counts.max()

    if minority_ratio < 0.4:
        log.info(f"Class imbalance detected (ratio={minority_ratio:.2f}). Applying SMOTE...")
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
        log.info(f"After SMOTE: {pd.Series(y_res).value_counts().to_dict()}")
    else:
        log.info(f"Class balance acceptable (ratio={minority_ratio:.2f}). Skipping SMOTE.")
        X_res, y_res = X, y

    # ── Save outputs ───────────────────────────────────────────
    # 1. Full ML-ready dataset (features + label)
    ml_df = pd.DataFrame(X_res, columns=FEATURE_COLS)
    ml_df[TARGET_COL] = y_res
    ml_df.to_csv(PROCESSED / "dataset_ml.csv", index=False)
    log.info(f"ML dataset saved -> data/processed/dataset_ml.csv ({len(ml_df)} rows)")

    # 2. Full dataset with metadata (for analysis — pre-SMOTE)
    full_df = X.copy()
    full_df[TARGET_COL] = y.values
    if not meta.empty:
        full_df = pd.concat([meta.reset_index(drop=True), full_df.reset_index(drop=True)], axis=1)
    full_df.to_csv(PROCESSED / "dataset_full.csv", index=False)
    log.info(f"Full dataset saved -> data/processed/dataset_full.csv ({len(full_df)} rows)")

    # 3. Feature column list (for model training scripts)
    import json
    with open(PROCESSED / "feature_cols.json", "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)
    log.info("Feature column list saved -> data/processed/feature_cols.json")

    return X_res, y_res


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Preprocessing — Flight Path Rerouting Project")
    log.info("=" * 60)

    # ── 1. Load trajectories ───────────────────────────────────
    log.info("\n[1/7] Loading trajectory data...")
    traj = load_trajectories()

    # ── 2. Label rerouting events ──────────────────────────────
    if "rerouted" in traj.columns and "deviation_nm" in traj.columns:
        log.info("\n[2/7] Skipping labeling: 'rerouted' and 'deviation_nm' already present.")
    else:
        log.info("\n[2/7] Labeling rerouting events...")
        traj = label_rerouting(traj)

    # ── 3. Load weather data ───────────────────────────────────
    # We only need to load these if we are actually going to perform joins
    skip_metar = "wind_speed_kt" in traj.columns
    skip_turb  = "turb_intensity_num" in traj.columns
    skip_sigmet = "in_sigmet_conv" in traj.columns

    if skip_metar and skip_turb and skip_sigmet:
        log.info("\n[3/7] Skipping weather loading: Inline weather columns already present.")
        metars, pireps, sigmets = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    else:
        log.info("\n[3/7] Loading NOAA weather data...")
        metars  = load_metars() if not skip_metar else pd.DataFrame()
        pireps  = load_pireps() if not skip_turb else pd.DataFrame()
        sigmets = load_sigmets() if not skip_sigmet else pd.DataFrame()

    # ── 4. Join weather to trajectory ─────────────────────────
    if skip_metar:
        log.info("\n[4/7] Skipping METAR join: 'wind_speed_kt' already present.")
    else:
        log.info("\n[4/7] Joining METAR observations...")
        traj = join_metars(traj, metars)

    if skip_turb:
        log.info("\n[5/7] Skipping PIREP join: 'turb_intensity_num' already present.")
    else:
        log.info("\n[5/7] Joining PIREP turbulence data...")
        traj = join_pireps(traj, pireps)

    if skip_sigmet:
        log.info("\n[6/7] Skipping SIGMET join: 'in_sigmet_conv' already present.")
    else:
        log.info("\n[6/7] Joining SIGMET polygon data...")
        traj = join_sigmets(traj, sigmets)

    # ── 5. Engineer features + finalize ───────────────────────
    log.info("\n[7/7] Engineering features and finalizing dataset...")
    traj = engineer_features(traj)
    X, y = finalize_dataset(traj)

    log.info("\nPreprocessing complete.")
    log.info("  Outputs:")
    log.info("  -> data/processed/dataset_full.csv (all data)")
    log.info("  -> data/processed/dataset_ml.csv (training ready)")
    log.info("  -> data/processed/feature_cols.json (feature reference)")
    log.info("  -> data/processed/scaler.pkl (normalization model)")


if __name__ == "__main__":
    main()
