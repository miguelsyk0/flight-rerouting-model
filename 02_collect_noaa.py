"""
02_collect_noaa.py
==================
Collects aviation weather data from NOAA Aviation Weather Center (AWC).

What this script does:
  1. Reads the trajectory CSV files produced by 01_collect_opensky.py
  2. Extracts unique (timestamp, latitude, longitude) points from trajectories
  3. For each waypoint cluster, fetches:
       - METAR  -> wind speed/direction, visibility, precipitation
       - PIREP  -> pilot-reported turbulence at altitude (best proxy for wind shear)
       - SIGMET -> active hazard polygons (thunderstorm cells, severe turbulence)
       - G-AIRMET -> area-wide advisories (IFR, turbulence, icing)
  4. Saves raw weather responses to /data/raw/noaa/

Requirements:
    pip install requests pandas python-dotenv shapely

Notes:
  - AWC API requires NO API key.
  - Rate limit: 100 req/min; recommended <= 1 req/min per thread.
  - Historical window: up to 15 days.
  - We map waypoints to the nearest ICAO airport to use as METAR station IDs.
"""

import time
import logging
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta
from math import radians, sin, cos, sqrt, atan2

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

BASE_AWC = "https://aviationweather.gov/api/data"

# All airports in our study — we'll query METARs for these
TARGET_AIRPORTS = [
    "KJFK", "KORD", "KATL", "KDFW", "KLAX", "EGLL",
    # Add nearby airports for better spatial coverage
    "KEWR", "KLGA",       # NYC area
    "KMDW", "KARR",       # Chicago area
    "KBOS", "KPHL",       # East Coast
    "KDEN", "KSLC",       # Mountain corridor
    "KSFO", "KOAK",       # Bay Area
]

RAW_DIR   = Path("data/raw/opensky")
OUT_DIR   = Path("data/raw/noaa")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Hours of weather history to pull per METAR query
METAR_HOURS = 3

# Spatial search radius for PIREPs (nautical miles from waypoint cluster)
PIREP_RADIUS_NM = 100

REQUEST_DELAY_SEC = 1.2   # ~50 req/min, well under the 100 limit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/02_collect_noaa.log"),
        logging.StreamHandler(),
    ]
)
Path("logs").mkdir(exist_ok=True)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# API HELPER
# ─────────────────────────────────────────────

def safe_get(endpoint: str, params: dict, retries: int = 3) -> dict | list | None:
    """
    Generic GET against the AWC Data API.
    Automatically retries on transient failures.
    """
    url = f"{BASE_AWC}/{endpoint}"
    headers = {
        "User-Agent": "FlightReroutingResearch/1.0 (academic; saint-louis-university)"
    }

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=20)
            if resp.status_code == 404:
                return None
            if resp.status_code == 429:
                log.warning("Rate limited by AWC. Sleeping 60s...")
                time.sleep(60)
                continue
            resp.raise_for_status()

            # AWC returns JSON arrays or objects depending on the endpoint
            ct = resp.headers.get("Content-Type", "")
            if "json" in ct:
                return resp.json()
            return resp.text   # fallback for text formats

        except requests.RequestException as e:
            log.warning(f"Attempt {attempt}/{retries} failed ({endpoint}): {e}")
            if hasattr(e, 'response') and e.response is not None:
                log.warning(f"  Response: {e.response.text[:200]}")
            time.sleep(5 * attempt)

    return None


# ─────────────────────────────────────────────
# METAR — Surface observations
# ─────────────────────────────────────────────

def fetch_metars(hours: int = METAR_HOURS) -> list[dict]:
    """
    Fetches METAR observations for the entire CONUS area.
    """
    # Bounding box for CONUS: [minLat, minLon, maxLat, maxLon]
    params = {
        "bbox":   "24,-125,49,-66",
        "format": "json",
        "hours":  hours,
        "taf":    "false",
    }
    log.info(f"Fetching all CONUS METARs (last {hours}h)...")
    data = safe_get("metar", params)
    time.sleep(REQUEST_DELAY_SEC)

    if not data:
        log.warning("METAR fetch returned no data.")
        return []

    # Normalize — AWC returns a list of observation dicts
    records = data if isinstance(data, list) else []
    log.info(f"  Got {len(records)} METAR records.")
    return records


# ─────────────────────────────────────────────
# PIREP — Pilot weather reports (turbulence)
# ─────────────────────────────────────────────

def fetch_pireps(lamin, lamax, lomin, lomax) -> list[dict]:
    """
    Fetches pilot weather reports (PIREPs) within a bounding box.
    """
    params = {
        "bbox":   f"{lamin},{lomin},{lamax},{lomax}",
        "format": "json",
        "age":    12, # Increase window to 12h for better coverage
    }
    log.info(f"Fetching PIREPs for bbox [{lomin}, {lamin}, {lomax}, {lamax}]...")
    data = safe_get("pirep", params)
    time.sleep(REQUEST_DELAY_SEC)

    if not data:
        return []

    records = data if isinstance(data, list) else []
    return records


# ─────────────────────────────────────────────
# SIGMET — Significant meteorological hazard polygons
# ─────────────────────────────────────────────

def fetch_sigmets() -> list[dict]:
    """
    Fetches all active SIGMETs — bounding polygons for significant aviation hazards.
    
    We fetch all hazards at once and return the GeoJSON features.
    """
    params = {"format": "geojson"}
    log.info("Fetching all active SIGMETs...")
    data = safe_get("sigmet", params)
    time.sleep(REQUEST_DELAY_SEC)

    if not data:
        return []

    # GeoJSON FeatureCollection
    if isinstance(data, dict) and "features" in data:
        return data["features"]

    return []


# ─────────────────────────────────────────────
# G-AIRMET — Area aviation advisories
# ─────────────────────────────────────────────

def fetch_gairmets() -> list[dict]:
    """
    Fetches G-AIRMET advisories (replaced text AIRMETs in Jan 2025).
    Covers contiguous U.S. airspace.

    G-AIRMETs are time-stamped polygon advisories — they provide the
    "look-ahead" weather context for your LSTM's temporal input window.

    Key returned fields:
      hazard       -> IFR / TURB / ICE / MTN OBSCN / SFC_WND
      severity     -> intensity
      geometry     -> polygon
      validTime    -> when the advisory is active
      altitudeLow / altitudeHi
    """
    params = {"format": "geojson"}
    log.info("Fetching G-AIRMETs...")
    data = safe_get("gairmet", params)
    time.sleep(REQUEST_DELAY_SEC)

    if not data:
        return []

    if isinstance(data, dict) and "features" in data:
        return data["features"]

    return []


# ─────────────────────────────────────────────
# TAF — Terminal aerodrome forecasts (look-ahead)
# ─────────────────────────────────────────────

def fetch_tafs(airport_ids: list[str]) -> list[dict]:
    """
    Fetches Terminal Aerodrome Forecasts (TAFs) for a list of airports.

    TAFs are 24–30 hr ahead forecasts — they feed the LSTM's future
    weather input window. Critical for predicting rerouting BEFORE
    conditions actually materialize.

    Key returned fields:
      forecast_time_from / forecast_time_to
      wind_speed_kt, wind_dir_degrees
      visibility_statute_mi
      wx_string (precipitation/weather codes)
      sky_condition (cloud ceiling forecast)
    """
    ids_str = ",".join(airport_ids)
    params = {
        "ids":    ids_str,
        "format": "json",
    }
    log.info(f"Fetching TAFs for {len(airport_ids)} stations...")
    data = safe_get("taf", params)
    time.sleep(REQUEST_DELAY_SEC)

    if not data:
        return []

    return data if isinstance(data, list) else []


# ─────────────────────────────────────────────
# SPATIAL HELPERS
# ─────────────────────────────────────────────

def haversine_nm(lat1, lon1, lat2, lon2) -> float:
    """
    Returns distance in nautical miles between two lat/lon points.
    Used to find the nearest airport to a trajectory waypoint.
    """
    R = 3440.065  # Earth radius in nautical miles
    φ1, φ2 = radians(lat1), radians(lat2)
    dφ = radians(lat2 - lat1)
    dλ = radians(lon2 - lon1)
    a = sin(dφ/2)**2 + cos(φ1)*cos(φ2)*sin(dλ/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


# Rough lat/lon centroids for our target airports (for spatial queries)
AIRPORT_COORDS = {
    "KJFK": (40.6413, -73.7781),
    "KORD": (41.9742, -87.9073),
    "KATL": (33.6367, -84.4281),
    "KDFW": (32.8998, -97.0403),
    "KLAX": (33.9425, -118.4081),
    "EGLL": (51.4775, -0.4614),
    "KEWR": (40.6925, -74.1687),
    "KLGA": (40.7772, -73.8726),
    "KMDW": (41.7868, -87.7522),
    "KARR": (41.7719, -88.4757),
    "KBOS": (42.3643, -71.0052),
    "KPHL": (39.8719, -75.2411),
    "KDEN": (39.8561, -104.6737),
    "KSLC": (40.7884, -111.9778),
    "KSFO": (37.6213, -122.3790),
    "KOAK": (37.7213, -122.2208),
}


def nearest_airport(lat: float, lon: float) -> str:
    """Returns the ICAO code of the nearest airport to a lat/lon."""
    best_id, best_dist = None, float("inf")
    for icao, (alat, alon) in AIRPORT_COORDS.items():
        d = haversine_nm(lat, lon, alat, alon)
        if d < best_dist:
            best_dist, best_id = d, icao
    return best_id


# ─────────────────────────────────────────────
# SAVE HELPERS
# ─────────────────────────────────────────────

def save_json_as_csv(records: list[dict], filename: str):
    if not records:
        log.warning(f"No records to save for {filename}.")
        return
    df = pd.json_normalize(records)
    out = OUT_DIR / filename
    df.to_csv(out, index=False)
    log.info(f"Saved {len(df)} records -> {out}")


def save_geojson_features(features: list[dict], filename: str):
    """Flattens GeoJSON features to a CSV, keeping geometry as a string."""
    if not features:
        log.warning(f"No GeoJSON features to save for {filename}.")
        return
    rows = []
    for feat in features:
        row = feat.get("properties", {}).copy()
        row["geometry"] = str(feat.get("geometry", {}))
        rows.append(row)
    df = pd.DataFrame(rows)
    out = OUT_DIR / filename
    df.to_csv(out, index=False)
    log.info(f"Saved {len(df)} features -> {out}")


# ─────────────────────────────────────────────
# PIREP COLLECTION — per trajectory cluster
# ─────────────────────────────────────────────

def collect_pireps_for_trajectories():
    """
    Reads all trajectory CSVs, samples representative lat/lon clusters,
    and fetches PIREPs for each cluster using localized bboxes.
    """
    all_pireps = []
    traj_files = list(RAW_DIR.glob("trajectories_*.csv"))

    if not traj_files:
        log.warning("No trajectory files found. Run 01_collect_opensky.py or 99_generate_synthetic_data.py first.")
        return

    # To avoid thousands of queries, we'll round waypoints to a 2.0 degree grid
    # and query each grid cell once.
    grid_points = set()

    for fpath in traj_files:
        df = pd.read_csv(fpath)
        if df.empty or "latitude" not in df.columns:
            continue
        
        # Grid decimation
        df["lat_grid"] = (df["latitude"] / 2.0).round() * 2.0
        df["lon_grid"] = (df["longitude"] / 2.0).round() * 2.0
        points = df[["lat_grid", "lon_grid"]].drop_duplicates()
        
        for _, row in points.iterrows():
            grid_points.add((row["lat_grid"], row["lon_grid"]))

    log.info(f"Discovered {len(grid_points)} unique 2x2 degree grid cells along flight paths.")

    for lat, lon in grid_points:
        # Create a 2x2 degree bbox around the grid center
        lamin, lamax = lat - 1.0, lat + 1.0
        lomin, lomax = lon - 1.0, lon + 1.0
        
        pireps = fetch_pireps(lamin, lamax, lomin, lomax)
        if pireps:
            for p in pireps:
                p["query_grid_lat"] = lat
                p["query_grid_lon"] = lon
            all_pireps.extend(pireps)
        
        # Small delay to respect rate limits
        time.sleep(REQUEST_DELAY_SEC)

    save_json_as_csv(all_pireps, "pireps_all.csv")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("NOAA Weather Data Collection — Flight Path Rerouting Project")
    log.info("=" * 60)

    # ── METARs ──────────────────────────────────────────────
    log.info("\n[1/5] Collecting METARs...")
    metars = fetch_metars()
    save_json_as_csv(metars, "metars_all.csv")

    # ── TAFs ────────────────────────────────────────────────
    log.info("\n[2/5] Collecting TAFs (forecasts)...")
    tafs = fetch_tafs(TARGET_AIRPORTS)
    save_json_as_csv(tafs, "tafs_all.csv")

    # ── SIGMETs (convective, turbulence, icing) ─────────────
    log.info("\n[3/5] Collecting SIGMETs...")
    all_sigmets_raw = fetch_sigmets()
    hazards_to_keep = ["CONVECTIVE", "TURB", "ICE"]
    all_sigmets = []
    
    for s in all_sigmets_raw:
        props = s.get("properties", {})
        h = props.get("hazard")
        if h in hazards_to_keep:
            props["hazard_query"] = h
            all_sigmets.append(s)
            
    save_geojson_features(all_sigmets, "sigmets_all.csv")

    # ── G-AIRMETs ───────────────────────────────────────────
    log.info("\n[4/5] Collecting G-AIRMETs...")
    gairmets = fetch_gairmets()
    save_geojson_features(gairmets, "gairmets_all.csv")

    # ── PIREPs (per trajectory cluster) ─────────────────────
    log.info("\n[5/5] Collecting PIREPs for trajectory clusters...")
    collect_pireps_for_trajectories()

    log.info("\n[SUCCESS] NOAA collection finished. Check data/raw/noaa/")


if __name__ == "__main__":
    main()
