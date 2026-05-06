"""
01_collect_opensky.py
==================
Collects flights by geographic bounding box instead of airport.

Pipeline:
  1. Iterate through each defined corridor (US_East_Coast, etc.)
  2. For each corridor, iterate through a range of hourly timestamps within a specific date range (default: last 7 days)
  3. For each timestamp, fetch all aircraft states within the corridor's bounding box using OpenSky's /states/all API
  4. Collect all unique aircraft IDs (ICAO24) that appear in any snapshot for each corridor
  5. Save the complete list of ICAO24s for each corridor to a separate JSON file
"""

import os, time, logging, requests, pandas as pd, json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_URL  = "https://opensky-network.org/api"
TOKEN_URL = (
    "https://auth.opensky-network.org/auth/realms/"
    "opensky-network/protocol/openid-connect/token"
)

# ── Define your study corridors as bounding boxes ──────────────────────────
# Each entry: (name, lamin, lamax, lomin, lomax)
CORRIDORS = [
    ("US_East_Coast",   24.0, 47.0,  -82.0,  -65.0),
    ("US_Midwest",      36.0, 48.0, -100.0,  -80.0),
    ("US_Crosscountry", 32.0, 42.0, -120.0,  -95.0),
    ("North_Atlantic",  45.0, 65.0,  -60.0,  -10.0),
    ("Gulf_of_Mexico",  18.0, 31.0,  -97.0,  -80.0),
]

# How many hourly snapshots to take per corridor per day
# (OpenSky historical states are available at 1-hour resolution for free users)
SNAPSHOTS_PER_DAY = 4   # e.g., 00:00, 06:00, 12:00, 18:00 UTC

COLLECTION_END   = datetime.now(timezone.utc) - timedelta(days=1)
COLLECTION_START = COLLECTION_END - timedelta(days=6) # Use 6 days to stay safely within 7-day API limits

OUT_DIR = Path("data/raw/opensky")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = OUT_DIR / "icao24_cache.json"
Path("logs").mkdir(exist_ok=True)

REQUEST_DELAY_SEC   = 2.0
MAX_TRACKS_PER_RUN  = 200   # total track fetches across all corridors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/01b_collect_opensky_bbox.log"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)


# ── Token manager (same as 01_collect_opensky.py) ─────────────────────────

class TokenManager:
    REFRESH_MARGIN_SEC = 60

    def __init__(self):
        self._token      = None
        self._expires_at = None
        self._client_id     = os.getenv("OPENSKY_CLIENT_ID")
        self._client_secret = os.getenv("OPENSKY_CLIENT_SECRET")
        if not self._client_id or not self._client_secret:
            raise EnvironmentError("Missing OPENSKY credentials in .env")

    def get_headers(self):
        return {"Authorization": f"Bearer {self._get_token()}"}

    def _get_token(self):
        now = datetime.now(timezone.utc)
        if self._token and self._expires_at and now < self._expires_at:
            return self._token
        return self._refresh()

    def _refresh(self):
        resp = requests.post(
            TOKEN_URL,
            data={
                "grant_type":    "client_credentials",
                "client_id":     self._client_id,
                "client_secret": self._client_secret,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]
        expires_in  = data.get("expires_in", 1800)
        self._expires_at = (
            datetime.now(timezone.utc)
            + timedelta(seconds=expires_in - self.REFRESH_MARGIN_SEC)
        )
        return self._token


def safe_get(url, params, headers, retries=3):
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=20)
            if resp.status_code == 404:  return None
            if resp.status_code == 429:
                wait = int(resp.headers.get("X-Rate-Limit-Retry-After-Seconds", 30))
                if wait > 60:
                    log.warning(f"Extreme rate limit detected ({wait}s). Skipping wait.")
                    return None
                log.warning(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            
            # Handle 400/403 with detailed debugging
            if resp.status_code in [400, 403]:
                log.error(f"{resp.status_code} Error: OpenSky denied access or rejected the request.")
                log.error(f"URL: {resp.url}")
                log.error(f"Response: {resp.text[:500]}")
                if resp.status_code == 403:
                    log.error("Possible reasons: Historical data (>1h) requires a paid/researcher account.")
                elif resp.status_code == 400:
                    log.error("Possible reasons: Time range too large (>7 days) or invalid parameters.")
                resp.raise_for_status()

            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            log.warning(f"Attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(5 * attempt)
    return None


# ── STEP 1: Global Flight Discovery ───────────────────────────────────────

def fetch_flights_all(begin_ts: int, end_ts: int, tokens: TokenManager) -> list[dict]:
    """
    Calls /flights/all for a specific time interval (max 2 hours).
    Returns a list of flight metadata dicts.
    """
    params = {
        "begin": begin_ts,
        "end":   end_ts,
    }
    data = safe_get(f"{BASE_URL}/flights/all", params, tokens.get_headers())
    return data if data else []


def discover_icao24s_by_time(
    start_dt: datetime, 
    end_dt: datetime, 
    tokens: TokenManager
) -> set[str]:
    """
    Iterates through the date range and collects unique ICAO24s 
    using the /flights/all endpoint (more accessible for history).
    """
    log.info(f"Discovering flights via /flights/all from {start_dt.date()} to {end_dt.date()}")
    icao24_set = set()
    MAX_DISCOVERY = 5000
    
    # Check if we have cached IDs to use as fallback
    cached_ids = []
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r") as f:
            cached_ids = json.load(f)

    # Discovery loop
    cursor = start_dt
    while cursor < end_dt and len(icao24_set) < MAX_DISCOVERY:
        for hour_start in [0, 8, 16]:
            if len(icao24_set) >= MAX_DISCOVERY: break
            begin = cursor.replace(hour=hour_start, minute=0, second=0, microsecond=0)
            end   = begin + timedelta(hours=2)
            if begin >= end_dt: break
            
            ts_begin = int(begin.timestamp())
            ts_end   = int(end.timestamp())
            
            log.info(f"  Sampling flights: {begin.strftime('%Y-%m-%d %H:%M')} UTC...")
            flights = fetch_flights_all(ts_begin, ts_end, tokens)
            
            # If rate limited, use cache as fallback
            if flights is None or (isinstance(flights, list) and not flights and len(icao24_set) == 0):
                if cached_ids:
                    log.warning("    Discovery failed or rate-limited. Falling back to cached IDs.")
                    return set(cached_ids)
                return set()

            if flights:
                for f in flights:
                    if f.get("icao24"):
                        icao24_set.add(f["icao24"])
            
            time.sleep(REQUEST_DELAY_SEC)
        cursor += timedelta(days=1)
        
    # Update cache if we found new IDs
    if icao24_set:
        with open(CACHE_FILE, "w") as f:
            json.dump(list(icao24_set), f)
            
    log.info(f"Total unique aircraft discovered: {len(icao24_set)}")
    return icao24_set


# (Removed collect_icao24s_for_corridor as we now discover globally first)


def fetch_tracks_and_filter_by_corridors(
    icao24_set: set[str],
    corridors: list[tuple],
    tokens: TokenManager,
    max_tracks: int = MAX_TRACKS_PER_RUN
):
    """
    For discovered ICAO24s, fetches full tracks and saves those 
    that pass through our study corridors.
    """
    sample = list(icao24_set)
    # Shuffle or just limit to avoid over-fetching
    import random
    random.shuffle(sample)
    sample = sample[:max_tracks * 2] # Fetch a bit more as some won't be in corridors
    
    log.info(f"Processing tracks for {len(sample)} aircraft...")

    # We'll group waypoints by corridor
    corridor_data = {name: [] for (name, *_) in corridors}
    
    start_ts = int(COLLECTION_START.timestamp())
    end_ts   = int(COLLECTION_END.timestamp())
    tracks_saved = 0

    for i, icao24 in enumerate(sample):
        if tracks_saved >= max_tracks: break
        
        # OpenSky /flights/aircraft limits queries to 2 partitions (days).
        # To be safe, we'll query in 24-hour chunks.
        flights = []
        chunk_start = start_ts
        while chunk_start < end_ts:
            chunk_end = min(chunk_start + (24 * 3600), end_ts)
            params = {"icao24": icao24.lower(), "begin": chunk_start, "end": chunk_end}
            chunk_flights = safe_get(f"{BASE_URL}/flights/aircraft", params, tokens.get_headers())
            if chunk_flights:
                flights.extend(chunk_flights)
            time.sleep(REQUEST_DELAY_SEC)
            chunk_start = chunk_end

        if not flights: continue

        for flight in flights[:2]: # Max 2 flights per aircraft
            first_seen = flight.get("firstSeen")
            if not first_seen: continue

            track_data = safe_get(f"{BASE_URL}/tracks/all", {"icao24": icao24.lower(), "time": first_seen}, tokens.get_headers())
            time.sleep(REQUEST_DELAY_SEC)

            if not track_data or "path" not in track_data: continue

            # Check if any part of the track falls into our corridors
            track_points = track_data["path"]
            assigned_corridors = set()
            
            for wp in track_points:
                lat, lon = wp[1], wp[2]
                if lat is None or lon is None: continue
                
                for (name, lamin, lamax, lomin, lomax) in corridors:
                    if lamin <= lat <= lamax and lomin <= lon <= lomax:
                        assigned_corridors.add(name)
                        corridor_data[name].append({
                            "icao24":            icao24.lower(),
                            "callsign":          track_data.get("callsign", "").strip(),
                            "corridor":          name,
                            "dep_airport":       flight.get("estDepartureAirport"),
                            "arr_airport":       flight.get("estArrivalAirport"),
                            "start_time":        first_seen,
                            "wp_time":           wp[0],
                            "latitude":          lat,
                            "longitude":         lon,
                            "baro_altitude_ft":  round(wp[3] * 3.28084) if wp[3] else None,
                            "true_track_deg":    wp[4],
                            "on_ground":         wp[5],
                        })

            if assigned_corridors:
                tracks_saved += 1
                if tracks_saved % 5 == 0:
                    log.info(f"  Progress: {tracks_saved}/{max_tracks} tracks saved...")

    # Save to CSVs
    for name, data in corridor_data.items():
        if data:
            df = pd.DataFrame(data)
            out = OUT_DIR / f"trajectories_{name}.csv"
            df.to_csv(out, index=False)
            log.info(f"Saved {len(df)} waypoints for {name} -> {out}")


# ── MAIN ──────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("OpenSky Bbox Collection — Flight Path Rerouting Project")
    log.info(f"Date range: {COLLECTION_START.date()} -> {COLLECTION_END.date()}")
    log.info("=" * 60)

    tokens = TokenManager()
    
    # Pass 1: Discover aircraft via /flights/all (Standard Account Friendly)
    icao24s = discover_icao24s_by_time(COLLECTION_START, COLLECTION_END, tokens)
    
    if not icao24s:
        log.error("No aircraft discovered. Check your internet connection or credentials.")
        return

    # Pass 2: Fetch tracks and filter them into corridors
    fetch_tracks_and_filter_by_corridors(
        icao24s, CORRIDORS, tokens, max_tracks=MAX_TRACKS_PER_RUN
    )

    log.info("\n[SUCCESS] Collection finished. Check data/raw/opensky/")
    log.info(f"  Trajectory CSVs: trajectories_<corridor>.csv -> feeds 03_preprocess.py")


if __name__ == "__main__":
    main()