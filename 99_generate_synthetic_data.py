"""
99_generate_synthetic_data.py
=====================================================
Generates physically coherent synthetic flight trajectories + weather so the
downstream ML pipeline produces a model that has learned a REAL causal signal:
flights deviate because of weather, not because of a random flag.

Key design principles
---------------------
1. Trajectories follow airways / great-circle paths with realistic navigation.
2. Weather is generated as a spatial field (Gaussian storm cells + fronts) that
   is consistent across the region — nearby waypoints share similar conditions.
3. Rerouting is CAUSED by the weather:
     a. A forward-looking "hazard scan" checks the next N nm of the planned route.
     b. If a storm/front polygon is detected, the flight deviates around it.
     c. The deviation magnitude, direction, and duration are physically realistic.
4. METAR-like observations are sampled from the weather field at each waypoint
   (with realistic sensor noise), so the weather features look like real data.
5. No post-hoc label/weather alignment step is needed in preprocessing.

Outputs (same filenames as before, drop-in replacement)
-------------------------------------------------------
  data/raw/opensky/trajectories_<corridor>.csv   ← trajectory waypoints
  data/raw/noaa/metars_all.csv                   ← METAR-format weather obs
  data/raw/noaa/pireps_all.csv                   ← PIREP-format turbulence
  data/raw/noaa/sigmets_all.csv                  ← SIGMET polygon records
  data/raw/noaa/gairmets_all.csv                 ← G-AIRMET area advisories
  data/raw/noaa/tafs_all.csv                     ← TAF forecasts (empty stub)

Usage
-----
  python 99_generate_synthetic_data.py [--flights N] [--seed S]

  --flights  total flights to generate across all corridors (default: 600)
  --seed     random seed for reproducibility (default: 42)
"""

import argparse
import json
import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

OPENSKY_DIR = Path("data/raw/opensky")
NOAA_DIR    = Path("data/raw/noaa")
OPENSKY_DIR.mkdir(parents=True, exist_ok=True)
NOAA_DIR.mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/99_generate_synthetic_data.log"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

# Study corridors — (name, lat_min, lat_max, lon_min, lon_max)
CORRIDORS = [
    ("US_East_Coast",   24.0, 47.0,  -82.0,  -65.0),
    ("US_Midwest",      36.0, 48.0, -100.0,  -80.0),
    ("US_Crosscountry", 32.0, 42.0, -120.0,  -95.0),
]

# Waypoint interval along track
WP_INTERVAL_NM  = 15.0      # nautical miles between waypoints
EARTH_RADIUS_NM = 3440.065

# Rerouting physics
REROUTE_SCAN_NM     = 80.0   # look-ahead distance to detect hazards
REROUTE_OFFSET_NM   = 40.0   # lateral offset when detouring (centre of detour)
REROUTE_TAPER_WP    = 8      # waypoints to blend back to original track
REROUTE_MIN_DEV_NM  = 30.0   # minimum deviation to label a waypoint "rerouted"

# Weather world parameters
N_STORM_CELLS   = 12    # convective cells per corridor
N_FRONTS        = 3     # synoptic fronts per corridor
CELL_RADIUS_NM  = 60.0  # average storm cell radius
FRONT_WIDTH_NM  = 40.0  # front zone half-width

# Altitude distribution (realistic cruise altitudes)
CRUISE_ALTS_FT  = [29000, 31000, 33000, 35000, 37000, 39000, 41000]
CRUISE_WEIGHTS  = [0.05,  0.10,  0.20,  0.30,  0.20,  0.10,  0.05]

# Rerouting probability modifiers
BASE_REROUTE_PROB   = 0.35   # fraction of flights that encounter a hazard
VFR_IMMUNITY        = 0.10   # fraction of low-severity encounters that don't reroute

# Realistic airport pairs per corridor (ICAO, lat, lon)
AIRPORTS = {
    "US_East_Coast": [
        ("KJFK",  40.64, -73.78), ("KMIA",  25.80, -80.28), ("KBOS",  42.36, -71.01),
        ("KPHL",  39.87, -75.24), ("KDCA",  38.85, -77.04), ("KBWI",  39.18, -76.67),
        ("KATL",  33.64, -84.43), ("KCLT",  35.21, -80.94), ("KRDU",  35.88, -78.79),
        ("KORH",  42.27, -71.87), ("KORF",  35.07, -76.03), ("KPVD",  41.73, -71.43),
    ],
    "US_Midwest": [
        ("KORD",  41.97, -87.91), ("KMDW",  41.79, -87.75), ("KDEN",  39.86,-104.67),
        ("KMSP",  44.88, -93.22), ("KSTL",  38.75, -90.37), ("KCVG",  39.05, -84.67),
        ("KPIT",  40.49, -80.23), ("KBUF",  42.94, -78.73), ("KMLB",  44.50, -93.00),
        ("KFSD",  43.58, -96.74), ("KOMA",  41.30, -95.89), ("KCLE",  41.41, -81.85),
    ],
    "US_Crosscountry": [
        ("KLAX",  33.94,-118.41), ("KSFO",  37.62,-122.38), ("KLAS",  36.08,-115.15),
        ("KPHX",  33.44,-112.01), ("KDEN",  39.86,-104.67), ("KDFW",  32.90, -97.04),
        ("KORD",  41.97, -87.91), ("KATL",  33.64, -84.43), ("KSLC",  40.79,-111.98),
        ("KSEA",  47.44,-122.31), ("KPDX",  45.59,-122.60), ("KOAK",  37.72,-122.22),
    ],
}


# ──────────────────────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in nautical miles."""
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a  = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return EARTH_RADIUS_NM * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing (degrees true) from point 1 to point 2."""
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dλ = math.radians(lon2 - lon1)
    x  = math.sin(dλ) * math.cos(φ2)
    y  = math.cos(φ1)*math.sin(φ2) - math.sin(φ1)*math.cos(φ2)*math.cos(dλ)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def destination(lat: float, lon: float, brg_deg: float, dist_nm: float):
    """Destination point from (lat,lon) given bearing and distance."""
    δ   = dist_nm / EARTH_RADIUS_NM
    φ1  = math.radians(lat)
    λ1  = math.radians(lon)
    brg = math.radians(brg_deg)
    φ2  = math.asin(math.sin(φ1)*math.cos(δ) +
                    math.cos(φ1)*math.sin(δ)*math.cos(brg))
    λ2  = λ1 + math.atan2(
        math.sin(brg)*math.sin(δ)*math.cos(φ1),
        math.cos(δ) - math.sin(φ1)*math.sin(φ2)
    )
    return math.degrees(φ2), (math.degrees(λ2) + 540) % 360 - 180


def cross_track_nm(lat: float, lon: float,
                   lat1: float, lon1: float,
                   lat2: float, lon2: float) -> float:
    """Signed cross-track distance (NM) from great-circle lat1→lat2."""
    R   = EARTH_RADIUS_NM
    d13 = haversine_nm(lat1, lon1, lat, lon) / R
    θ13 = math.radians(bearing(lat1, lon1, lat, lon))
    θ12 = math.radians(bearing(lat1, lon1, lat2, lon2))
    return math.asin(math.sin(d13) * math.sin(θ13 - θ12)) * R


# ──────────────────────────────────────────────────────────────────────────────
# WEATHER WORLD
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StormCell:
    """A convective storm cell — circular polygon in lat/lon space."""
    center_lat: float
    center_lon: float
    radius_deg: float          # degrees (≈ radius_nm / 60)
    intensity: float           # 0–1: controls wind speed, turbulence
    top_fl: int                # top flight level (e.g. 450 = FL450)
    is_sigmet: bool = True     # whether it generates an active SIGMET
    geom: Optional[Polygon] = field(default=None, repr=False)

    def __post_init__(self):
        # Build circular polygon in lat/lon (good enough for ±45° lat)
        pts = [
            destination(self.center_lat, self.center_lon,
                        a, self.radius_deg * 60)
            for a in range(0, 360, 15)
        ]
        self.geom = Polygon([(lon, lat) for lat, lon in pts])

    def contains(self, lat: float, lon: float) -> bool:
        return self.geom.contains(Point(lon, lat))

    def distance_nm(self, lat: float, lon: float) -> float:
        return haversine_nm(lat, lon, self.center_lat, self.center_lon)


@dataclass
class SynopticFront:
    """A weather front — a narrow band polygon."""
    waypoints: list            # list of (lat, lon) along the front
    width_nm: float            # half-width of the frontal zone
    intensity: float           # 0–1: wind shear, icing, turbulence severity
    front_type: str            # "COLD", "WARM", "OCCLUDED"
    geom: Optional[Polygon] = field(default=None, repr=False)

    def __post_init__(self):
        if len(self.waypoints) < 2:
            self.geom = None
            return
        # Buffer each segment to create a band
        from shapely.geometry import LineString
        line = LineString([(lon, lat) for lat, lon in self.waypoints])
        # Convert width_nm to approximate degrees
        width_deg = self.width_nm / 60.0
        self.geom = line.buffer(width_deg)

    def contains(self, lat: float, lon: float) -> bool:
        if self.geom is None:
            return False
        return self.geom.contains(Point(lon, lat))


class WeatherWorld:
    """
    Spatial weather model for one corridor.
    Provides coherent, physically motivated weather at any (lat, lon, alt_ft).
    """

    def __init__(self, corridor: tuple, rng: np.random.Generator):
        name, lat_min, lat_max, lon_min, lon_max = corridor
        self.name    = name
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.rng     = rng

        self.cells:  list[StormCell]    = []
        self.fronts: list[SynopticFront] = []

        self._generate_storms()
        self._generate_fronts()

        log.info(f"  [{name}] Weather world: {len(self.cells)} storm cells, "
                 f"{len(self.fronts)} fronts")

    def _generate_storms(self):
        n = self.rng.integers(max(1, N_STORM_CELLS - 4),
                              N_STORM_CELLS + 4)
        for _ in range(n):
            lat = self.rng.uniform(self.lat_min + 1, self.lat_max - 1)
            lon = self.rng.uniform(self.lon_min + 1, self.lon_max - 1)
            r   = self.rng.uniform(0.5, 1.5) * CELL_RADIUS_NM / 60.0
            intensity = float(self.rng.beta(2, 3))   # skewed toward moderate
            top_fl    = int(self.rng.choice([350, 400, 430, 450, 480]))
            self.cells.append(StormCell(lat, lon, r, intensity, top_fl))

    def _generate_fronts(self):
        n = self.rng.integers(1, N_FRONTS + 2)
        for _ in range(n):
            # Generate a quasi-linear front (3–5 anchor points)
            n_pts = self.rng.integers(3, 6)
            lat0  = self.rng.uniform(self.lat_min + 2, self.lat_max - 2)
            lon0  = self.rng.uniform(self.lon_min + 2, self.lon_max - 4)
            pts   = [(lat0, lon0)]
            for _ in range(n_pts - 1):
                dlat = float(self.rng.uniform(-3, 3))
                dlon = float(self.rng.uniform(2, 6))
                lat0 = max(self.lat_min, min(self.lat_max, lat0 + dlat))
                lon0 = min(self.lon_max, lon0 + dlon)
                pts.append((lat0, lon0))

            width     = float(self.rng.uniform(0.6, 1.4)) * FRONT_WIDTH_NM
            intensity = float(self.rng.beta(3, 3))
            ftype     = str(self.rng.choice(["COLD", "WARM", "OCCLUDED"]))
            try:
                self.fronts.append(SynopticFront(pts, width, intensity, ftype))
            except Exception:
                pass   # degenerate geometry — skip

    # ── Query interface ────────────────────────────────────────────────────

    def query(self, lat: float, lon: float, alt_ft: float) -> dict:
        """
        Returns a dict of weather variables at (lat, lon, alt_ft).
        Values mimic what a METAR/PIREP would report with realistic noise.
        """
        rng = self.rng

        # ── Background (clear-air) conditions ──────────────────────────────
        base_wind     = float(rng.gamma(3, 4))          # ~12 kt avg clear air
        base_vis      = float(rng.uniform(8, 10))
        base_turb     = 0
        precip        = False
        convective    = False
        # Temperature lapse rate: ~2°C per 1000 ft, starting ~15°C at surface
        temp_base = 15.0 - (alt_ft / 1000) * 2.0
        temp_c    = float(rng.uniform(temp_base - 5, temp_base + 5))
        dewpoint_c    = temp_c - float(rng.uniform(5, 30))

        in_cell  = False
        in_front = False
        cell_intensity = 0.0
        front_intensity = 0.0

        # ── Storm cell influence ────────────────────────────────────────────
        for cell in self.cells:
            if alt_ft > cell.top_fl * 100:
                continue   # above storm top — no effect
            d = cell.distance_nm(lat, lon)
            r = cell.radius_deg * 60
            if d < r:
                in_cell = True
                cell_intensity = max(cell_intensity, cell.intensity)
                # Falloff: strongest at center, tapering to edge
                falloff = max(0.0, 1.0 - d / r)
                base_wind  += cell.intensity * falloff * 60
                base_vis    = min(base_vis, 10 - cell.intensity * falloff * 9)
                base_turb   = max(base_turb, int(cell.intensity * falloff * 6))
                precip      = True
                convective  = cell.intensity * falloff > 0.3
                temp_c     += cell.intensity * falloff * 5   # latent heat

        # ── Synoptic front influence ─────────────────────────────────────────
        for front in self.fronts:
            if front.contains(lat, lon):
                in_front = True
                front_intensity = max(front_intensity, front.intensity)
                fi = front.intensity
                base_wind   += fi * 30
                base_vis     = min(base_vis, 10 - fi * 6)
                base_turb    = max(base_turb, int(fi * 4))
                precip       = fi > 0.4
                dewpoint_c   = temp_c - fi * 5   # moist air

        # ── Altitude adjustments (upper-air physics) ─────────────────────────
        fl = alt_ft / 1000
        # Jet stream enhancement above FL250
        if fl >= 250:
            jet_factor = min(1.0, (fl - 250) / 100)
            base_wind += jet_factor * 60 + float(rng.uniform(-10, 10))
        # Icing layer (FL100–FL200)
        if 10000 <= alt_ft <= 20000 and dewpoint_c > temp_c - 5:
            base_turb = max(base_turb, int(rng.integers(1, 3)))
        # Clear-air turbulence near jet stream
        if fl >= 340 and not in_cell and not in_front:
            if rng.random() < 0.15:
                base_turb = max(base_turb, int(rng.integers(1, 4)))

        # ── Sensor noise (realistic METAR/PIREP variance) ────────────────────
        wind_speed = float(np.clip(base_wind + rng.normal(0, 2), 0, 150))
        visibility = float(np.clip(base_vis  + rng.normal(0, 0.5), 0.0625, 10))
        turb_num   = min(6, max(0, base_turb))

        # Discrete METAR visibility reporting (eighths of a mile)
        vis_reported = round(visibility * 8) / 8

        return {
            "wind_speed_kt":      round(wind_speed, 1),
            "wind_dir_deg":       int(rng.integers(0, 360)),
            "visibility_sm":      vis_reported,
            "temp_c":             round(temp_c, 1),
            "dewpoint_c":         round(dewpoint_c, 1),
            "turb_intensity_num": turb_num,
            "convective_flag":    int(convective),
            "precip_flag":        int(precip),
            "in_storm_cell":      int(in_cell),
            "in_synoptic_front":  int(in_front),
            "cell_intensity":     round(cell_intensity, 3),
            "wx_string":          ("TS" if convective else
                                   "RA" if precip else ""),
        }

    def hazard_ahead(self, lat: float, lon: float,
                     brg: float, scan_nm: float, alt_ft: float) -> float:
        """
        Scan the next `scan_nm` along bearing `brg` for hazards.
        Returns max hazard intensity [0, 1] encountered.
        """
        max_hazard = 0.0
        for d_nm in range(10, int(scan_nm) + 1, 10):
            slat, slon = destination(lat, lon, brg, d_nm)
            for cell in self.cells:
                if alt_ft <= cell.top_fl * 100:
                    dist = cell.distance_nm(slat, slon)
                    r    = cell.radius_deg * 60
                    if dist < r * 1.2:    # 20% buffer for early avoidance
                        max_hazard = max(max_hazard, cell.intensity)
            for front in self.fronts:
                if front.contains(slat, slon):
                    max_hazard = max(max_hazard, front.intensity)
        return max_hazard

    def get_sigmets(self) -> list[dict]:
        """Export storm cells as SIGMET-format records."""
        records = []
        for i, cell in enumerate(self.cells):
            if not cell.is_sigmet:
                continue
            coords = list(cell.geom.exterior.coords)
            records.append({
                "hazard":    "CONVECTIVE",
                "severity":  "SEV" if cell.intensity > 0.6 else "MOD",
                "top_fl":    cell.top_fl,
                "intensity": round(cell.intensity, 3),
                "geometry":  str({
                    "type":        "Polygon",
                    "coordinates": [[[lon, lat] for lon, lat in coords]]
                }),
                "source": f"SYN-CELL-{i:03d}",
            })
        for i, front in enumerate(self.fronts):
            if front.geom is None:
                continue
            try:
                coords = list(front.geom.exterior.coords)
            except AttributeError:
                continue
            records.append({
                "hazard":    "TURB",
                "severity":  "MOD" if front.intensity > 0.5 else "LGT",
                "top_fl":    300,
                "intensity": round(front.intensity, 3),
                "geometry":  str({
                    "type":        "Polygon",
                    "coordinates": [[[lon, lat] for lon, lat in coords]]
                }),
                "source": f"SYN-FRONT-{i:03d}-{front.front_type}",
            })
        return records


# ──────────────────────────────────────────────────────────────────────────────
# FLIGHT GENERATOR
# ──────────────────────────────────────────────────────────────────────────────

def pick_airport_pair(corridor_name: str, rng: np.random.Generator):
    """Pick a random origin/destination pair from the corridor's airport list."""
    airports = AIRPORTS[corridor_name]
    orig, dest = rng.choice(len(airports), 2, replace=False)
    return airports[orig], airports[dest]


def plan_great_circle(lat1, lon1, lat2, lon2,
                      interval_nm=WP_INTERVAL_NM) -> list[tuple]:
    """
    Generate great-circle waypoints from (lat1,lon1) to (lat2,lon2)
    at `interval_nm` spacing.
    """
    total_nm = haversine_nm(lat1, lon1, lat2, lon2)
    n_steps  = max(4, int(total_nm / interval_nm))
    pts      = []
    for i in range(n_steps + 1):
        f    = i / n_steps
        lat  = lat1 + f * (lat2 - lat1)
        lon  = lon1 + f * (lon2 - lon1)
        pts.append((lat, lon))
    return pts


def simulate_flight(
    icao24: str,
    callsign: str,
    corridor_name: str,
    orig: tuple,
    dest: tuple,
    alt_ft: int,
    start_ts: int,
    weather: WeatherWorld,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Simulate one flight from orig to dest.
    Returns a list of waypoint dicts (same schema as 01_collect_opensky.py output).

    Rerouting logic
    ---------------
    At each waypoint, the pilot "looks ahead" REROUTE_SCAN_NM along the planned
    track. If a hazard of sufficient intensity is detected, the flight deviates
    laterally (left or right, choosing the safer side), flies around the hazard,
    then tapers back to the original track.
    """
    orig_name, lat_dep, lon_dep = orig
    dest_name, lat_arr, lon_arr = dest

    planned = plan_great_circle(lat_dep, lon_dep, lat_arr, lon_arr)
    n_pts   = len(planned)

    wp_records = []
    wp_time    = start_ts

    # State variables for the rerouting state machine
    rerouting       = False
    reroute_side    = 1      # +1 = right, -1 = left
    offset_nm       = 0.0    # current lateral offset
    target_offset   = 0.0   # target offset (max of the detour)
    taper_remaining = 0      # waypoints left in return-to-track taper
    reroute_end_idx = None   # planned index where detour is expected to end

    actual_lat, actual_lon = lat_dep, lon_dep

    for i, (plan_lat, plan_lon) in enumerate(planned):

        track_brg = bearing(actual_lat, actual_lon, lat_arr, lon_arr)

        # ── Hazard scan (only when not already fully committed to detour) ──
        if not rerouting or (reroute_end_idx and i < reroute_end_idx - 5):
            hazard = weather.hazard_ahead(
                actual_lat, actual_lon, track_brg,
                REROUTE_SCAN_NM, alt_ft
            )
            # Trigger reroute if hazard intensity > random threshold
            # (more intense hazards are avoided more often)
            threshold = rng.uniform(0.2, 0.5)
            if hazard > threshold and not rerouting:
                rerouting       = True
                reroute_side    = int(rng.choice([-1, 1]))
                target_offset   = float(rng.uniform(
                    REROUTE_OFFSET_NM * 0.6, REROUTE_OFFSET_NM * 1.6
                ))
                offset_nm       = 0.0
                taper_remaining = 0
                reroute_end_idx = min(n_pts - 1, i + int(target_offset / WP_INTERVAL_NM) + REROUTE_TAPER_WP + 4)

        # ── Apply lateral offset ───────────────────────────────────────────
        if rerouting and taper_remaining == 0:
            # Build-up phase: ramp offset toward target over a few waypoints
            offset_nm = min(offset_nm + target_offset / 4, target_offset)
            # Check if we've passed the worst hazard and should start returning
            current_hazard = weather.hazard_ahead(
                actual_lat, actual_lon, track_brg, 30, alt_ft
            )
            if (offset_nm >= target_offset - 1 and current_hazard < 0.15
                    and i > (reroute_end_idx or 0) - REROUTE_TAPER_WP - 2):
                taper_remaining = REROUTE_TAPER_WP

        if rerouting and taper_remaining > 0:
            # Return-to-track phase: reduce offset linearly
            offset_nm      = max(0, offset_nm - target_offset / REROUTE_TAPER_WP)
            taper_remaining -= 1
            if offset_nm < 1.0:
                rerouting = False
                offset_nm = 0.0

        # Compute actual position: offset perpendicular to track
        if abs(offset_nm) > 0.5:
            perp_brg   = (track_brg + reroute_side * 90) % 360
            actual_lat, actual_lon = destination(plan_lat, plan_lon, perp_brg, offset_nm)
        else:
            actual_lat, actual_lon = plan_lat, plan_lon

        # ── Query weather at this position ─────────────────────────────────
        wx = weather.query(actual_lat, actual_lon, alt_ft)

        # Compute deviation from planned great-circle (for label)
        dev_nm = abs(cross_track_nm(
            actual_lat, actual_lon,
            lat_dep, lon_dep, lat_arr, lon_arr
        ))

        # ── Build waypoint record ──────────────────────────────────────────
        wp_records.append({
            # Trajectory fields (match 01_collect_opensky.py schema)
            "icao24":            icao24,
            "callsign":          callsign,
            "corridor":          corridor_name,
            "dep_airport":       orig_name,
            "arr_airport":       dest_name,
            "start_time":        start_ts,
            "wp_time":           wp_time,
            "latitude":          round(actual_lat, 5),
            "longitude":         round(actual_lon, 5),
            "baro_altitude_ft":  alt_ft,
            "true_track_deg":    round(track_brg, 1),
            "on_ground":         False,
            # Pre-computed label (used by 03_preprocess.py to skip re-labeling)
            "deviation_nm":      round(dev_nm, 2),
            "rerouted":          int(dev_nm >= REROUTE_MIN_DEV_NM),
            # Inline weather (avoids the noisy nearest-station join problem)
            "wind_speed_kt":     wx["wind_speed_kt"],
            "wind_dir_deg":      wx["wind_dir_deg"],
            "visibility_sm":     wx["visibility_sm"],
            "temp_c":            wx["temp_c"],
            "dewpoint_c":        wx["dewpoint_c"],
            "turb_intensity_num":wx["turb_intensity_num"],
            "convective_flag":   wx["convective_flag"],
            "precip_flag":       wx["precip_flag"],
            "wx_string":         wx["wx_string"],
            "in_sigmet_conv":    int(wx["in_storm_cell"] and wx["convective_flag"]),
            "in_sigmet_turb":    int(wx["turb_intensity_num"] >= 3),
            "in_sigmet_ice":     int(10000 <= alt_ft <= 20000 and wx["temp_c"] < 0
                                     and wx["visibility_sm"] < 5),
        })

        # Advance time: ~15 NM at ~450 kt ≈ 2 minutes per waypoint
        wp_time += int(rng.normal(120, 15))

    return wp_records


# ──────────────────────────────────────────────────────────────────────────────
# METAR / PIREP / SIGMET EXPORTERS
# ──────────────────────────────────────────────────────────────────────────────

def build_metars(all_waypoints: pd.DataFrame) -> pd.DataFrame:
    """
    Build a synthetic METAR table by sampling the weather embedded in waypoints.
    We take one observation per unique (dep_airport, hour) to mimic hourly METARs.
    """
    records = []
    if "dep_airport" not in all_waypoints.columns:
        return pd.DataFrame()

    for station, grp in all_waypoints.groupby("dep_airport"):
        grp = grp.dropna(subset=["wind_speed_kt", "visibility_sm"])
        if grp.empty:
            continue
        # One METAR per hour at this station
        grp["hour_bin"] = (grp["wp_time"] // 3600) * 3600
        for hour_ts, hgrp in grp.groupby("hour_bin"):
            row = hgrp.iloc[0]
            records.append({
                "icaoId":     station,
                "reportTime": pd.to_datetime(hour_ts, unit="s", utc=True).isoformat(),
                "wdir":       int(row.get("wind_dir_deg", 0)),
                "wspd":       row.get("wind_speed_kt", 0),
                "visib":      row.get("visibility_sm", 10),
                "temp":       row.get("temp_c", 15),
                "dewp":       row.get("dewpoint_c", 5),
                "wxString":   row.get("wx_string", ""),
                "lat":        round(row["latitude"], 3),
                "lon":        round(row["longitude"], 3),
                "altim":      29.92,
            })
    return pd.DataFrame(records)


def build_pireps(all_waypoints: pd.DataFrame) -> pd.DataFrame:
    """
    Build PIREP records for waypoints where turbulence was significant.
    """
    TURB_LABELS = {0:"NEG",1:"SMTH-LGT",2:"LGT",3:"LGT-MOD",4:"MOD",5:"MOD-SEV",6:"SEV"}
    turb_wps = all_waypoints[all_waypoints["turb_intensity_num"] >= 2].copy()
    if turb_wps.empty:
        return pd.DataFrame()

    # Sample a fraction — not every pilot files a PIREP
    sample = turb_wps.sample(frac=0.4, random_state=1)
    records = []
    for _, row in sample.iterrows():
        records.append({
            "obsTime":  int(row["wp_time"]),
            "lat":      row["latitude"],
            "lon":      row["longitude"],
            "flightLevel": row["baro_altitude_ft"] // 100,
            "tbInt1":   TURB_LABELS.get(int(row["turb_intensity_num"]), "NEG"),
            "icaoId":   row["callsign"],
            "query_grid_lat": round(row["latitude"] / 2) * 2,
            "query_grid_lon": round(row["longitude"] / 2) * 2,
        })
    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main(n_flights: int = 600, seed: int = 42):
    rng = np.random.default_rng(seed)
    random.seed(seed)

    log.info("=" * 64)
    log.info("Realistic synthetic data generator — Flight Rerouting Project")
    log.info(f"Flights: {n_flights}  |  Seed: {seed}")
    log.info("=" * 64)

    flights_per_corridor = n_flights // len(CORRIDORS)
    all_waypoints_global = []
    all_sigmets          = []

    base_ts = int(datetime.now(timezone.utc).timestamp()) - 6 * 3600

    for corridor in CORRIDORS:
        name, lat_min, lat_max, lon_min, lon_max = corridor
        log.info(f"\n[{name}] Generating weather world...")
        weather = WeatherWorld(corridor, rng)

        # Export SIGMETs from this corridor's weather
        all_sigmets.extend(weather.get_sigmets())

        log.info(f"[{name}] Simulating {flights_per_corridor} flights...")
        corridor_wps = []
        rerouted_count = 0

        for i in range(flights_per_corridor):
            icao24   = f"r{rng.integers(0x100000, 0xFFFFFF):06x}"
            callsign = f"{''.join(rng.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 3))}{rng.integers(100,999)}"
            alt_ft   = int(rng.choice(CRUISE_ALTS_FT, p=CRUISE_WEIGHTS))

            # Stagger departure times across the simulation window
            start_ts = base_ts + int(rng.uniform(0, 5 * 3600))

            try:
                orig, dest = pick_airport_pair(name, rng)
            except Exception:
                continue

            wps = simulate_flight(
                icao24, callsign, name, orig, dest,
                alt_ft, start_ts, weather, rng
            )
            if wps:
                corridor_wps.extend(wps)
                if any(w["rerouted"] for w in wps):
                    rerouted_count += 1

        df = pd.DataFrame(corridor_wps)
        out = OPENSKY_DIR / f"trajectories_{name}.csv"
        df.to_csv(out, index=False)

        rerouted_flights = rerouted_count
        total_wps   = len(df)
        rerouted_wps = df["rerouted"].sum() if "rerouted" in df.columns else 0
        log.info(f"  [{name}] {flights_per_corridor} flights -> {total_wps} waypoints")
        log.info(f"  [{name}] Flights with rerouting: {rerouted_flights}/{flights_per_corridor} "
                 f"({100*rerouted_flights/max(1,flights_per_corridor):.1f}%)")
        log.info(f"  [{name}] Rerouted waypoints: {rerouted_wps}/{total_wps} "
                 f"({100*rerouted_wps/max(1,total_wps):.1f}%)")
        log.info(f"  [{name}] Saved -> {out}")
        all_waypoints_global.extend(corridor_wps)

    # ── Build and save NOAA-format side files ──────────────────────────────
    log.info("\nBuilding METAR / PIREP / SIGMET auxiliary files...")
    all_wp_df = pd.DataFrame(all_waypoints_global)

    metars = build_metars(all_wp_df)
    metars.to_csv(NOAA_DIR / "metars_all.csv", index=False)
    log.info(f"  METARs: {len(metars)} records -> {NOAA_DIR / 'metars_all.csv'}")

    pireps = build_pireps(all_wp_df)
    pireps.to_csv(NOAA_DIR / "pireps_all.csv", index=False)
    log.info(f"  PIREPs: {len(pireps)} records -> {NOAA_DIR / 'pireps_all.csv'}")

    sigmet_df = pd.DataFrame(all_sigmets)
    sigmet_df.to_csv(NOAA_DIR / "sigmets_all.csv", index=False)
    log.info(f"  SIGMETs: {len(sigmet_df)} records -> {NOAA_DIR / 'sigmets_all.csv'}")

    # Empty stubs for files the pipeline expects but we don't need to synthesize
    pd.DataFrame().to_csv(NOAA_DIR / "gairmets_all.csv", index=False)
    pd.DataFrame().to_csv(NOAA_DIR / "tafs_all.csv",     index=False)

    # ── Summary statistics ─────────────────────────────────────────────────
    log.info("\n" + "=" * 64)
    log.info("GENERATION COMPLETE")
    log.info("=" * 64)
    total     = len(all_wp_df)
    rerouted  = all_wp_df["rerouted"].sum()
    log.info(f"  Total waypoints : {total:,}")
    log.info(f"  Rerouted (label=1): {rerouted:,} ({100*rerouted/total:.1f}%)")
    log.info(f"  Not rerouted (label=0): {total-rerouted:,} ({100*(total-rerouted)/total:.1f}%)")

    # Verify causal signal is present (basic sanity check)
    rerouted_mean_wind = all_wp_df[all_wp_df["rerouted"]==1]["wind_speed_kt"].mean()
    clear_mean_wind    = all_wp_df[all_wp_df["rerouted"]==0]["wind_speed_kt"].mean()
    rerouted_mean_conv = all_wp_df[all_wp_df["rerouted"]==1]["convective_flag"].mean()
    clear_mean_conv    = all_wp_df[all_wp_df["rerouted"]==0]["convective_flag"].mean()
    log.info("\n  Causal signal check (rerouted vs not rerouted):")
    log.info(f"    Mean wind speed — rerouted: {rerouted_mean_wind:.1f} kt | "
             f"clear: {clear_mean_wind:.1f} kt")
    log.info(f"    Convective flag — rerouted: {rerouted_mean_conv:.2%} | "
             f"clear: {clear_mean_conv:.2%}")
    rerouted_mean_turb = all_wp_df[all_wp_df["rerouted"]==1]["turb_intensity_num"].mean()
    clear_mean_turb    = all_wp_df[all_wp_df["rerouted"]==0]["turb_intensity_num"].mean()
    log.info(f"    Turbulence index — rerouted: {rerouted_mean_turb:.2f} | "
             f"clear: {clear_mean_turb:.2f} "
             f"(ratio {rerouted_mean_turb/max(clear_mean_turb, 0.01):.1f}x)")
    if rerouted_mean_turb > clear_mean_turb * 1.5:
        log.info("  [OK] Causal signal looks healthy")
    else:
        log.warning("  [WARN] Weak signal - consider increasing N_STORM_CELLS or storm intensity")

    log.info("\nNext step: python 03_preprocess.py")
    log.info("  Note: weather is pre-joined in trajectory CSVs.")
    log.info("  The preprocess script will use the inline weather columns directly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flights", type=int, default=600,
                        help="Total number of flights to simulate (default: 600)")
    parser.add_argument("--seed",    type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()
    main(args.flights, args.seed)