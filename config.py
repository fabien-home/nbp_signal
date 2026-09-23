"""Central configuration for the NBP weather-signal project.

Everything that another module might need to agree on lives here: file paths,
the geographic domain, physical thresholds, the seasonal window, and the
variable lists for MARS requests. Keeping these in one place means the
offline training pipeline (Stage 1) and the daily operational script
(Stage 2) compute *identical* indices - which is essential, because a
mismatch between training features and live features silently destroys a
model's real-world accuracy.

This file contains constants only. No credentials are stored here; MARS/CDS
credentials live in ~/.ecmwfapirc and ~/.cdsapirc (see README).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# PROJECT_ROOT is resolved relative to this file so the code works regardless
# of the directory the user runs it from (scripts are launched from the repo
# root, but we do not want to depend on that).
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # downloaded GRIB / price files (gitignored)
PROCESSED_DIR = DATA_DIR / "processed"  # cleaned indices as Parquet (gitignored)
MODELS_DIR = PROJECT_ROOT / "models_saved"
OUTPUT_DIR = PROJECT_ROOT / "output"

# NBP price data: raw daily CSV in and cleaned Parquet out.
RAW_NBP_FILE = RAW_DIR / "nbp_daily.csv"
PROCESSED_NBP_FILE = PROCESSED_DIR / "nbp_daily.parquet"

# Optional start date for the *usable* price series. The raw export goes back
# to 1997, but early NBP was thinly traded (flat OHLC, zero volume), which is
# poor training material. Set to a date string (e.g. "2005-01-01") to trim, or
# None to keep the full history. This is a configurable choice, not a hardcode:
# Phase 2's summary output lets us decide the right cut before modelling.
NBP_HISTORY_START = None

# ---------------------------------------------------------------------------
# Geographic domain: NW Europe (UK + North Sea + near-continent).
# Chosen so we capture the continental / German-wind drivers that move TTF
# and therefore NBP, not just UK weather. Bounding box, degrees.
#   North/South latitude, West/East longitude.
# ---------------------------------------------------------------------------
DOMAIN_NORTH = 62.0
DOMAIN_SOUTH = 35.0
DOMAIN_WEST = -12.0
DOMAIN_EAST = 20.0
GRID_RESOLUTION = 0.25  # degrees (ERA5 single-levels native grid)

# MARS "area" is ordered [North, West, South, East].
MARS_AREA = [DOMAIN_NORTH, DOMAIN_WEST, DOMAIN_SOUTH, DOMAIN_EAST]
MARS_GRID = [GRID_RESOLUTION, GRID_RESOLUTION]

# ---------------------------------------------------------------------------
# Physical / feature constants
# ---------------------------------------------------------------------------
# Heating Degree Day base temperature (deg C). Below this, buildings need
# heating; the standard UK/European energy-demand threshold is 15.5 C.
HDD_BASE_TEMP_C = 15.5

# Climatology reference period for anomalies (30-year standard).
CLIMATOLOGY_START_YEAR = 1991
CLIMATOLOGY_END_YEAR = 2020

# Day-of-year climatology is smoothed with a centred window (days) so the
# reference curve is not noisy from year-to-year sampling. 15 days is a common,
# gentle choice.
CLIMATOLOGY_SMOOTH_DAYS = 15

# Wind-turbine power-curve reference wind speeds (m/s), used to convert a 100 m
# wind speed into a normalised generation fraction. Generic modern-turbine
# values: below cut-in nothing spins; power rises steeply (~cubically) to the
# rated speed; it is capped at rated output up to cut-out; above cut-out the
# turbine shuts down for safety. This nonlinearity is why we transform wind
# SPEED into generation rather than feeding raw speed to the model.
WIND_CUT_IN_MS = 3.0
WIND_RATED_MS = 12.0
WIND_CUT_OUT_MS = 25.0

# Heating season: the signal is active September-April inclusive. Outside
# this window the daily script reports "no signal - outside heating season".
HEATING_SEASON_MONTHS = {9, 10, 11, 12, 1, 2, 3, 4}

# Round-trip trading cost (fraction) used in the trading-simulation metric.
ROUND_TRIP_COST = 0.0024  # ~0.24% for CMC NBP

# ---------------------------------------------------------------------------
# ECMWF variables (single-levels). Names are the cfgrib/short-name forms.
# ---------------------------------------------------------------------------
ERA5_VARIABLES = [
    "2t",     # 2 metre temperature        -> HDD / temperature anomaly index
    "100u",   # 100 m U wind component     -> wind speed / generation index
    "100v",   # 100 m V wind component     -> wind speed / generation index
    "tcc",    # total cloud cover          -> cloud index (area average)
    "ssrd",   # surface solar radiation down-> solar index (area average)
]

# ---------------------------------------------------------------------------
# Prediction horizon (days ahead).
#   0 = same-day: today's 00z run -> today's settlement move (v1).
# The whole pipeline is written to accept other horizons so v2 can activate
# the 2-4 day forecasts by configuration rather than by rewriting code.
# ---------------------------------------------------------------------------
HORIZON_DAYS = 0

# ---------------------------------------------------------------------------
# Feature flags: v1 = weather-only. These hooks let v2 switch on the
# deferred components (fundamentals, ENS-spread-as-trained-feature,
# multi-horizon) without structural changes.
# ---------------------------------------------------------------------------
ENABLE_FUNDAMENTALS = False        # storage (AGSI+), LNG / pipeline flows
ENABLE_ENS_SPREAD_FEATURE = False  # ENS spread as a *trained* feature (needs reforecasts)
ENABLE_ENS_SPREAD_GATE = True      # ENS spread as an *operational* confidence gate
