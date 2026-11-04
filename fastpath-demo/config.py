"""Shared settings for the FastPath finance demo. Every script imports from here."""
from pathlib import Path

import os

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
REFERENCE = DATA / "reference"
HOLDINGS = DATA / "built" / "holdings.parquet"   # 13F positions, shared by both universes (05_download_13f.py)
# Two price universes side by side: unset = the 99-ticker set in data/raw, data/built, results/;
# FASTPATH_UNIVERSE=sp500 = data/sp500/raw, data/sp500/built, results/sp500/
UNIVERSE = os.environ.get("FASTPATH_UNIVERSE", "")
_U = DATA / UNIVERSE if UNIVERSE else DATA
RAW = _U / "raw"              # downloaded prices (and optional earnings dates)
BUILT = _U / "built"          # graph-ready CSVs produced by 02_build_events.py
RESULTS = ROOT / "results" / UNIVERSE if UNIVERSE else ROOT / "results"

# --- universe & period ------------------------------------------------------------
START = "2023-01-01"          # snapshots start ~9 months later; covers AI rally, summer-2024 unwind, April-2025 tariff shock
END = None                    # None = today
EPOCH = "2020-01-01"          # event/observation times are integer days since EPOCH

# --- event construction (all point-in-time: only past data is used at each date) ----
VOL_WINDOW = 60               # trailing trading days used to z-score returns, volume, range
MIN_HISTORY = 40              # an asset needs this many past returns before events are emitted
MOVE_EDGES = [-2.0, -1.0, 1.0, 2.0]   # z-score cut points -> 5 buckets (0 .. 4), 2 = "normal day"
VOL_EDGES = [-1.0, 1.0, 2.0]          # volume z cut points -> 4 buckets (0 .. 3)
MARKET_SHOCK_Z = 2.0          # |z| of the equal-weighted universe return that flags a shock day
MOMENT_MIN_ABS_Z = 1.0        # an event links to a Moment only if |excess z| >= this

# --- snapshots (FastPath base nodes) ------------------------------------------------
LOOKBACK_DAYS = 180           # calendar days FastPath looks back from each observation day
NUM_TIME_ANCHORS = 26         # ~1 anchor per week over the look-back
SNAPSHOT_FREQ = "M"           # month-end snapshots

# --- evaluation (the reveal) ----------------------------------------------------------
FORWARD_DAYS = 60             # trading days after the snapshot used as ground truth
TOP_K = 10                    # neighbours scored per asset
TRAILING_WINDOWS = [30, 120]  # Pearson baselines (30 = the original demo)

# FOMC statement days (editable reference file, see data/reference/fomc_dates.csv)
FOMC_FILE = REFERENCE / "fomc_dates.csv"
UNIVERSE_FILE = REFERENCE / "sp500_constituents.csv"
