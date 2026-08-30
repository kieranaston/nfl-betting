"""Shared paths and constants."""

from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
PREDICTIONS_DIR = ROOT / "predictions"
RESULTS_DIR = ROOT / "results"
SITE_DIR = ROOT / "site"

POSITIONS = ("WR", "TE")
MIN_AVG_TARGETS = 5.0
MIN_EV = 0.04  # minimum +EV per $1 staked (4%; range 3-5%)
TRAINING_SEASONS = [2022, 2023, 2024, 2025]
SEASON_INDEX_BASE = TRAINING_SEASONS[0]  # season_idx = season - base (continuous year)

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_SPORT = "americanfootball_nfl"
ODDS_MARKET = "player_receptions"
ODDS_REGIONS = "us"

MODEL_FILE = MODEL_DIR / "receptions_glm.pkl"
MODEL_META_FILE = MODEL_DIR / "model_meta.json"
TRAINING_SET_FILE = DATA_DIR / "training_set.csv"
SUMMARY_FILE = RESULTS_DIR / "summary.json"
