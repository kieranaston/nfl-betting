"""Pull raw NFL data via nflreadpy and build training set."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import nflreadpy as nfl
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_DIR, TRAINING_SEASONS, TRAINING_SET_FILE
from features import prepare_training_frame


def pull_raw_data() -> dict[str, pd.DataFrame]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    stats = nfl.load_player_stats(seasons=TRAINING_SEASONS, summary_level="week").to_pandas()
    snaps = nfl.load_snap_counts(seasons=TRAINING_SEASONS).to_pandas()
    schedules = nfl.load_schedules(seasons=TRAINING_SEASONS).to_pandas()
    injuries = nfl.load_injuries(seasons=TRAINING_SEASONS).to_pandas()

    stats.to_csv(DATA_DIR / "player_stats.csv", index=False)
    snaps.to_csv(DATA_DIR / "snap_counts.csv", index=False)
    schedules.to_csv(DATA_DIR / "schedules.csv", index=False)
    injuries.to_csv(DATA_DIR / "injuries.csv", index=False)

    meta = {
        "pulled_at": datetime.now(timezone.utc).isoformat(),
        "seasons": TRAINING_SEASONS,
        "stats_rows": len(stats),
        "snaps_rows": len(snaps),
    }
    (DATA_DIR / "pull_meta.json").write_text(json.dumps(meta, indent=2))

    return {"stats": stats, "snaps": snaps, "schedules": schedules, "injuries": injuries}


def build_training_set(stats: pd.DataFrame, snaps: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    training = prepare_training_frame(stats, snaps, schedules)
    training.to_csv(TRAINING_SET_FILE, index=False)
    return training


def main() -> None:
    raw = pull_raw_data()
    training = build_training_set(raw["stats"], raw["snaps"], raw["schedules"])
    print(f"Pulled data for seasons {TRAINING_SEASONS}")
    print(f"Training set: {len(training)} rows -> {TRAINING_SET_FILE}")


if __name__ == "__main__":
    main()
