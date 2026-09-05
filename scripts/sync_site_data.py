"""Copy results and predictions JSON into site/data for GitHub Pages."""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PREDICTIONS_DIR, RESULTS_DIR, SITE_DIR


def sync() -> None:
    dest = SITE_DIR / "data"
    dest.mkdir(parents=True, exist_ok=True)

    summary = RESULTS_DIR / "summary.json"
    if summary.exists():
        shutil.copy(summary, dest / "summary.json")
    else:
        (dest / "summary.json").write_text(
            json.dumps(
                {
                    "mae": None,
                    "bias": None,
                    "brier": None,
                    "log_loss": None,
                    "n_scored": 0,
                    "cumulative_mae": [],
                    "cumulative_bias": [],
                    "cumulative_brier": [],
                    "cumulative_log_loss": [],
                    "weekly": [],
                },
                indent=2,
            )
        )

    picks_files = sorted(
        PREDICTIONS_DIR.glob("week_*_picks.json"),
        key=lambda p: _week_file_key(p),
    )
    for pf in picks_files[-4:]:  # keep last 4 weeks on dashboard
        shutil.copy(pf, dest / pf.name)
    if picks_files:
        shutil.copy(picks_files[-1], dest / "latest_picks.json")

    weekly_dir = RESULTS_DIR / "weekly"
    if weekly_dir.exists():
        for wf in sorted(weekly_dir.glob("week_*.json"), key=_week_file_key):
            shutil.copy(wf, dest / wf.name)

    print(f"Synced site data -> {dest}")


def _week_file_key(path: Path) -> tuple[int, int]:
    """Sort week_YYYY_WW_*.json by season, week."""
    parts = path.stem.split("_")
    try:
        return int(parts[1]), int(parts[2])
    except (IndexError, ValueError):
        return (0, 0)


if __name__ == "__main__":
    sync()
