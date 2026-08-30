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
                    "total_wins": 0,
                    "total_losses": 0,
                    "total_pushes": 0,
                    "win_pct": None,
                    "cumulative_brier": [],
                    "cumulative_clv": [],
                    "weekly": [],
                },
                indent=2,
            )
        )

    picks_files = sorted(PREDICTIONS_DIR.glob("week_*_picks.json"))
    for pf in picks_files[-4:]:  # keep last 4 weeks on dashboard
        shutil.copy(pf, dest / pf.name)

    weekly_dir = RESULTS_DIR / "weekly"
    if weekly_dir.exists():
        for wf in sorted(weekly_dir.glob("week_*.json")):
            shutil.copy(wf, dest / wf.name)

    print(f"Synced site data -> {dest}")


if __name__ == "__main__":
    sync()
