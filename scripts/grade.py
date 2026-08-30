"""Grade last week's picks, compute CLV + Brier, update results."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PREDICTIONS_DIR, RESULTS_DIR, SUMMARY_FILE
from features import american_to_implied_prob
from pull_data import pull_raw_data
from predict import current_nfl_week, normalize_name


def find_previous_week_picks(schedules: pd.DataFrame) -> Path | None:
    """Locate the most recent picks file for a completed week."""
    season, current_week = current_nfl_week(schedules)
    # Grade the week before current (or current if games already played)
    for w in range(current_week, 0, -1):
        path = PREDICTIONS_DIR / f"week_{season}_{w:02d}_picks.json"
        if path.exists():
            return path
    # Try prior season week 18
    prev_season = season - 1
    for w in range(18, 0, -1):
        path = PREDICTIONS_DIR / f"week_{prev_season}_{w:02d}_picks.json"
        if path.exists():
            return path
    return None


def grade_pick(pick: dict, actual_receptions: float, closing_line: float | None) -> dict:
    """Grade a single pick with result, Brier contribution, and CLV."""
    line = pick["line"]
    side = pick["pick"]
    model_prob = pick["model_prob"]

    if side == "over":
        hit = actual_receptions > line
        outcome = 1.0 if hit else 0.0
    else:
        hit = actual_receptions < line
        outcome = 1.0 if hit else 0.0

    push = actual_receptions == line
    if push:
        hit = None
        outcome = None

    brier = (model_prob - outcome) ** 2 if outcome is not None else None

    # CLV: closing line movement in our favor (over: closing > pick line; under: closing < pick line)
    clv = None
    if closing_line is not None and closing_line != line:
        if side == "over":
            clv = closing_line - line
        else:
            clv = line - closing_line

    return {
        **pick,
        "actual_receptions": actual_receptions,
        "hit": hit,
        "push": push,
        "brier": round(brier, 4) if brier is not None else None,
        "closing_line": closing_line,
        "clv": round(clv, 2) if clv is not None else 0.0,
    }


def get_actual_receptions(stats: pd.DataFrame, player_id: str, season: int, week: int) -> float | None:
    row = stats[
        (stats["player_id"] == player_id)
        & (stats["season"] == season)
        & (stats["week"] == week)
        & (stats["season_type"] == "REG")
    ]
    if row.empty:
        return None
    return float(row.iloc[0]["receptions"])


def get_closing_line(snapshot_path: Path, player_name: str) -> float | None:
    if not snapshot_path.exists():
        return None
    data = json.loads(snapshot_path.read_text())
    key = normalize_name(player_name)
    for prop in data.get("props", []):
        if normalize_name(prop["player"]) == key:
            return float(prop["line"])
    return None


def grade_week(picks_path: Path, stats: pd.DataFrame) -> dict:
    picks_data = json.loads(picks_path.read_text())
    season = picks_data["season"]
    week = picks_data["week"]
    snapshot_path = PREDICTIONS_DIR / f"week_{season}_{week:02d}_odds_snapshot.json"

    graded = []
    for pick in picks_data.get("picks", []):
        actual = get_actual_receptions(stats, pick["player_id"], season, week)
        if actual is None:
            continue
        closing = get_closing_line(snapshot_path, pick["player"])
        graded.append(grade_pick(pick, actual, closing))

    hits = [g for g in graded if g["hit"] is True]
    losses = [g for g in graded if g["hit"] is False]
    briers = [g["brier"] for g in graded if g["brier"] is not None]
    clvs = [g["clv"] for g in graded if g["clv"] is not None]

    return {
        "season": season,
        "week": week,
        "graded_at": datetime.now(timezone.utc).isoformat(),
        "picks": graded,
        "record": {"wins": len(hits), "losses": len(losses), "pushes": len(graded) - len(hits) - len(losses)},
        "brier_score": round(sum(briers) / len(briers), 4) if briers else None,
        "avg_clv": round(sum(clvs) / len(clvs), 3) if clvs else 0.0,
    }


def update_summary(new_week: dict) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    weekly_dir = RESULTS_DIR / "weekly"
    weekly_dir.mkdir(exist_ok=True)

    week_file = weekly_dir / f"week_{new_week['season']}_{new_week['week']:02d}.json"
    week_file.write_text(json.dumps(new_week, indent=2))

    if SUMMARY_FILE.exists():
        summary = json.loads(SUMMARY_FILE.read_text())
    else:
        summary = {
            "total_wins": 0,
            "total_losses": 0,
            "total_pushes": 0,
            "cumulative_brier": [],
            "cumulative_clv": [],
            "weekly": [],
        }

    # Avoid double-counting same week
    summary["weekly"] = [w for w in summary["weekly"] if not (w["season"] == new_week["season"] and w["week"] == new_week["week"])]
    summary["weekly"].append(
        {
            "season": new_week["season"],
            "week": new_week["week"],
            "wins": new_week["record"]["wins"],
            "losses": new_week["record"]["losses"],
            "brier_score": new_week["brier_score"],
            "avg_clv": new_week["avg_clv"],
        }
    )
    summary["weekly"] = sorted(summary["weekly"], key=lambda x: (x["season"], x["week"]))

    summary["total_wins"] = sum(w["wins"] for w in summary["weekly"])
    summary["total_losses"] = sum(w["losses"] for w in summary["weekly"])
    summary["total_pushes"] = 0

    summary["cumulative_brier"] = [
        {"week": f"{w['season']}-W{w['week']:02d}", "brier": w["brier_score"]} for w in summary["weekly"] if w["brier_score"] is not None
    ]
    summary["cumulative_clv"] = [
        {"week": f"{w['season']}-W{w['week']:02d}", "clv": w["avg_clv"]} for w in summary["weekly"]
    ]

    win_total = summary["total_wins"] + summary["total_losses"]
    summary["win_pct"] = round(summary["total_wins"] / win_total, 3) if win_total else None
    summary["updated_at"] = datetime.now(timezone.utc).isoformat()

    SUMMARY_FILE.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    raw = pull_raw_data()
    picks_path = find_previous_week_picks(raw["schedules"])
    if picks_path is None:
        print("No picks file found to grade.")
        return

    print(f"Grading {picks_path.name}")
    graded = grade_week(picks_path, raw["stats"])
    summary = update_summary(graded)
    print(f"Record: {graded['record']}")
    print(f"Brier: {graded['brier_score']}, Avg CLV: {graded['avg_clv']}")
    print(f"Season totals: {summary['total_wins']}-{summary['total_losses']}")


if __name__ == "__main__":
    main()
