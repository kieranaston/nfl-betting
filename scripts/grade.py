"""Grade the previous completed week: MAE/bias on μ, Brier/log loss on P(over).

Scores every prop in the week's odds snapshot (full model slate), not the board shortlist.
Expects fresh nflverse CSVs on disk (run pull_data.py first in the Wed cycle).
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PREDICTIONS_DIR, RESULTS_DIR, SUMMARY_FILE
from pull_data import load_raw_data

PROB_EPS = 1e-6


def previous_completed_week(schedules: pd.DataFrame) -> tuple[int, int] | None:
    """Latest REG week whose last gameday is at least 1 day in the past."""
    now = pd.Timestamp.now(tz="UTC")
    sched = schedules.copy()
    sched["gameday"] = pd.to_datetime(sched["gameday"], utc=True)
    reg = sched[sched["game_type"] == "REG"]
    if reg.empty:
        return None

    completed: list[tuple[int, int]] = []
    for (season, week), group in reg.groupby(["season", "week"]):
        last_gameday = group["gameday"].max()
        if last_gameday + pd.Timedelta(days=1) <= now:
            completed.append((int(season), int(week)))

    if not completed:
        return None
    return max(completed)


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


def binary_log_loss(p_over: float, over_hit: bool) -> float:
    p = min(max(p_over, PROB_EPS), 1.0 - PROB_EPS)
    return -math.log(p if over_hit else (1.0 - p))


def grade_prop(prop: dict, actual: float) -> dict | None:
    """Score one snapshot prop. Returns None if required fields missing."""
    mu = prop.get("model_mu")
    line = prop.get("line")
    p_over = prop.get("p_over")
    if mu is None or line is None or p_over is None:
        return None

    mu = float(mu)
    line = float(line)
    p_over = float(p_over)
    error = mu - actual
    abs_error = abs(error)

    push = actual == line
    over_hit = None if push else actual > line
    brier = None
    log_loss = None
    if over_hit is not None:
        outcome = 1.0 if over_hit else 0.0
        brier = (p_over - outcome) ** 2
        log_loss = binary_log_loss(p_over, over_hit)

    return {
        "player": prop.get("player"),
        "player_id": prop.get("player_id"),
        "position": prop.get("position"),
        "team": prop.get("team"),
        "line": line,
        "model_mu": round(mu, 3),
        "p_over": round(p_over, 4),
        "actual_receptions": actual,
        "error": round(error, 3),
        "abs_error": round(abs_error, 3),
        "push": push,
        "over_hit": over_hit,
        "brier": round(brier, 4) if brier is not None else None,
        "log_loss": round(log_loss, 4) if log_loss is not None else None,
        "matchup": prop.get("matchup"),
        "commence_time": prop.get("commence_time"),
        "event_id": prop.get("event_id"),
    }


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def grade_week(snapshot_path: Path, stats: pd.DataFrame) -> dict:
    data = json.loads(snapshot_path.read_text())
    season = data["season"]
    week = data["week"]

    graded: list[dict] = []
    skipped = 0
    for prop in data.get("props") or []:
        player_id = prop.get("player_id")
        if not player_id:
            skipped += 1
            continue
        actual = get_actual_receptions(stats, player_id, season, week)
        if actual is None:
            skipped += 1
            continue
        row = grade_prop(prop, actual)
        if row is None:
            skipped += 1
            continue
        graded.append(row)

    abs_errors = [g["abs_error"] for g in graded]
    errors = [g["error"] for g in graded]
    sq_errors = [e * e for e in errors]
    briers = [g["brier"] for g in graded if g["brier"] is not None]
    log_losses = [g["log_loss"] for g in graded if g["log_loss"] is not None]

    rmse = round(math.sqrt(sum(sq_errors) / len(sq_errors)), 4) if sq_errors else None

    return {
        "season": season,
        "week": week,
        "graded_at": datetime.now(timezone.utc).isoformat(),
        "n_props": len(data.get("props") or []),
        "n_scored": len(graded),
        "n_skipped": skipped,
        "mae": mean(abs_errors),
        "rmse": rmse,
        "bias": mean(errors),
        "brier": mean(briers),
        "log_loss": mean(log_losses),
        "props": graded,
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
        summary = {"weekly": []}

    summary["weekly"] = [
        w
        for w in summary.get("weekly", [])
        if not (w["season"] == new_week["season"] and w["week"] == new_week["week"])
    ]
    summary["weekly"].append(
        {
            "season": new_week["season"],
            "week": new_week["week"],
            "mae": new_week["mae"],
            "rmse": new_week["rmse"],
            "bias": new_week["bias"],
            "brier": new_week["brier"],
            "log_loss": new_week["log_loss"],
            "n_scored": new_week["n_scored"],
        }
    )
    summary["weekly"] = sorted(summary["weekly"], key=lambda x: (x["season"], x["week"]))

    def series(field: str) -> list[dict]:
        out = []
        for w in summary["weekly"]:
            if w.get(field) is None:
                continue
            out.append({"week": f"{w['season']}-W{w['week']:02d}", field: w[field]})
        return out

    summary["cumulative_mae"] = series("mae")
    summary["cumulative_bias"] = series("bias")
    summary["cumulative_brier"] = series("brier")
    summary["cumulative_log_loss"] = series("log_loss")

    scored = [w for w in summary["weekly"] if w.get("n_scored")]
    summary["mae"] = mean([w["mae"] for w in scored if w.get("mae") is not None])
    summary["bias"] = mean([w["bias"] for w in scored if w.get("bias") is not None])
    summary["brier"] = mean([w["brier"] for w in scored if w.get("brier") is not None])
    summary["log_loss"] = mean([w["log_loss"] for w in scored if w.get("log_loss") is not None])
    summary["n_scored"] = sum(w.get("n_scored") or 0 for w in summary["weekly"])
    summary["updated_at"] = datetime.now(timezone.utc).isoformat()

    SUMMARY_FILE.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    raw = load_raw_data()
    target = previous_completed_week(raw["schedules"])
    if target is None:
        print("No completed REG week to grade yet.")
        return

    season, week = target
    snapshot_path = PREDICTIONS_DIR / f"week_{season}_{week:02d}_odds_snapshot.json"
    if not snapshot_path.exists():
        print(f"No odds snapshot for {season} week {week:02d} at {snapshot_path}")
        return

    print(f"Grading {season} week {week:02d} from {snapshot_path.name}")
    graded = grade_week(snapshot_path, raw["stats"])
    summary = update_summary(graded)
    print(
        f"Scored {graded['n_scored']}/{graded['n_props']} props "
        f"(skipped {graded['n_skipped']}): "
        f"MAE={graded['mae']}, bias={graded['bias']}, "
        f"Brier={graded['brier']}, log_loss={graded['log_loss']}"
    )
    print(f"Season MAE={summary.get('mae')}, Brier={summary.get('brier')}")


if __name__ == "__main__":
    main()
