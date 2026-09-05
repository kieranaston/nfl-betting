"""Generate weekly reception prop picks from model + odds.

Merges into the week ledger: events in the current Odds API pull overwrite;
picks for other games (already kicked off / not in this pull) are kept.
Board ranking uses model confidence vs the line (no prices).
"""

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import MIN_SIDE_PROB, MODEL_FILE, PREDICTIONS_DIR
from features import (
    latest_player_features,
    model_feature_row,
    nb_over_prob,
)
from odds_api import consensus_lines, fetch_all_reception_props
from pull_data import load_raw_data
from train import load_model


def json_safe(value):
    """Convert pandas/numpy NA and non-finite floats to JSON-safe None."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (ValueError, AttributeError):
            pass
    return value


def sanitize(obj):
    """Recursively make a structure JSON-serializable for browsers (no NaN)."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return json_safe(obj)


def current_nfl_week(schedules: pd.DataFrame) -> tuple[int, int]:
    """Return (season, week) for the upcoming regular-season slate."""
    now = pd.Timestamp.now(tz="UTC")
    sched = schedules.copy()
    sched["gameday"] = pd.to_datetime(sched["gameday"], utc=True)
    upcoming = sched[(sched["gameday"] >= now - pd.Timedelta(days=1)) & (sched["game_type"] == "REG")]
    if not upcoming.empty:
        row = upcoming.sort_values("gameday").iloc[0]
        return int(row["season"]), int(row["week"])

    reg = sched[sched["game_type"] == "REG"].sort_values("gameday")
    past = reg[reg["gameday"] < now]
    if past.empty:
        row = reg.iloc[0]
        return int(row["season"]), int(row["week"])
    last = past.iloc[-1]
    season, week = int(last["season"]), int(last["week"])
    if week < 18:
        return season, week + 1
    return season + 1, 1


def normalize_name(name: str) -> str:
    return name.lower().strip().replace(".", "").replace("'", "")


def best_model_side(p_over: float, p_under: float) -> tuple[str | None, float | None, float | None]:
    """
    Prefer the higher-probability side vs the line. Filter by MIN_SIDE_PROB.
    Returns (side, model_prob, edge_vs_coin) where edge_vs_coin = model_prob - 0.5.
    """
    if p_over >= p_under:
        side, prob = "over", p_over
    else:
        side, prob = "under", p_under
    if prob < MIN_SIDE_PROB:
        return None, None, None
    return side, prob, prob - 0.5


def load_week_list(path: Path, season: int, week: int, key: str) -> list:
    """Load a list field from an existing week JSON if season/week match."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    if data.get("season") != season or data.get("week") != week:
        return []
    return data.get(key) or []


def merge_by_event(prior: list, fresh: list, refreshed_events: set[str]) -> list:
    """Keep prior rows for events not in this pull; append fresh rows for refreshed events."""
    kept = [row for row in prior if str(row.get("event_id") or "") not in refreshed_events]
    return kept + fresh


def generate_picks(season: int, week: int, raw: dict | None = None) -> dict:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"No model at {MODEL_FILE}. Run train.py first.")

    model, alpha = load_model()

    if raw is None:
        raw = load_raw_data()
    features = latest_player_features(raw["stats"], raw["snaps"], raw["schedules"], season, week)
    consensus = consensus_lines(fetch_all_reception_props())

    matched: list[tuple[pd.Series, dict]] = []
    for _, row in features.iterrows():
        key = normalize_name(row["player_display_name"])
        if key not in consensus:
            continue
        matched.append((row, consensus[key]))

    picks: list[dict] = []
    snapshot: list[dict] = []
    if not matched:
        mus = []
    else:
        feature_frame = pd.DataFrame([model_feature_row(row, season) for row, _ in matched])
        mus = [float(x) for x in model.predict(feature_frame)]

    for (row, market), mu in zip(matched, mus):
        line = float(market["line"])
        p_over = nb_over_prob(mu, alpha, line)
        p_under = 1.0 - p_over
        pick_side, model_prob, edge = best_model_side(p_over, p_under)

        over_price = market.get("over_price")
        under_price = market.get("under_price")
        if pick_side == "over":
            price, price_book, price_book_title, price_link = (
                over_price,
                market.get("over_book"),
                market.get("over_book_title"),
                market.get("over_link"),
            )
        elif pick_side == "under":
            price, price_book, price_book_title, price_link = (
                under_price,
                market.get("under_book"),
                market.get("under_book_title"),
                market.get("under_link"),
            )
        else:
            price = price_book = price_book_title = price_link = None

        prop_record = {
            "player": row["player_display_name"],
            "player_id": row["player_id"],
            "position": row["position"],
            "team": json_safe(row.get("team")) or "",
            "opponent": json_safe(row.get("opponent")) or "",
            "line": line,
            "line_book": market.get("line_book"),
            "line_book_title": market.get("line_book_title"),
            "line_link": market.get("line_link"),
            "model_mu": round(mu, 2),
            "mu_gap": round(mu - line, 2),
            "p_over": round(p_over, 4),
            "p_under": round(p_under, 4),
            "over_price": over_price,
            "over_book": market.get("over_book"),
            "over_book_title": market.get("over_book_title"),
            "over_link": market.get("over_link"),
            "under_price": under_price,
            "under_book": market.get("under_book"),
            "under_book_title": market.get("under_book_title"),
            "under_link": market.get("under_link"),
            "num_books": market.get("num_books"),
            "books": market.get("books"),
            "team_spread": json_safe(row.get("team_spread")),
            "total_line": json_safe(row.get("total_line")),
            "opp_pass_funnel_rank": json_safe(row.get("opp_pass_funnel_rank")),
            "wind": json_safe(row.get("wind")),
            "outdoor": json_safe(row.get("outdoor")),
            "event_id": market["event_id"],
            "matchup": f"{market['away_team']} @ {market['home_team']}",
            "commence_time": market["commence_time"],
        }
        snapshot.append(prop_record)

        if pick_side:
            picks.append(
                {
                    **prop_record,
                    "pick": pick_side,
                    "model_prob": round(model_prob, 4),
                    "edge": round(edge, 4),
                    "price": price,
                    "price_book": price_book,
                    "price_book_title": price_book_title,
                    "price_link": price_link,
                }
            )

    generated_at = datetime.now(timezone.utc).isoformat()
    refreshed_events = {str(p["event_id"]) for p in snapshot if p.get("event_id")}

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    picks_path = PREDICTIONS_DIR / f"week_{season}_{week:02d}_picks.json"
    snapshot_path = PREDICTIONS_DIR / f"week_{season}_{week:02d}_odds_snapshot.json"

    prior_picks = load_week_list(picks_path, season, week, "picks")
    prior_props = load_week_list(snapshot_path, season, week, "props")
    merged_picks = merge_by_event(prior_picks, picks, refreshed_events)
    merged_props = merge_by_event(prior_props, snapshot, refreshed_events)
    kept_picks = len(merged_picks) - len(picks)

    output = {
        "season": season,
        "week": week,
        "generated_at": generated_at,
        "min_side_prob": MIN_SIDE_PROB,
        "picks": sorted(
            merged_picks,
            key=lambda x: (-(x.get("model_prob") or 0), x.get("commence_time") or ""),
        ),
        "n_picks": len(merged_picks),
        "n_props": len(merged_props),
        "n_picks_this_refresh": len(picks),
        "n_events_refreshed": len(refreshed_events),
    }

    picks_path.write_text(json.dumps(sanitize(output), indent=2, allow_nan=False))
    snapshot_path.write_text(
        json.dumps(
            sanitize(
                {
                    "season": season,
                    "week": week,
                    "snapshot_at": generated_at,
                    "props": merged_props,
                    "n_events_refreshed": len(refreshed_events),
                }
            ),
            indent=2,
            allow_nan=False,
        )
    )

    print(
        f"Refresh: {len(picks)} new picks / {len(snapshot)} props across {len(refreshed_events)} events; "
        f"kept {kept_picks} prior picks → {len(merged_picks)} week total "
        f"(min side P {MIN_SIDE_PROB:.0%})"
    )
    print(f"Saved -> {picks_path}")
    return output


def main() -> None:
    raw = load_raw_data()
    season, week = current_nfl_week(raw["schedules"])
    generate_picks(season, week, raw=raw)


if __name__ == "__main__":
    main()
