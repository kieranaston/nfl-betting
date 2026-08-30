"""Generate weekly reception prop picks from model + odds."""

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import MIN_EV, MODEL_FILE, PREDICTIONS_DIR
from features import (
    american_to_implied_prob,
    expected_value,
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

    # Offseason / schedule not published yet: next REG week after last completed REG game
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


def best_ev_side(p_over: float, p_under: float, over_price, under_price) -> tuple[str | None, float | None, float | None, float | None, int | None]:
    """Return best +EV side (side, model_prob, market_prob, ev, price) if above MIN_EV."""
    best = None

    if over_price is not None:
        ev_over = expected_value(p_over, over_price)
        if ev_over >= MIN_EV and (best is None or ev_over > best[3]):
            best = ("over", p_over, american_to_implied_prob(over_price), ev_over, over_price)

    if under_price is not None:
        ev_under = expected_value(p_under, under_price)
        if ev_under >= MIN_EV and (best is None or ev_under > best[3]):
            best = ("under", p_under, american_to_implied_prob(under_price), ev_under, under_price)

    if best is None:
        return None, None, None, None, None
    return best


def generate_picks(season: int, week: int, raw: dict | None = None) -> dict:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"No model at {MODEL_FILE}. Run train.py first.")

    model, alpha = load_model()

    if raw is None:
        raw = load_raw_data()
    features = latest_player_features(raw["stats"], raw["snaps"], raw["schedules"], season, week)

    props_raw = fetch_all_reception_props()
    consensus = consensus_lines(props_raw)

    picks = []
    snapshot = []

    for _, row in features.iterrows():
        player_name = row["player_display_name"]
        key = normalize_name(player_name)
        if key not in consensus:
            continue

        market = consensus[key]
        line = float(market["line"])

        pred = model.predict(pd.DataFrame([model_feature_row(row, season)]))[0]
        mu = float(pred)
        p_over = nb_over_prob(mu, alpha, line)
        p_under = 1.0 - p_over

        over_price = market.get("over_price")
        under_price = market.get("under_price")

        pick_side, model_prob, market_prob, ev, price = best_ev_side(
            p_over, p_under, over_price, under_price
        )

        price_book = None
        price_book_title = None
        if pick_side == "over":
            price_book = market.get("over_book")
            price_book_title = market.get("over_book_title")
        elif pick_side == "under":
            price_book = market.get("under_book")
            price_book_title = market.get("under_book_title")

        prop_record = {
            "player": player_name,
            "player_id": row["player_id"],
            "position": row["position"],
            "team": json_safe(row.get("team")) or "",
            "opponent": json_safe(row.get("opponent")) or "",
            "line": line,
            "line_book": market.get("line_book"),
            "line_book_title": market.get("line_book_title"),
            "model_mu": round(mu, 2),
            "p_over": round(p_over, 4),
            "p_under": round(p_under, 4),
            "over_price": over_price,
            "over_book": market.get("over_book"),
            "over_book_title": market.get("over_book_title"),
            "under_price": under_price,
            "under_book": market.get("under_book"),
            "under_book_title": market.get("under_book_title"),
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
                    "market_prob": round(market_prob, 4),
                    "ev": round(ev, 4),
                    "edge": round(model_prob - market_prob, 4),
                    "price": price,
                    "price_book": price_book,
                    "price_book_title": price_book_title,
                }
            )

    generated_at = datetime.now(timezone.utc).isoformat()
    output = {
        "season": season,
        "week": week,
        "generated_at": generated_at,
        "min_ev": MIN_EV,
        "picks": sorted(picks, key=lambda x: -x["ev"]),
        "n_picks": len(picks),
        "n_props": len(snapshot),
    }

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    picks_path = PREDICTIONS_DIR / f"week_{season}_{week:02d}_picks.json"
    snapshot_path = PREDICTIONS_DIR / f"week_{season}_{week:02d}_odds_snapshot.json"

    picks_path.write_text(json.dumps(sanitize(output), indent=2, allow_nan=False))
    snapshot_path.write_text(
        json.dumps(
            sanitize({"season": season, "week": week, "snapshot_at": generated_at, "props": snapshot}),
            indent=2,
            allow_nan=False,
        )
    )

    print(f"Generated {len(picks)} picks from {len(snapshot)} props (min EV {MIN_EV:.0%})")
    print(f"Saved -> {picks_path}")
    return output


def main() -> None:
    raw = load_raw_data()
    season, week = current_nfl_week(raw["schedules"])
    generate_picks(season, week, raw=raw)


if __name__ == "__main__":
    main()
