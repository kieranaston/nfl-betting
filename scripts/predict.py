"""Generate weekly reception prop picks from model + odds."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import MIN_EDGE, MODEL_FILE, PREDICTIONS_DIR, TRAINING_SEASONS
from features import american_to_implied_prob, latest_player_features, nb_over_prob
from odds_api import consensus_lines, fetch_all_reception_props
from pull_data import pull_raw_data
from train import load_model


def current_nfl_week(schedules: pd.DataFrame) -> tuple[int, int]:
    """Return (season, week) for the upcoming slate."""
    now = pd.Timestamp.now(tz="UTC")
    sched = schedules.copy()
    sched["gameday"] = pd.to_datetime(sched["gameday"], utc=True)
    upcoming = sched[(sched["gameday"] >= now - pd.Timedelta(days=1)) & (sched["game_type"] == "REG")]
    if upcoming.empty:
        season = int(schedules["season"].max())
        week = int(schedules[schedules["season"] == season]["week"].max())
        return season, week
    row = upcoming.sort_values("gameday").iloc[0]
    return int(row["season"]), int(row["week"])


def normalize_name(name: str) -> str:
    return name.lower().strip().replace(".", "").replace("'", "")


def generate_picks(season: int, week: int) -> dict:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"No model at {MODEL_FILE}. Run train.py first.")

    model = load_model()
    alpha = float(model.scale) if model.scale else 1.0

    raw = pull_raw_data()
    features = latest_player_features(raw["stats"], raw["snaps"], season, week)

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

        pred = model.predict(
            pd.DataFrame(
                [
                    {
                        "targets_l5": row["targets_l5"],
                        "receptions_l5": row["receptions_l5"],
                        "snap_pct": row["snap_pct"],
                        "position_te": row["position_te"],
                        "home": 0,
                        "season": season,
                    }
                ]
            )
        )[0]
        mu = float(pred)
        p_over = nb_over_prob(mu, alpha, line)
        p_under = 1.0 - p_over

        over_price = market.get("over_price")
        under_price = market.get("under_price")

        pick_side = None
        model_prob = None
        market_prob = None
        edge = None
        price = None

        if over_price is not None:
            implied_over = american_to_implied_prob(over_price)
            edge_over = p_over - implied_over
            if edge_over >= MIN_EDGE and (edge is None or edge_over > edge):
                pick_side = "over"
                model_prob = p_over
                market_prob = implied_over
                edge = edge_over
                price = over_price

        if under_price is not None:
            implied_under = american_to_implied_prob(under_price)
            edge_under = p_under - implied_under
            if edge_under >= MIN_EDGE and (edge is None or edge_under > edge):
                pick_side = "under"
                model_prob = p_under
                market_prob = implied_under
                edge = edge_under
                price = under_price

        prop_record = {
            "player": player_name,
            "player_id": row["player_id"],
            "position": row["position"],
            "team": row.get("team", ""),
            "line": line,
            "model_mu": round(mu, 2),
            "p_over": round(p_over, 4),
            "p_under": round(p_under, 4),
            "over_price": over_price,
            "under_price": under_price,
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
                    "edge": round(edge, 4),
                    "price": price,
                }
            )

    generated_at = datetime.now(timezone.utc).isoformat()
    output = {
        "season": season,
        "week": week,
        "generated_at": generated_at,
        "picks": sorted(picks, key=lambda x: -x["edge"]),
        "n_picks": len(picks),
        "n_props": len(snapshot),
    }

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    picks_path = PREDICTIONS_DIR / f"week_{season}_{week:02d}_picks.json"
    snapshot_path = PREDICTIONS_DIR / f"week_{season}_{week:02d}_odds_snapshot.json"

    picks_path.write_text(json.dumps(output, indent=2))
    snapshot_path.write_text(
        json.dumps(
            {"season": season, "week": week, "snapshot_at": generated_at, "props": snapshot},
            indent=2,
        )
    )

    print(f"Generated {len(picks)} picks from {len(snapshot)} props")
    print(f"Saved -> {picks_path}")
    return output


def main() -> None:
    raw = pull_raw_data()
    season, week = current_nfl_week(raw["schedules"])
    generate_picks(season, week)


if __name__ == "__main__":
    main()
