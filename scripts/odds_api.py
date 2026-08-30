"""The Odds API client for NFL player reception props."""

import os
import time
from typing import Any

import requests

from config import ODDS_API_BASE, ODDS_MARKET, ODDS_REGIONS, ODDS_SPORT


def _api_key() -> str:
    key = os.environ.get("ODDS_API_KEY", "")
    if not key:
        raise RuntimeError("ODDS_API_KEY environment variable is not set")
    return key


def get_upcoming_events() -> list[dict[str, Any]]:
    """List upcoming NFL events (free — no quota cost)."""
    url = f"{ODDS_API_BASE}/sports/{ODDS_SPORT}/events"
    resp = requests.get(url, params={"apiKey": _api_key()}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_event_reception_props(event_id: str) -> dict[str, Any]:
    """Fetch player_receptions odds for a single event."""
    url = f"{ODDS_API_BASE}/sports/{ODDS_SPORT}/events/{event_id}/odds"
    params = {
        "apiKey": _api_key(),
        "regions": ODDS_REGIONS,
        "markets": ODDS_MARKET,
        "oddsFormat": "american",
    }
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code == 429:
        time.sleep(2)
        resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_all_reception_props() -> list[dict[str, Any]]:
    """
    Pull player_receptions for all upcoming events.
    Returns flat list of prop dicts with player, line, over/under prices, book, event info.
    """
    events = get_upcoming_events()
    props: list[dict[str, Any]] = []

    for event in events:
        event_id = event["id"]
        commence = event.get("commence_time", "")
        home = event.get("home_team", "")
        away = event.get("away_team", "")

        try:
            odds_data = get_event_reception_props(event_id)
        except requests.HTTPError:
            continue

        for bookmaker in odds_data.get("bookmakers", []):
            book_key = bookmaker["key"]
            for market in bookmaker.get("markets", []):
                if market["key"] != ODDS_MARKET:
                    continue
                # Group outcomes by player (description field)
                by_player: dict[str, dict] = {}
                for outcome in market.get("outcomes", []):
                    player = outcome.get("description", "")
                    if not player:
                        continue
                    side = outcome["name"].lower()
                    if player not in by_player:
                        by_player[player] = {
                            "player": player,
                            "event_id": event_id,
                            "home_team": home,
                            "away_team": away,
                            "commence_time": commence,
                            "book": book_key,
                            "line": outcome.get("point"),
                        }
                    by_player[player][f"{side}_price"] = outcome.get("price")
                    if outcome.get("point") is not None:
                        by_player[player]["line"] = outcome["point"]

                props.extend(by_player.values())

        time.sleep(0.3)  # gentle rate limit

    return props


def consensus_lines(props: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Aggregate props to consensus line per player (median line, best over/under prices).
    Key = lowercase player name.
    """
    from statistics import median

    by_player: dict[str, list] = {}
    for p in props:
        key = p["player"].lower()
        by_player.setdefault(key, []).append(p)

    consensus: dict[str, dict[str, Any]] = {}
    for key, rows in by_player.items():
        lines = [r["line"] for r in rows if r.get("line") is not None]
        if not lines:
            continue
        line = median(lines)
        overs = [r["over_price"] for r in rows if r.get("over_price") is not None]
        unders = [r["under_price"] for r in rows if r.get("under_price") is not None]
        consensus[key] = {
            "player": rows[0]["player"],
            "line": line,
            "over_price": max(overs) if overs else None,
            "under_price": max(unders) if unders else None,
            "event_id": rows[0]["event_id"],
            "home_team": rows[0]["home_team"],
            "away_team": rows[0]["away_team"],
            "commence_time": rows[0]["commence_time"],
            "num_books": len(rows),
        }
    return consensus
