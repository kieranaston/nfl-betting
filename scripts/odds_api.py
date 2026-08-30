"""The Odds API client for NFL player reception props."""

import os
import time
from typing import Any

import requests

from config import ODDS_API_BASE, ODDS_MARKET, ODDS_REGIONS, ODDS_SPORT

# Display names for common bookmaker keys
BOOK_TITLES = {
    "draftkings": "DraftKings",
    "fanduel": "FanDuel",
    "betmgm": "BetMGM",
    "caesars": "Caesars",
    "pointsbetus": "PointsBet",
    "betrivers": "BetRivers",
    "williamhill_us": "Caesars",
    "bovada": "Bovada",
    "betonlineag": "BetOnline",
    "mybookieag": "MyBookie",
    "lowvig": "LowVig",
    "betus": "BetUS",
    "superbook": "SuperBook",
    "wynnbet": "WynnBET",
    "unibet_us": "Unibet",
    "fanatics": "Fanatics",
}


def book_label(key: str | None, title: str | None = None) -> str:
    if title:
        return title
    if not key:
        return ""
    return BOOK_TITLES.get(key, key.replace("_", " ").title())


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
            book_title = bookmaker.get("title") or book_label(book_key)
            for market in bookmaker.get("markets", []):
                if market["key"] != ODDS_MARKET:
                    continue
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
                            "book_title": book_title,
                            "line": outcome.get("point"),
                        }
                    by_player[player][f"{side}_price"] = outcome.get("price")
                    if outcome.get("point") is not None:
                        by_player[player]["line"] = outcome["point"]

                props.extend(by_player.values())

        time.sleep(0.3)

    return props


def consensus_lines(props: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Aggregate props to consensus line per player (median line, best over/under prices).
    Tracks which book offered the median line and the best prices.
    """
    from statistics import median

    by_player: dict[str, list] = {}
    for p in props:
        key = p["player"].lower()
        by_player.setdefault(key, []).append(p)

    consensus: dict[str, dict[str, Any]] = {}
    for key, rows in by_player.items():
        lined = [r for r in rows if r.get("line") is not None]
        if not lined:
            continue

        line = median(r["line"] for r in lined)
        # Prefer a book that posts exactly the median line
        line_matches = [r for r in lined if r["line"] == line]
        line_src = line_matches[0] if line_matches else lined[0]

        over_rows = [r for r in rows if r.get("over_price") is not None]
        under_rows = [r for r in rows if r.get("under_price") is not None]
        best_over = max(over_rows, key=lambda r: r["over_price"]) if over_rows else None
        best_under = max(under_rows, key=lambda r: r["under_price"]) if under_rows else None

        consensus[key] = {
            "player": rows[0]["player"],
            "line": line,
            "line_book": line_src.get("book"),
            "line_book_title": book_label(line_src.get("book"), line_src.get("book_title")),
            "over_price": best_over["over_price"] if best_over else None,
            "over_book": best_over.get("book") if best_over else None,
            "over_book_title": book_label(best_over.get("book"), best_over.get("book_title")) if best_over else None,
            "under_price": best_under["under_price"] if best_under else None,
            "under_book": best_under.get("book") if best_under else None,
            "under_book_title": book_label(best_under.get("book"), best_under.get("book_title")) if best_under else None,
            "event_id": rows[0]["event_id"],
            "home_team": rows[0]["home_team"],
            "away_team": rows[0]["away_team"],
            "commence_time": rows[0]["commence_time"],
            "num_books": len({r.get("book") for r in rows if r.get("book")}),
            "books": sorted({book_label(r.get("book"), r.get("book_title")) for r in rows if r.get("book")}),
        }
    return consensus
