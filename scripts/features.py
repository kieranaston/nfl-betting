"""Feature engineering for reception prop modeling."""

import numpy as np
import pandas as pd

from config import SEASON_INDEX_BASE


def season_index(season: int | float) -> float:
    """Continuous year index: 2022 → 0, 2023 → 1, …, 2026 → 4."""
    return float(season) - SEASON_INDEX_BASE


def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=1).mean()


def american_to_decimal(american: int | float) -> float:
    """Convert American odds to decimal payout (includes stake)."""
    if american >= 0:
        return 1 + american / 100
    return 1 + 100 / abs(american)


def american_to_implied_prob(american: int | float) -> float:
    """Convert American odds to implied probability (includes vig)."""
    return 1 / american_to_decimal(american)


def expected_value(model_prob: float, american: int | float) -> float:
    """
    +EV per $1 staked (decimal ROI).
    Uses posted American odds, so vig is reflected in the payout side.
    Example: 0.05 means +5% expected return on each $1 bet.
    """
    return model_prob * american_to_decimal(american) - 1


def build_defense_funnel(stats: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling L5 WR/TE targets allowed per defense, ranked weekly.
    Higher rank (closer to 32) = more pass funnel (allows more WR/TE targets).
    """
    wrte = stats[(stats["season_type"] == "REG") & (stats["position"].isin(["WR", "TE"]))]
    allowed = (
        wrte.groupby(["season", "week", "opponent_team"], as_index=False)["targets"]
        .sum()
        .rename(columns={"opponent_team": "def_team", "targets": "wrte_targets_allowed"})
    )
    allowed = allowed.sort_values(["def_team", "season", "week"])
    allowed["wrte_targets_l5"] = allowed.groupby("def_team")["wrte_targets_allowed"].transform(
        lambda s: _rolling_mean(s, 5)
    )
    allowed["opp_pass_funnel_rank"] = allowed.groupby(["season", "week"])["wrte_targets_l5"].rank(
        ascending=True, method="average"
    )
    return allowed[["season", "week", "def_team", "opp_pass_funnel_rank", "wrte_targets_l5"]]


def build_team_game_context(schedules: pd.DataFrame) -> pd.DataFrame:
    """Spread, total, and weather features per team-game from schedules."""
    sched = schedules[schedules["game_type"] == "REG"][
        ["season", "week", "home_team", "away_team", "spread_line", "total_line", "temp", "wind", "roof"]
    ].copy()

    home = sched.rename(columns={"home_team": "team"}).assign(
        home=1,
        team_spread=sched["spread_line"].values,
    )
    away = sched.rename(columns={"away_team": "team"}).assign(
        home=0,
        team_spread=-sched["spread_line"].values,
    )
    ctx = pd.concat([home, away], ignore_index=True)
    ctx["outdoor"] = ctx["roof"].isin(["outdoors", "open"]).astype(int)
    ctx["temp"] = ctx["temp"].fillna(65)
    ctx["wind"] = ctx["wind"].fillna(0)
    return ctx[
        ["season", "week", "team", "home", "team_spread", "total_line", "temp", "wind", "outdoor"]
    ]


def merge_matchup_features(df: pd.DataFrame, stats: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Attach spread, total, weather, and opponent pass-funnel rank."""
    ctx = build_team_game_context(schedules)
    funnel = build_defense_funnel(stats)

    out = df.merge(ctx, on=["season", "week", "team"], how="left", suffixes=("", "_ctx"))
    if "home_ctx" in out.columns:
        out["home"] = out["home"].fillna(out["home_ctx"])
        out = out.drop(columns=["home_ctx"])

    if "opponent" not in out.columns and "opponent_team" in out.columns:
        out = out.rename(columns={"opponent_team": "opponent"})
    elif "opponent_team" in out.columns:
        out = out.drop(columns=["opponent_team"])
    out = out.merge(
        funnel.rename(columns={"def_team": "opponent"}),
        on=["season", "week", "opponent"],
        how="left",
    )
    out["team_spread"] = out["team_spread"].fillna(0)
    median_total = out["total_line"].median()
    if pd.isna(median_total):
        median_total = 44.0
    out["total_line"] = out["total_line"].fillna(median_total)
    out["opp_pass_funnel_rank"] = out["opp_pass_funnel_rank"].fillna(16)
    out["temp"] = out["temp"].fillna(65)
    out["wind"] = out["wind"].fillna(0)
    out["outdoor"] = out["outdoor"].fillna(0)
    out["home"] = out["home"].fillna(0)
    return out


def build_player_game_features(stats: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    """Build per-game feature rows for WR/TE with rolling stats."""
    df = stats.copy()
    df = df[(df["season_type"] == "REG") & (df["position"].isin(["WR", "TE"]))].copy()
    df = df.sort_values(["player_id", "season", "week"])

    snap_cols = ["season", "week", "player", "offense_pct"]
    snap_sub = snaps[snap_cols].rename(columns={"player": "player_display_name", "offense_pct": "snap_pct"})
    df = df.merge(snap_sub, on=["season", "week", "player_display_name"], how="left")
    df["snap_pct"] = df["snap_pct"].fillna(0)

    grouped = df.groupby("player_id", group_keys=False)
    df["targets_l3"] = grouped["targets"].apply(lambda s: _rolling_mean(s, 3))
    df["targets_l5"] = grouped["targets"].apply(lambda s: _rolling_mean(s, 5))
    df["receptions_l3"] = grouped["receptions"].apply(lambda s: _rolling_mean(s, 3))
    df["receptions_l5"] = grouped["receptions"].apply(lambda s: _rolling_mean(s, 5))
    df["games_played"] = grouped.cumcount()

    df = df.dropna(subset=["targets_l5"])
    df = df[df["targets_l5"] >= 5.0]
    return df


def prepare_training_frame(stats: pd.DataFrame, snaps: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Full training dataset with features and target."""
    df = build_player_game_features(stats, snaps)
    df = merge_matchup_features(df, stats, schedules)
    df["position_te"] = (df["position"] == "TE").astype(int)
    df["season_idx"] = df["season"].map(season_index)
    df = df.dropna(subset=["receptions", "targets_l5", "snap_pct"])
    return df


def attach_upcoming_opponent(features: pd.DataFrame, schedules: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Set opponent team for players entering the given week."""
    sched = schedules[
        (schedules["season"] == season) & (schedules["week"] == week) & (schedules["game_type"] == "REG")
    ]
    home = sched[["home_team", "away_team"]].rename(
        columns={"home_team": "team", "away_team": "opponent"}
    )
    away = sched[["away_team", "home_team"]].rename(
        columns={"away_team": "team", "home_team": "opponent"}
    )
    opponents = pd.concat([home, away], ignore_index=True)
    return features.merge(opponents, on="team", how="left")


def games_before_week(stats: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """
    All regular-season player games strictly before (season, week),
    including prior seasons — so Week 1 uses end of last year.
    """
    prior = stats[
        (stats["season_type"] == "REG")
        & (
            (stats["season"] < season)
            | ((stats["season"] == season) & (stats["week"] < week))
        )
    ].copy()
    return prior


def latest_player_features(
    stats: pd.DataFrame, snaps: pd.DataFrame, schedules: pd.DataFrame, season: int, week: int
) -> pd.DataFrame:
    """Most recent feature row per player entering the given week (cross-season)."""
    history = games_before_week(stats, season, week)
    if history.empty:
        history = stats[stats["season_type"] == "REG"].copy()

    df = build_player_game_features(history, snaps)
    df = df.sort_values(["player_id", "season", "week"])
    latest = df.groupby("player_id", as_index=False).tail(1).copy()

    # Keep players with a REG game in the prior or current season (Week 1 uses last year)
    latest = latest[latest["season"] >= season - 1].copy()

    latest["season"] = season
    latest["week"] = week
    latest["season_idx"] = season_index(season)
    latest["position_te"] = (latest["position"] == "TE").astype(int)
    latest = attach_upcoming_opponent(latest, schedules, season, week)
    latest = merge_matchup_features(latest, stats, schedules)
    return latest


def model_feature_row(row: pd.Series, season: int) -> dict:
    """Feature dict for a single prediction row."""
    return {
        "targets_l5": row["targets_l5"],
        "receptions_l5": row["receptions_l5"],
        "snap_pct": row["snap_pct"],
        "position_te": row["position_te"],
        "home": float(row["home"]) if pd.notna(row.get("home")) else 0.0,
        "team_spread": float(row["team_spread"]) if pd.notna(row.get("team_spread")) else 0.0,
        "total_line": float(row["total_line"]) if pd.notna(row.get("total_line")) else 44.0,
        "opp_pass_funnel_rank": float(row["opp_pass_funnel_rank"]) if pd.notna(row.get("opp_pass_funnel_rank")) else 16.0,
        "wind": float(row["wind"]) if pd.notna(row.get("wind")) else 0.0,
        "outdoor": float(row["outdoor"]) if pd.notna(row.get("outdoor")) else 0.0,
        "season_idx": float(row["season_idx"]) if pd.notna(row.get("season_idx")) else season_index(season),
    }


def nb_over_prob(mu: float, alpha: float, line: float) -> float:
    """
    P(receptions > line) for negative binomial with mean mu and dispersion alpha.
    Line is typically x.5 (e.g. 4.5 -> need 5+ receptions for over).
    """
    from scipy.stats import nbinom

    if mu <= 0:
        return 0.0
    if alpha <= 0:
        alpha = 1e-6
    n = 1.0 / alpha
    p = n / (n + mu)
    threshold = int(np.floor(line)) + 1
    return float(1.0 - nbinom.cdf(threshold - 1, n, p))
