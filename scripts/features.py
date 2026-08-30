"""Feature engineering for reception prop modeling."""

import numpy as np
import pandas as pd


def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=1).mean()


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


def merge_schedule_home(df: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Add home indicator from schedule data."""
    sched = schedules[["season", "week", "home_team", "away_team"]].drop_duplicates()
    home_rows = sched.rename(columns={"home_team": "team"}).assign(home=1)[["season", "week", "team", "home"]]
    away_rows = sched.rename(columns={"away_team": "team"}).assign(home=0)[["season", "week", "team", "home"]]
    loc = pd.concat([home_rows, away_rows], ignore_index=True)
    return df.merge(loc, on=["season", "week", "team"], how="left").fillna({"home": 0})


def prepare_training_frame(stats: pd.DataFrame, snaps: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Full training dataset with features and target."""
    df = build_player_game_features(stats, snaps)
    df = merge_schedule_home(df, schedules)
    df["position_te"] = (df["position"] == "TE").astype(int)
    df = df.dropna(subset=["receptions", "targets_l5", "snap_pct"])
    return df


def latest_player_features(
    stats: pd.DataFrame, snaps: pd.DataFrame, season: int, week: int
) -> pd.DataFrame:
    """Most recent feature row per player entering the given week."""
    sched_week = stats[(stats["season"] == season) & (stats["week"] < week)]
    if sched_week.empty:
        sched_week = stats[stats["season"] == season]

    df = build_player_game_features(sched_week, snaps)
    df = df.sort_values(["player_id", "season", "week"])
    latest = df.groupby("player_id", as_index=False).tail(1).copy()
    latest["season"] = season
    latest["week"] = week
    latest["position_te"] = (latest["position"] == "TE").astype(int)
    return latest


def american_to_implied_prob(american: int | float) -> float:
    """Convert American odds to implied probability (no vig removal)."""
    if american >= 0:
        return 100 / (american + 100)
    return abs(american) / (abs(american) + 100)


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
    # statsmodels NB parameterization: var = mu + alpha * mu^2
    # Convert to scipy nbinom: n = 1/alpha, p = n / (n + mu)
    n = 1.0 / alpha
    p = n / (n + mu)
    threshold = int(np.floor(line)) + 1  # over 4.5 means 5+
    return float(1.0 - nbinom.cdf(threshold - 1, n, p))
