"""Train negative binomial GLM for WR/TE receptions."""

import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import MODEL_DIR, MODEL_FILE, MODEL_META_FILE, TRAINING_SET_FILE
from pull_data import build_training_set, pull_raw_data


FORMULA = (
    "receptions ~ targets_l5 + receptions_l5 + snap_pct + position_te + home"
    " + team_spread + total_line + opp_pass_funnel_rank + wind + outdoor"
    " + season_idx"
)


def estimate_nb_alpha(df: pd.DataFrame, formula: str) -> float:
    """Method-of-moments dispersion from a Poisson fit (var = mu + alpha * mu^2)."""
    pois = smf.glm(formula, data=df, family=sm.families.Poisson()).fit(maxiter=100)
    mu = np.asarray(pois.fittedvalues, dtype=float)
    y = df["receptions"].to_numpy(dtype=float)
    denom = float(np.sum(mu**2))
    if denom <= 0:
        return 0.1
    alpha = float(np.sum((y - mu) ** 2 - mu) / denom)
    return max(alpha, 1e-4)


def train_model(training: pd.DataFrame):
    """Fit negative binomial GLM with estimated alpha; return (result, n_rows, alpha)."""
    df = training.dropna(
        subset=["receptions", "targets_l5", "receptions_l5", "snap_pct", "team_spread", "total_line"]
    ).copy()
    df["snap_pct"] = df["snap_pct"].clip(0, 1)
    if "season_idx" not in df.columns:
        from config import SEASON_INDEX_BASE

        df["season_idx"] = df["season"] - SEASON_INDEX_BASE

    alpha = estimate_nb_alpha(df, FORMULA)
    model = smf.glm(FORMULA, data=df, family=sm.families.NegativeBinomial(alpha=alpha))
    result = model.fit(maxiter=100)
    return result, len(df), alpha


def save_model(result, n_rows: int, alpha: float) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump({"result": result, "alpha": alpha}, f)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "formula": FORMULA,
        "n_rows": n_rows,
        "alpha": alpha,
        "aic": float(result.aic),
        "pseudo_r2": float(getattr(result, "pseudo_rsquared", lambda: 0)()) if hasattr(result, "pseudo_rsquared") else None,
        "params": {k: float(v) for k, v in result.params.items()},
    }
    MODEL_META_FILE.write_text(json.dumps(meta, indent=2))
    print(f"Model saved -> {MODEL_FILE}")
    print(f"AIC: {meta['aic']:.2f}, rows: {n_rows}, alpha: {alpha:.4f}")


def load_model():
    """Return (fitted result, alpha). Supports legacy pickle of bare result."""
    with open(MODEL_FILE, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "result" in obj:
        return obj["result"], float(obj.get("alpha") or 0.1)
    if MODEL_META_FILE.exists():
        meta = json.loads(MODEL_META_FILE.read_text())
        if "alpha" in meta:
            return obj, float(meta["alpha"])
    return obj, 0.1


def main() -> None:
    if TRAINING_SET_FILE.exists():
        training = pd.read_csv(TRAINING_SET_FILE)
    else:
        raw = pull_raw_data()
        training = build_training_set(raw["stats"], raw["snaps"], raw["schedules"])

    result, n_rows, alpha = train_model(training)
    save_model(result, n_rows, alpha)


if __name__ == "__main__":
    main()
