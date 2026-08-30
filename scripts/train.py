"""Train negative binomial GLM for WR/TE receptions."""

import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import MODEL_DIR, MODEL_FILE, MODEL_META_FILE, TRAINING_SET_FILE
from pull_data import build_training_set, pull_raw_data


FORMULA = "receptions ~ targets_l5 + receptions_l5 + snap_pct + position_te + home + C(season)"


def train_model(training: pd.DataFrame):
    """Fit negative binomial GLM and return fitted model."""
    df = training.dropna(subset=["receptions", "targets_l5", "receptions_l5", "snap_pct"]).copy()
    df["snap_pct"] = df["snap_pct"].clip(0, 100)

    model = smf.glm(FORMULA, data=df, family=sm.families.NegativeBinomial())
    result = model.fit(maxiter=100)
    return result, len(df)


def save_model(result, n_rows: int) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(result, f)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "formula": FORMULA,
        "n_rows": n_rows,
        "aic": float(result.aic),
        "pseudo_r2": float(getattr(result, "pseudo_rsquared", lambda: 0)()) if hasattr(result, "pseudo_rsquared") else None,
        "params": {k: float(v) for k, v in result.params.items()},
    }
    MODEL_META_FILE.write_text(json.dumps(meta, indent=2))
    print(f"Model saved -> {MODEL_FILE}")
    print(f"AIC: {meta['aic']:.2f}, rows: {n_rows}")


def load_model():
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


def main() -> None:
    if TRAINING_SET_FILE.exists():
        training = pd.read_csv(TRAINING_SET_FILE)
    else:
        raw = pull_raw_data()
        training = build_training_set(raw["stats"], raw["snaps"], raw["schedules"])

    result, n_rows = train_model(training)
    save_model(result, n_rows)


if __name__ == "__main__":
    main()
