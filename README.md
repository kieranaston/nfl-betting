# NFL Reception Props

Automated system that predicts NFL WR/TE reception O/U props, publishes picks before each week's kickoffs, and tracks edge (CLV + calibration) against the closing line through the season.

## Stack

| Layer | Tool |
|-------|------|
| Data | [nflreadpy](https://nflreadpy.nflverse.com/) — player stats, snap counts, schedules, injuries |
| Odds | [The Odds API](https://the-odds-api.com/) — `player_receptions` market via event-odds endpoint |
| Model | statsmodels GLM, negative binomial family |
| Storage | Flat JSON/CSV committed to the repo |
| Automation | GitHub Actions scheduled workflows |
| Frontend | GitHub Pages — static HTML + Chart.js |

## Scope (v1)

- Receptions O/U only
- WR/TE averaging 5+ targets/game
- Minimum 3% edge vs implied probability to publish a pick

## Repo Layout

```
/data          nflreadpy pulls + training set
/model         training script + saved model
/predictions   weekly picks + closing-line snapshot
/results       graded outcomes, running CLV + Brier score
/site          static dashboard (index.html + Chart.js)
/scripts       Python pipeline
/.github/workflows
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your Odds API key
```

### Initial run (local)

```bash
python scripts/pull_data.py    # fetch nflverse data, build training set
python scripts/train.py        # fit negative binomial GLM
python scripts/predict.py        # generate picks (reads ODDS_API_KEY from .env)
python scripts/sync_site_data.py             # copy JSON into site/data/
```

## Automation

Two scheduled GitHub Actions jobs keep the system running and prevent workflow idle-disable (60-day rule):

| Job | Schedule | Actions |
|-----|----------|---------|
| **Weekly Train** | Tue 10:00 ET | Grade last week → pull data → retrain → update results |
| **Sunday Picks** | Sun 11:00 ET | Pull odds → generate picks → log closing-line snapshot |

### GitHub secrets

Add `ODDS_API_KEY` in **Settings → Secrets → Actions**. The key never touches client-side code.

### Local `.env`

```bash
cp .env.example .env
# edit .env and set ODDS_API_KEY=your_key
```

Scripts load `.env` automatically via `config.py`. You can still override with `export ODDS_API_KEY=...` in the shell.

### GitHub Pages

GitHub's "Deploy from a branch" dropdown only offers `/` or `/docs` — not `/site`. This project uses a **GitHub Actions** workflow instead (`.github/workflows/deploy_pages.yml`).

1. **Settings → Pages → Build and deployment → Source**: select **GitHub Actions** (not "Deploy from a branch")
2. Push the repo — the **Deploy GitHub Pages** workflow runs when `site/` changes
3. Or trigger manually: **Actions → Deploy GitHub Pages → Run workflow**

Site URL: `https://YOUR_USERNAME.github.io/nfl-betting/`

Before the first deploy, sync dashboard data locally and push:

```bash
python scripts/sync_site_data.py
git add site/data predictions/
git commit -m "Sync dashboard data"
git push
```

## Model

Negative binomial GLM predicting weekly receptions:

```
receptions ~ targets_l5 + receptions_l5 + snap_pct + position_te + home + season
```

For each prop line, the model computes P(over) and P(under) and compares against book-implied probabilities. Picks require ≥3% edge.

## Tracking

- **Brier score** — calibration of predicted probabilities vs outcomes
- **CLV** — closing line value (line movement in pick direction; meaningful when lines move between snapshot and kickoff)
- **Record** — win/loss on published picks

## License

Data from nflverse is CC-BY 4.0. This project code is MIT.
