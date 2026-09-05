# NFL Reception Props

Automated system that predicts NFL WR/TE reception O/U props, refreshes picks as markets open through the week, and tracks model accuracy (MAE, bias, Brier, log loss) on the full prop slate.

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
- Minimum model confidence **55%** on a side vs the line (board filter; prices not used for ranking)
- Dashboard shows picks only for games kicking off within **48 hours**
- Week ledger merges refreshes (upcoming events overwrite; earlier games kept)

## Repo Layout

```
/data          nflreadpy pulls + training set
/model         training script + saved model
/predictions   weekly picks ledger + odds snapshot
/results       graded accuracy metrics
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

| Job | Schedule | Actions |
|-----|----------|---------|
| **Picks Refresh** | Wed 6:00 PM EST | Pull data → grade previous week → retrain → odds → merge week ledger → sync site |
| **Picks Refresh** | Sat 7:00 PM EST | Odds → merge week ledger (overwrite upcoming; keep finished games) → sync site |

Wednesday is the full cycle (pull last week’s games → evaluate → retrain → predict Thu/Fri slate). Saturday only refreshes predictions for the rest of the week. Manual dispatch defaults to the full Wednesday cycle; uncheck **full_cycle** for a refresh-only run.

Grading uses the week's **odds snapshot** (every lined prop the model scored), not the board shortlist.

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
receptions ~ targets_l5 + receptions_l5 + snap_pct + position_te + home
           + team_spread + total_line + opp_pass_funnel_rank + wind + outdoor
           + season_idx
```

`season_idx` is a continuous year index (`season - 2022`), so 2026 predicts as `4` with no unseen-category crash and no forced 2022 baseline.

For each prop line, the model computes P(over)/P(under) from μ. The board lists the side the model prefers when that probability is **≥55%**, ranked by model confidence (not by price/+EV). Book prices/links are shown for convenience only.

## Tracking

All metrics are on the **full snapshot slate** (every prop with a model μ and line), not betting results:

- **MAE / bias** — accuracy of predicted receptions (μ) vs actual; bias = mean(μ − actual)
- **Brier / log loss** — quality of P(over) vs whether receptions cleared the line (pushes skipped; prices unused)

## License

Data from nflverse is CC-BY 4.0. This project code is MIT.
