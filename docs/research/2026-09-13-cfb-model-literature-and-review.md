# CFB prediction model: literature ingest and review of the matchup model

**Date:** 2026-09-13
**Scope:** three sources the user supplied, the CFBD API surface, and the model that
`sports-etl` fits and this server serves (`workflows/cfb/matchup.py`).
**Location note:** this repo had no research-notes convention (only
`docs/superpowers/{specs,plans}`), so notes of this kind live in `docs/research/`.

---

## 1. Sources

### 1.1 South & Egros (2020), *Forecasting college football game outcomes using modern modeling techniques*

- Journal of Sports Analytics 6(1):25–33, DOI 10.3233/JSA-190314, CC BY-NC 4.0.
  Authors at Southern Methodist University. (Crossref record:
  https://api.crossref.org/works/10.3233/JSA-190314)
- **Full text was not retrievable.** journals.sagepub.com, content.iospress.com,
  researchgate.net and scholar.archive.org all sit behind a bot-verification wall,
  and web.archive.org is blocked from this environment. Everything below comes from
  the abstract (Crossref, Semantic Scholar) plus one indexed excerpt surfaced by web
  search. Treat the excerpt as medium confidence.
- **Design:** train on 2011–2014 seasons, test on 2015. Response is game outcome
  (win/loss); RMSE is also reported, so a margin-type response was modelled too.
- **Models compared:** ridge, lasso, elastic net, neural network, random forest,
  k-nearest neighbours, stochastic gradient boosting, and a Bayesian regression with
  team-specific variances.
- **Result:** the top four (lasso, Bayesian regression with team-specific variances,
  stochastic gradient boosting, random forests) each picked the winner **over 70 %**
  of the time on 2015; lasso was the most accurate overall.
- **Excerpt (search-indexed, unverified against the PDF):** the lasso had the least
  RMSE variability across 50 repetitions but was top-ranked in only 5 of 50, while
  random forests and XGBoost were first 16 and 15 times respectively. Reading: a
  regularised linear model is the safest single choice; tree ensembles win more
  individual splits but are noisier.
- Sources: https://journals.sagepub.com/doi/10.3233/JSA-190314 ,
  https://content.iospress.com/articles/journal-of-sports-analytics/jsa190314 ,
  https://www.researchgate.net/publication/338728435 (all same paper).

### 1.2 Rad Sports Analytics, *College football modeling tips*

https://radsportsanalytics.com/blog/college-football-modeling-tips/ (fetched in full)

1. Start with clean, structured data (games, drives, plays, advanced stats, team metadata).
2. Start training at week 5; weeks 0–4 are too thin for reliable opponent strength.
3. **Opponent-adjust everything** ("raw stats lie"): adjusted EPA/play, adjusted
   success rate, adjusted line yards.
4. **Model score margin first, win probability second** — derive WP, cover
   probability, totals and confidence from the margin model.
5. Prefer high-signal features: opponent-adjusted efficiency, talent composites,
   run/pass ratios, havoc, explosive-play rate.
6. Include a talent composite (247-style) as a prior, strongest early in the season.
7. Proper cross-validation: k-fold shuffled by week or game id; guard against
   leakage from team-level stats.
8. Build linear/logistic baselines before XGBoost, random forests, tabular NNs.
9. **Visualise errors**, not just MAE/RMSE: predicted vs actual margin, residuals by
   team, over/under patterns, to catch systematic bias (team types, garbage time).
10. Use prebuilt tooling so effort goes to modelling.

### 1.3 CollegeFootballData API surface

https://api.collegefootballdata.com/getting-started (the getting-started page is thin;
the endpoint inventory below is from the OpenAPI document at
https://api.collegefootballdata.com/api-docs.json). Groups relevant to prediction:

| Group | What it offers | Do we ingest it? |
|---|---|---|
| Games | schedules, scores, pregame/postgame Elo, neutral site, classification | yes (`cfb_games`) |
| Team stats, as-of cumulative | `get_team_stats(start_week, end_week)` | yes (`cfb_team_stats_asof`) |
| Advanced season / game stats | PPA, success rate, explosiveness, havoc, line yards, field position | partly: PPA + success rate as-of; per-game PPA only in `cfb_exp` |
| Plays | play-by-play with PPA | only in `cfb_exp` (feeds our own CORE) |
| Ratings | SP+, SRS, Elo, FPI, CORE (end-of-season only) | no |
| Betting lines | spreads, totals, moneylines, several books | **no** |
| Talent / recruiting / transfer portal | composite talent, position-group ratings, portal entries | only in `cfb_exp` |
| Returning production | percent PPA returning | yes (`cfb_returning`) |
| Venues | capacity, lat/long, elevation, grass/turf, dome | no |
| Win probability | CFBD's own pregame WP | no |
| Weather | per-game weather (paid tier) | no |

---

## 2. What our model does today

Fitting lives in `sports-etl` (`model/cfb_model.py`, `transform/cfb_profile.py`,
`pipeline_cfb_model.py`); this server only applies stored coefficients
(`workflows/cfb/matchup.py:predict`).

- **Features (4):** `elo_diff` (CFBD pregame Elo), `adj_diff` (shrunk ridge-Massey
  points rating, FCS pooled, prior = last season's final rating regressed by
  returning production), `eff_diff` (raw garbage-time-excluded net PPA, shrunk), and
  a constant `home_field`.
- **Heads (3):** logistic win, ridge margin, ridge total — same four features each.
- **Validation:** walk-forward by season 2016→2026, pooled log loss gated against
  the Elo baseline. Shrinkage `k` chosen on held-out log loss (k=2, grid edge).
- **Leakage discipline:** week-N game joins the profile through week N-1; postseason
  joins the final regular week. Good.
- **Stored record (fit 2026-09-11):** accuracy 0.7205, Brier 0.1837, log loss 0.5433
  vs Elo 0.7035 / 0.1927 / 0.5682. Weeks 1–4 slice: 0.7559 / 0.5044 vs Elo 0.7441 / 0.5212.

Experiments already run and rejected in `sports-etl` history (numbers in the commit
messages): margin cap (worse), FCS down-weighting (worse), opponent-adjusted PPA
(r = 0.95 with `adj_margin`, no gain), our own as-of CORE ratings (genuinely new
information, r ≈ 0.5 with `adj_margin`, still no gain), offence/defence unit
split (not significant). Talent prior **helped** (+0.0012 pooled, +0.0035 early
log loss, both significant) but was only measured in the `cfb_exp` namespace.

## 3. Measurements made today (walk-forward, 2016–2026, n = 7,660)

| Head / variant | MAE | RMSE | Note |
|---|---|---|---|
| Margin head, current features | 13.09 | 16.52 | naive "home by 2.5" is 16.37 MAE |
| **Total head, current features** | **14.16** | 17.84 | **season-mean constant: 14.15 / 17.82** |
| Total head + (home_off + away_off), (home_def + away_def) | 13.51 | 17.01 | |
| … + raw PPA sums | 13.48 | 16.97 | |

| Win-probability variant | log loss | favourite disagreements |
|---|---|---|
| Logistic head (current) | 0.5431 | — |
| Φ(margin_head / 16.5) derived from the margin head | 0.5421 | 90 of 7,660 |

Feature ablation, win head, pooled log loss: Elo only 0.5493; Massey only 0.5450;
Elo+Massey 0.5427; all three 0.5431 (raw PPA adds accuracy, not log loss).

Home margin by season (FBS v FBS, non-neutral): ~4.1–4.6 for 2015–2023, then 5.4
(2024), 5.2 (2025). 2020 has 534 games and mostly empty stadiums.

Calibration (stored bins): the 0–0.1 bin predicts 0.062 and realises 0.101; the
0.3–0.4 bin predicts 0.351 and realises 0.380. Big home underdogs win more often than
the model says.

## 4. Scorecard against the sources

| Recommendation | Status |
|---|---|
| Opponent adjustment | Done for points (Massey); tried for efficiency, redundant |
| Margin first, WP derived | Not done: WP is a separate logistic head; the two disagree on 90 favourites |
| Talent composite as an early-season prior | Built and validated, **not in production** (tables empty in `cfb`, profile pipeline never passes `talent=`) |
| Linear baseline before ensembles | Done; ensembles never tried |
| Regularised model selects features (lasso) | Win head is effectively unregularised (C=1e6); fine at 4 features, matters if more are added |
| Walk-forward / leak-free CV | Done, by season |
| Residual visualisation | Only calibration bins; no residuals by team, conference, week, spread size, or over/under |
| High-signal features (explosiveness, havoc, line yards, run/pass) | Not ingested |
| Totals need a pace/strength-sum signal | Missing: total head uses only differences |

## 5. Opportunities, ranked by expected value per effort

1. **Fix the total head.** It cannot beat a constant because every feature is a
   difference. Add sums (`adj_off_h + adj_off_a`, `adj_def_h + adj_def_a`, and a
   plays-per-game pace term from the as-of stats). Measured −0.65 MAE today with
   just the two sums. This directly improves the projected scores on the matchup
   and slate pages.
2. **Promote the talent prior to production.** Run `pipeline_cfb_talent.py` into the
   `cfb` namespace, pass `talent=` in `pipeline_cfb_profile.py`, add it to
   `pipeline_cfb_refresh.py`, refit. The gain is already measured and significant,
   largest in weeks 1–4.
3. **Ingest betting lines as a benchmark.** The closing spread is the strongest
   public predictor; without it we cannot say whether 72 % / 13.1 MAE is good or
   merely Elo-plus. Cheap to fetch (`/lines`), and it enables a "model vs market"
   edge on the page. The 2026-09-09 spec listed lines as a non-goal, so this is a
   scope decision, not a code one.
4. **Derive win probability from the margin head** (Rad tip 4). Slightly better log
   loss, and it removes the projected-score / win-probability inconsistency the UI
   can currently show.
5. **Residual diagnostics** (Rad tip 9): residual by team, conference, week, spread
   bucket, and neutral/non-neutral. The low-probability calibration gap and the
   2024–25 home-margin jump are the first two things to look at; a time-varying or
   per-team home-field term, rest days, and venue distance (venues endpoint has
   lat/long) are the cheap follow-ups.
6. **Widen the shrinkage grid below k=2** — config notes the grid is monotonic to
   its edge.
7. **More features, chosen by a lasso, then a GBM**, in that order (paper + Rad
   tip 8). Explosiveness, havoc, line yards and success rate are one endpoint away
   (`/stats/season/advanced`, already fetched as-of but only PPA and success rate
   are kept). Given three rejected feature experiments, expect small gains; gate
   every step on the walk-forward log loss as today.
8. **Store the weeks 1–4 slice in `cfb_model`** so the page can show early-season
   accuracy alongside pooled.
