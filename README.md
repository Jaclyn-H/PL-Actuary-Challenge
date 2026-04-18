# Pacific Life AI Actuary Challenge 2026

End-to-end pipeline for evaluating whether wearable health data improves
10-year mortality prediction beyond traditional underwriting variables.

## TL;DR

- **Submitted model:** logistic regression on 21 traditional underwriting
  variables (M0 baseline). OOF log-loss = 0.2593, calibration slope = 0.96,
  Hosmer-Lemeshow p = 0.23.
- **Wearable + diagnosis features (M1) were tested and rejected.** Bootstrap
  95% CI for Δ log-loss (M1 − M0) is strictly positive: **+0.0014 to +0.0044**.
  M1 also fails Hosmer-Lemeshow (p < 0.001).
- **Selection bias is the explanation.** Adopters self-select on
  `weekly_exercise` (OR = 1.69), and a placebo test shows mortality varies
  by *operating system version* — proof of socioeconomic confounding the
  model cannot remove.
- **Recommendation:** do not use wearable data for pricing in its current
  form; use only for wellness program eligibility.

## Project Layout

```
data/                            raw CSVs (kept off-screen — large)
  └── processed/                 engineered features (parquet + csv)
python/                          modeling pipeline
  ├── phase1_features.py         feature engineering + cohort labels
  ├── phase2_models.py           M0 / M1, logreg + lgbm, 5-fold CV
  ├── phase4_validation.py       calibration, subgroups, stress tests
  └── phase5_deliverables.py     pickle artifact + scoring key
r/                                       selection-effect / causal analysis
  └── phase3_selection_effects.Rmd       IPTW, negative controls, dose-response
                                         (open in RStudio: knit or run by chunk)
output/
  ├── figures/                   plots (calibration, IPTW balance, etc.)
  ├── tables/                    CSV metric tables
  ├── models/                    joblib of fitted full-train models
  └── submission/                team_X_model_artifact.pkl + scoring key
deck/
  └── deck_outline.md            10-slide deck outline (judging-ready)
```

## How to Run

Python venv is already created. From the project root:

```bash
source .venv/bin/activate
python python/phase1_features.py             # feature engineering
python python/phase2_models.py               # M0 vs M1 modeling
R -e 'rmarkdown::render("r/phase3_selection_effects.Rmd")'  # propensity / IPTW
                                              # (or knit interactively in RStudio)
python python/phase4_validation.py           # calibration & stress tests
python -m python.phase5_deliverables         # pickle + scoring key
```

Total runtime ≈ 2–3 minutes on a laptop.

## Submission Artifacts

Before submitting, replace `TEAM_NUMBER = "X"` in
`python/phase5_deliverables.py` with your assigned number, re-run that
script, and rename the outputs in `output/submission/` accordingly.

Required submission files (per the problem statement):
1. `team_[n]_model_artifact.pkl`  ✅ generated
2. `team_[n]_scoring_key.csv`     ✅ generated
3. `team_[n]_code.zip`            — `zip -r` everything except `data/` and `.venv/`
4. `team_[n]_presentation.pdf`    — convert `deck/deck_outline.md` to slides
5. `team_[n]_video_presentation.mp4` — record per spec (≤ 7 min)

## Key Findings Cheat-Sheet (for live Q&A)

| Question judges may ask | Short answer |
|---|---|
| Why not LightGBM? | Failed Hosmer-Lemeshow (p < 0.001), calibration slope 0.62. Worse log-loss than logreg. |
| Why exclude diagnosis records? | Dated 2018-2025, post-baseline. Including them as predictors mixes model with outcome window. Tested in M1 — degraded performance anyway. |
| Doesn't IPTW say wearable is protective? | Crude RR = 0.70, IPTW RR = 0.68 — barely changes. But the negative-control test (OS predicts mortality) proves residual confounding remains. We cannot causally identify the wearable effect with the variables provided. |
| If you had to use wearable data, how? | Wellness-program eligibility flag. No pricing use until adoption broadens (currently 9.7%) and a negative-control test passes. |
| Strongest piece of evidence? | The OS placebo test. Mortality difference between iOS 18 and Android 13 users (4.9% vs 7.9%) is larger than the entire wearable vs non-wearable difference, and OS cannot causally affect death. |
```
