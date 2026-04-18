# Pacific Life AI Actuary Challenge 2026 — Slide Deck Outline

**Recommended:** 9 slides + appendix. Built to map directly to the judging rubric in the problem statement.

Figures referenced live in `output/figures/`; tables in `output/tables/` and `output/submission/`.

---

## Slide 1 — Title & Question

**Title:** *Wearable Data in Mortality Underwriting: Real Signal or Selection Mirage?*

**Subtitle:** Pacific Life AI Actuary Challenge 2026 — Team [N]

**One-line thesis (place at bottom of slide):**
> "Wearable adoption appears to halve 10-year mortality — but the entire effect is captured by traditional underwriting variables. We recommend **not** using wearable data for pricing."

---

## Slide 2 — Actuarial Question & Hypothesis

**The question (verbatim from the brief):**
> Does wearable access and engagement provide meaningful, reliable incremental value for mortality risk assessment beyond traditional underwriting variables?

**Our null hypothesis (H₀):** Wearable signals add no incremental predictive value once age, sex, BMI, smoker, family history and diagnosed conditions are controlled.

**Our alternative (H₁):** Wearable engagement adds calibrated, stable incremental signal that justifies inclusion in pricing.

**Decision frame:** A wearable feature is worth using only if it (a) improves out-of-fold log-loss, (b) preserves calibration, (c) survives selection-effect adjustment, and (d) holds across subgroups. We test all four.

---

## Slide 3 — Data & Cohort Design

| Source | Rows × Cols | Period |
|---|---|---|
| Participant profile | 20,000 × 26 | baseline |
| Wearable daily panel | 507,864 × 33 | 2016–2017 |
| Diagnosis records | 11,537 × 5 | 2018–2025 |

**Cohort split (train):**
- Non-wearable: **18,057** (90.3%)
- Wearable, low engagement (1–2): **841** (4.2%)
- Wearable, high engagement (3–5): **1,102** (5.5%)

**Crude mortality rates:**

| Cohort | Mortality |
|---|---|
| Non-wearable | **8.34%** |
| Wearable users (any) | **5.82%** |
| Engagement tier 4 (highest non-extreme) | 3.83% |
| Engagement tier 5 (most engaged) | **7.11%** ← reversal |

**Hook:** Tier 5's reversal is our first hint that engagement is not a clean signal.

> Figure: `output/figures/phase3_dose_response.png` — raw + age-stratified dose-response.

---

## Slide 4 — Model Construction (Two-Model Discipline)

| | M0 — Baseline | M1 — Augmented |
|---|---|---|
| Features | 21 traditional underwriting | 21 + wearable + diagnosis = 99 |
| Family | Logistic regression (primary) + LightGBM (sensitivity) | Same |
| CV | 5-fold stratified, fixed seed | Same folds |
| Reported | log-loss, AUC, Brier, calibration slope/intercept, HL p-value | Same |

**Why logistic regression as primary?** Calibrated, fully interpretable coefficient table, trivially auditable by an underwriter. The brief explicitly says interpretable approaches are encouraged.

> Detailed metric table: `output/tables/phase2_metrics.csv`.

---

## Slide 5 — Headline Result: M1 Does Not Beat M0

**Out-of-fold metrics (5-fold CV):**

| Model | Spec | Log-loss | AUC | Calibration slope | HL p-value |
|---|---|---|---|---|---|
| logreg | **M0 baseline** | **0.25932** | **0.6890** | **0.96** | **0.227** |
| logreg | M1 augmented | 0.26213 | 0.6815 | 0.83 | **0.0003** |
| lgbm | M0 baseline | 0.27263 | 0.6562 | 0.63 | <0.001 |
| lgbm | M1 augmented | 0.27287 | 0.6576 | 0.61 | <0.001 |

**Bootstrap (500 resamples) of Δlog-loss (M1 − M0):** mean = +0.00285, **95% CI [+0.0014, +0.0044]** — strictly positive.

**Translation:** Adding wearable & diagnosis features *reliably* makes the model worse and breaks calibration. Best model on every metric is the simple, baseline logistic regression.

---

## Slide 6 — Why? Selection Effects + Confounding

**Propensity model (P[wearable=1 | baseline]):**

| Predictor | Odds Ratio | p |
|---|---|---|
| Weekly exercise (per day/week) | **1.69** | <1e-141 |
| Alcohol frequency (per tier) | 0.78 | <1e-33 |
| BMI | 0.99 | 0.011 |

→ Exercisers, low-drinkers, leaner people are dramatically over-represented among wearable adopters.

**Negative-control (placebo) test — variables that shouldn't predict mortality:**

| OS version | Mortality |
|---|---|
| Android 13 | 7.87% |
| Android 14 | 6.62% |
| iOS 17 | 5.66% |
| **iOS 18** | **4.88%** |

**OS predicts mortality.** Newest/most-expensive devices = lowest mortality. This is a smoking gun for **socioeconomic confounding** that wearable features will inherit.

> Figure: `output/figures/phase3_pscore_overlap.png` (propensity overlap)
> Figure: `output/figures/phase3_balance_loveplot.png` (covariate balance pre/post IPTW)

---

## Slide 7 — Validation, Calibration, Stress Tests

**Calibration deciles** — only logreg M0 lies on the diagonal.
> Figure: `output/figures/phase4_calibration.png`

**Stress test summary** (Δlog-loss = M1 − M0; positive is bad):

| Scenario | Δ log-loss | Δ AUC |
|---|---|---|
| All data | +0.0028 | −0.008 |
| Drop highest-engagement tier | +0.0026 | −0.008 |
| **Wearable users only** | **+0.0216** | **−0.055** |
| Non-wearable only | +0.0008 | −0.004 |
| **Age 65+** | **+0.0301** | −0.010 |

**Headline:** M1 fails worst on the very cohorts wearable data is supposed to help — wearable users themselves and the oldest age band.

> Detailed table: `output/tables/phase4_stress_tests.csv`.

---

## Slide 8 — Recommendation

> **Do not incorporate wearable data into mortality pricing in its current form.**

**Three findings together justify this:**
1. **No incremental predictive value:** bootstrap CI of log-loss difference is strictly positive (M1 worse).
2. **No causal protective effect cleanly identifiable:** IPTW-adjusted relative risk = 0.68 — but negative-control test (OS predicts mortality) proves residual confounding cannot be removed with available variables.
3. **Calibration breaks:** adding wearable features fails Hosmer-Lemeshow (p < 0.001) — unacceptable for a regulated underwriting model.

**Conditional path forward (governance-approved use case):**
- ✅ Use wearable engagement as a **discount eligibility / wellness program criterion**, never as a price modifier.
- ✅ Re-evaluate when adoption broadens beyond the current 9.7% (selection bias should attenuate as the population diversifies).
- ❌ Do not segment, surcharge, or offer preferred classes based on any wearable-derived variable until the negative-control test passes.

**Deployed model:** `team_[N]_model_artifact.pkl` is the calibrated M0 logistic regression — log-loss 0.2593 OOF, calibration slope 0.96.

---

## Slide 9 — Limitations & Risks

- **Selection bias is fundamental, not fixable:** only 9.7% of the cohort uses wearables; adopters differ on every observable. We cannot assume signals generalize.
- **Diagnosis records are post-baseline (2018–2025).** We deliberately excluded them from M0 to avoid look-ahead leakage.
- **Subgroup AUC degrades for age 65+ (0.59).** Any product targeting this band needs a dedicated model.
- **No claim cause-of-death data:** we cannot distinguish whether wearable signals predict cardiac vs. cancer mortality differently.
- **Survival framing not used:** the outcome is binary 10-year survival, not time-to-event. A Cox proportional-hazards extension is a reasonable follow-up if censoring data becomes available.

---

## Slide 10 — AI Use Disclosure (REQUIRED)

**Where AI helped (the "starting spark"):**
- Drafting the analytical pipeline structure (Phase 1 → Phase 5).
- Brainstorming feature-engineering candidates from the wearable panel.
- Suggesting selection-effect tests (IPTW, negative controls).
- First-pass code scaffolding for sklearn pipelines and R `WeightIt` calls.

**Where human actuarial judgment took over:**
- **Choice of M0 over M1 for submission** — based on calibration tests, not raw AUC.
- **Decision to use logistic regression over LightGBM** — explicitly rejected the higher-flexibility model because it failed Hosmer-Lemeshow.
- **Negative-control variable selection** — chose OS and dominant-hand because these are *causally implausible* for mortality, ensuring the falsification test was meaningful.
- **Final recommendation language** — softened from "do not use" to "do not use for pricing; use for wellness eligibility" after considering business reality.

**Validation steps:**
- Every AI-generated metric was re-run end-to-end from raw CSVs.
- Coefficient signs were checked against actuarial expectation (smoker positive, exercise negative, etc. — all consistent).
- The model artifact was re-loaded from pickle and re-scored to confirm reproducibility.

> AI is a supplement, not a substitute. We own every conclusion in this deck.

---

## Appendix slides (technical detail)

**A1 — Feature engineering inventory** (all 99 augmented features grouped: baseline, wearable adoption, wearable clinical means, wearable clinical std/percentiles, diagnosis bucket counts, diagnosis variety).

**A2 — Diagnosis text normalization map** — `output/tables/../diagnosis_buckets.csv`.

**A3 — Top 15 logreg coefficients** — `output/submission/feature_coefficients.csv`.

**A4 — Subgroup metrics table** — `output/tables/phase4_subgroup_metrics.csv`.

**A5 — Propensity model coefficients** — `output/tables/phase3_propensity_coefs.csv`.

**A6 — Reproducibility:** every figure & table is regenerated by running
```
python python/phase1_features.py
python python/phase2_models.py
Rscript r/phase3_selection_effects.R
python python/phase4_validation.py
python -m python.phase5_deliverables
```
in that order, against the raw CSVs in `data/`.
