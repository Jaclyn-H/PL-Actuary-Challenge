"""
Phase 2: Build M0 (baseline underwriter) and M1 (augmented) models.

For each model we report (out-of-fold, 5-fold stratified):
  - log-loss        (the scored metric)
  - AUC
  - Brier score
  - calibration slope and intercept (logistic recalibration of OOF probs)

We compare two model families:
  - logistic regression (primary, interpretable, judged-friendly)
  - LightGBM           (sensitivity check, monotonic where applicable)

Output: output/tables/phase2_metrics.csv plus joblib of fitted full-train models.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import statsmodels.api as sm
from lightgbm import LGBMClassifier

DATA = Path("data/processed")
OUT_T = Path("output/tables"); OUT_T.mkdir(parents=True, exist_ok=True)
OUT_M = Path("output/models"); OUT_M.mkdir(parents=True, exist_ok=True)

RNG = 42

# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------

BASELINE_NUMERIC = [
    "age", "height", "weight", "bmi", "weekly_exercise",
    "alcohol_consumption_frequency",
]
BASELINE_BINARY = [
    "smoker",
    "diagnosed_diabetes", "diagnosed_lung_cancer", "diagnosed_other_cancer",
    "diagnosed_sleep_apnea", "diagnosed_afib", "diagnosed_hypertension",
    "diagnosed_dementia",
]
BASELINE_ORDINAL = [
    "family_medical_diabetes", "family_medical_cancer",
    "family_medical_dementia", "family_medical_heart_disease",
]
BASELINE_CATEGORICAL = ["sex", "age_bin"]

BASELINE_FEATURES = (
    BASELINE_NUMERIC + BASELINE_BINARY + BASELINE_ORDINAL + BASELINE_CATEGORICAL
)

# Augmented adds wearable engagement, wearable-derived clinical features,
# and diagnosis-derived features.
WEARABLE_FLAGS = ["wearable", "engagement_level"]
WEARABLE_ADOPTION = [
    "n_days_recorded", "n_days_with_measurement",
    "app_crash_rate", "mean_battery", "recording_span_days", "measurement_density",
]
WEARABLE_CLINICAL_BASE = [
    "n_wakeups", "sleep_duration", "sleep_light", "sleep_deep", "sleep_rem",
    "exercise_duration", "steps",
    "SpO2_U", "Sp02_L", "RHR_U", "RHR_L", "MaxHR", "skin_temp",
]
WEARABLE_CLINICAL = (
    [f"{c}_mean" for c in WEARABLE_CLINICAL_BASE]
    + [f"{c}_std" for c in WEARABLE_CLINICAL_BASE]
    + [f"{c}_p10" for c in WEARABLE_CLINICAL_BASE]
    + [f"{c}_p90" for c in WEARABLE_CLINICAL_BASE]
    + ["RHR_range_mean", "SpO2_range_mean", "sleep_deep_frac", "sleep_rem_frac",
       "rate_snoring", "rate_afib_daily", "rate_exercise"]
)

DIAGNOSIS_COUNTS = [
    "dx_count_cardiac_other", "dx_count_dementia_neurodegen", "dx_count_diabetes",
    "dx_count_ent_eye", "dx_count_gi", "dx_count_hypertension",
    "dx_count_neuro_pain", "dx_count_other", "dx_count_respiratory_infection",
    "dx_count_skin", "dx_count_sleep_apnea", "dx_count_unknown",
    "dx_total", "dx_distinct_buckets", "dx_distinct_providers",
    "dx_distinct_facilities", "dx_active_span_days",
]

AUGMENTED_FEATURES = (
    BASELINE_FEATURES + WEARABLE_FLAGS + WEARABLE_ADOPTION
    + WEARABLE_CLINICAL + DIAGNOSIS_COUNTS
)


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

def make_preprocessor(features):
    numeric = [f for f in features if f not in BASELINE_CATEGORICAL]
    categorical = [f for f in features if f in BASELINE_CATEGORICAL]
    transformers = []
    if numeric:
        transformers.append((
            "num",
            Pipeline([
                ("imp", SimpleImputer(strategy="median", add_indicator=True)),
                ("sc", StandardScaler()),
            ]),
            numeric,
        ))
    if categorical:
        transformers.append((
            "cat",
            Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore", drop="first")),
            ]),
            categorical,
        ))
    return ColumnTransformer(transformers, remainder="drop")


def make_logreg(features):
    return Pipeline([
        ("prep", make_preprocessor(features)),
        ("clf", LogisticRegression(max_iter=2000, C=1.0, solver="liblinear")),
    ])


def make_lgbm(features):
    # LightGBM tolerates raw NaNs; we still encode categoricals via the same
    # preprocessor so coefficient comparisons are fair.
    return Pipeline([
        ("prep", make_preprocessor(features)),
        ("clf", LGBMClassifier(
            n_estimators=500, learning_rate=0.03, num_leaves=31,
            min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=RNG, n_jobs=-1, verbose=-1,
        )),
    ])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def calibration_slope_intercept(y, p):
    """Logistic recalibration: logit(p_actual) = a + b * logit(p_pred)."""
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    logit_p = np.log(p / (1 - p))
    X = sm.add_constant(logit_p)
    res = sm.Logit(y, X).fit(disp=False)
    params = np.asarray(res.params)
    return float(params[0]), float(params[1])


def cv_evaluate(pipeline, X, y, n_splits=5, seed=RNG):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        pipeline.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = pipeline.predict_proba(X.iloc[va])[:, 1]
    metrics = {
        "log_loss": log_loss(y, oof),
        "auc": roc_auc_score(y, oof),
        "brier": brier_score_loss(y, oof),
    }
    a, b = calibration_slope_intercept(y, oof)
    metrics["calibration_intercept"] = a
    metrics["calibration_slope"] = b
    return metrics, oof


def main():
    train = pd.read_parquet(DATA / "features_train.parquet")
    y = train["outcome"].astype(int)

    results = []
    oof_store = {}

    for model_name, builder in [("logreg", make_logreg), ("lgbm", make_lgbm)]:
        for spec_name, features in [("M0_baseline", BASELINE_FEATURES),
                                    ("M1_augmented", AUGMENTED_FEATURES)]:
            X = train[features]
            pipe = builder(features)
            metrics, oof = cv_evaluate(pipe, X, y)
            metrics.update({"model": model_name, "spec": spec_name,
                            "n_features": len(features)})
            results.append(metrics)
            oof_store[f"{model_name}__{spec_name}"] = oof
            print(f"{model_name:7s} | {spec_name:12s} | "
                  f"logloss={metrics['log_loss']:.5f}  "
                  f"auc={metrics['auc']:.4f}  "
                  f"brier={metrics['brier']:.5f}  "
                  f"calib_slope={metrics['calibration_slope']:.3f}")

            # Also fit on full train and persist (used in Phase 5)
            pipe.fit(X, y)
            joblib.dump(pipe, OUT_M / f"{model_name}__{spec_name}.joblib")

    res_df = pd.DataFrame(results)[
        ["model", "spec", "n_features", "log_loss", "auc", "brier",
         "calibration_intercept", "calibration_slope"]
    ]
    res_df.to_csv(OUT_T / "phase2_metrics.csv", index=False)

    # Incremental value summary (M1 - M0)
    inc = []
    for model in ["logreg", "lgbm"]:
        m0 = res_df[(res_df.model == model) & (res_df.spec == "M0_baseline")].iloc[0]
        m1 = res_df[(res_df.model == model) & (res_df.spec == "M1_augmented")].iloc[0]
        inc.append({
            "model": model,
            "delta_log_loss": m1["log_loss"] - m0["log_loss"],
            "delta_auc": m1["auc"] - m0["auc"],
            "delta_brier": m1["brier"] - m0["brier"],
            "rel_log_loss_pct": 100 * (m1["log_loss"] - m0["log_loss"]) / m0["log_loss"],
        })
    inc_df = pd.DataFrame(inc)
    inc_df.to_csv(OUT_T / "phase2_incremental_value.csv", index=False)
    print("\n=== Incremental value (M1 - M0) ===")
    print(inc_df.round(5).to_string(index=False))

    # Save OOF for later use (calibration plots, propensity export, etc.)
    oof_df = pd.DataFrame(oof_store)
    oof_df["ID"] = train["ID"].values
    oof_df["outcome"] = y.values
    oof_df.to_parquet(OUT_T / "phase2_oof_predictions.parquet", index=False)


if __name__ == "__main__":
    main()
