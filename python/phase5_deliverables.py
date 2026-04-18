"""
Phase 5: Build the submission artifact and scoring key.

Selected model: Logistic Regression on M0_baseline features.

Rationale (defensible to judges):
  - Best out-of-fold log-loss (0.2593 vs. 0.2621 for M1; 0.2726 for LGBM).
  - Only model that passes Hosmer-Lemeshow (p = 0.23 vs. 0.0003 for M1).
  - Calibration slope 0.96 (near-ideal) vs. 0.83 for M1, 0.62 for LGBM.
  - Bootstrap 95% CI for M1 - M0 log-loss is strictly positive: adding
    wearable features RELIABLY degrades performance.
  - Simpler, fully interpretable: every coefficient defensible to underwriters.

Outputs:
  output/submission/team_7_model_artifact.pkl
  output/submission/team_7_scoring_key.csv
  output/submission/feature_coefficients.csv  (transparency table)

Replace TEAM_NUMBER below before final submission.
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import joblib

TEAM_NUMBER = "7"  # <<< replace with your assigned team number before submission
OUT = Path("output/submission"); OUT.mkdir(parents=True, exist_ok=True)

# Load fitted full-train M0 logreg from Phase 2
pipe = joblib.load("output/models/logreg__M0_baseline.joblib")

# Re-load the canonical feature list to guarantee column order
from python.phase2_models import BASELINE_FEATURES  # noqa: E402

# -------- Build scoring key on test set --------
test = pd.read_parquet("data/processed/features_test.parquet")
X_test = test[BASELINE_FEATURES]
proba = pipe.predict_proba(X_test)[:, 1]

scoring = pd.DataFrame({
    "ID": test["ID"].astype(int),
    "predicted_probability": proba,
})
scoring_path = OUT / f"team_{TEAM_NUMBER}_scoring_key.csv"
scoring.to_csv(scoring_path, index=False)

# -------- Persist model artifact (pickle, per submission spec) --------
artifact_path = OUT / f"team_{TEAM_NUMBER}_model_artifact.pkl"
with open(artifact_path, "wb") as f:
    pickle.dump(
        {
            "model": pipe,
            "feature_list": BASELINE_FEATURES,
            "predict": (
                "model.predict_proba(df[feature_list])[:, 1] returns "
                "P(outcome=1) over the 10-year study window."
            ),
            "training_metrics_oof": {
                "log_loss": 0.25932,
                "auc": 0.6890,
                "brier": 0.06954,
                "hosmer_lemeshow_p": 0.2274,
                "calibration_slope": 0.9575,
                "calibration_intercept": -0.0997,
            },
            "spec": "M0_baseline (traditional underwriting only). "
                    "Wearable & diagnosis features were tested (M1) and "
                    "rejected: they degrade log-loss, calibration, and "
                    "incremental value (bootstrap 95% CI strictly positive).",
        },
        f,
    )

# -------- Coefficient table --------
prep = pipe.named_steps["prep"]
clf = pipe.named_steps["clf"]
feature_names = prep.get_feature_names_out()
coefs = pd.DataFrame({
    "feature": feature_names,
    "coef_log_odds": clf.coef_[0],
    "odds_ratio": np.exp(clf.coef_[0]),
}).sort_values("coef_log_odds", key=lambda s: s.abs(), ascending=False)
coefs.to_csv(OUT / "feature_coefficients.csv", index=False)

# -------- Verify reproducibility: load artifact and re-score --------
with open(artifact_path, "rb") as f:
    art = pickle.load(f)
verify_proba = art["model"].predict_proba(test[art["feature_list"]])[:, 1]
assert np.allclose(verify_proba, proba), "Pickle reproducibility check failed."

print(f"Model artifact: {artifact_path}")
print(f"Scoring key:    {scoring_path}")
print(f"Coefficients:   {OUT / 'feature_coefficients.csv'}")
print(f"\nScoring key head:\n{scoring.head()}")
print(f"\nPredicted probability summary:\n{scoring['predicted_probability'].describe().round(4)}")
print(f"\nTop 10 coefficients (by |log-odds|):")
print(coefs.head(10).round(4).to_string(index=False))
print("\nReproducibility check: PASS.")
