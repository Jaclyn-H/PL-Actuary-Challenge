"""
Phase 4: Validation, calibration, subgroup performance, stress tests.

Inputs:
  output/tables/phase2_oof_predictions.parquet  (OOF probs from M0/M1, both families)
  data/processed/features_train.parquet         (full feature set + cohort)

Outputs:
  output/figures/phase4_calibration_*.png
  output/tables/phase4_subgroup_metrics.csv
  output/tables/phase4_stress_tests.csv
  output/tables/phase4_hosmer_lemeshow.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from scipy.stats import chi2

OUT_F = Path("output/figures"); OUT_F.mkdir(parents=True, exist_ok=True)
OUT_T = Path("output/tables")

train = pd.read_parquet("data/processed/features_train.parquet")
oof = pd.read_parquet(OUT_T / "phase2_oof_predictions.parquet")
df = train.merge(oof, on=["ID", "outcome"], how="inner")

PROB_COLS = [c for c in oof.columns if "__" in c]


# ---------------------------------------------------------------------------
# 1. Calibration plot (deciles) per model
# ---------------------------------------------------------------------------
def calibration_table(y, p, n_bins=10):
    df_ = pd.DataFrame({"y": y, "p": p})
    df_["bin"] = pd.qcut(df_["p"], n_bins, labels=False, duplicates="drop")
    g = df_.groupby("bin").agg(
        mean_pred=("p", "mean"),
        actual=("y", "mean"),
        n=("y", "size"),
    ).reset_index()
    return g


fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, family in zip(axes, ["logreg", "lgbm"]):
    for spec, color in [("M0_baseline", "#7f8fa6"),
                        ("M1_augmented", "#1e3799")]:
        col = f"{family}__{spec}"
        ct = calibration_table(df["outcome"].values, df[col].values, 10)
        ax.plot(ct["mean_pred"], ct["actual"], "-o",
                label=spec, color=color, lw=2, markersize=6)
    ax.plot([0, ct["mean_pred"].max() * 1.05],
            [0, ct["mean_pred"].max() * 1.05],
            "--", color="black", lw=1, alpha=0.5)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Actual mortality rate")
    ax.set_title(f"{family.upper()}: calibration by decile")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_F / "phase4_calibration.png", dpi=200)
plt.close()


# ---------------------------------------------------------------------------
# 2. Hosmer-Lemeshow goodness-of-fit
# ---------------------------------------------------------------------------
def hosmer_lemeshow(y, p, g=10):
    df_ = pd.DataFrame({"y": y, "p": p})
    df_["bin"] = pd.qcut(df_["p"], g, labels=False, duplicates="drop")
    grp = df_.groupby("bin").agg(o1=("y", "sum"),
                                  e1=("p", "sum"), n=("y", "size"))
    grp["o0"] = grp["n"] - grp["o1"]
    grp["e0"] = grp["n"] - grp["e1"]
    chi = (((grp["o1"] - grp["e1"]) ** 2 / grp["e1"]).sum()
           + ((grp["o0"] - grp["e0"]) ** 2 / grp["e0"]).sum())
    dof = max(g - 2, 1)
    p_value = 1 - chi2.cdf(chi, dof)
    return chi, dof, p_value


hl_rows = []
for col in PROB_COLS:
    chi, dof, pv = hosmer_lemeshow(df["outcome"].values, df[col].values)
    hl_rows.append({"model_spec": col, "HL_chi2": chi,
                    "df": dof, "p_value": pv})
hl_df = pd.DataFrame(hl_rows)
hl_df.to_csv(OUT_T / "phase4_hosmer_lemeshow.csv", index=False)
print("Hosmer-Lemeshow GOF (high p-value = good fit):")
print(hl_df.round(4).to_string(index=False))


# ---------------------------------------------------------------------------
# 3. Subgroup performance — by sex, age_bin, cohort
# ---------------------------------------------------------------------------
def subgroup_metrics(d, prob_col, group_col):
    rows = []
    for level, sub in d.groupby(group_col):
        if sub["outcome"].nunique() < 2 or len(sub) < 30:
            continue
        rows.append({
            "group_var": group_col,
            "level": str(level),
            "n": len(sub),
            "actual_rate": sub["outcome"].mean(),
            "pred_rate": sub[prob_col].mean(),
            "log_loss": log_loss(sub["outcome"], sub[prob_col],
                                 labels=[0, 1]),
            "auc": roc_auc_score(sub["outcome"], sub[prob_col]),
            "brier": brier_score_loss(sub["outcome"], sub[prob_col]),
        })
    return rows


sub_all = []
target_col = "logreg__M0_baseline"  # use the calibrated baseline as primary
for gc in ["sex", "age_bin", "cohort", "region"]:
    if gc in df.columns:
        rows = subgroup_metrics(df, target_col, gc)
        for r in rows:
            r["prob_col"] = target_col
        sub_all.extend(rows)
sub_df = pd.DataFrame(sub_all)[
    ["prob_col", "group_var", "level", "n",
     "actual_rate", "pred_rate", "log_loss", "auc", "brier"]
]
sub_df.to_csv(OUT_T / "phase4_subgroup_metrics.csv", index=False)
print("\nSubgroup metrics (M0 baseline, logreg):")
print(sub_df.round(4).to_string(index=False))


# ---------------------------------------------------------------------------
# 4. Stress tests
# ---------------------------------------------------------------------------
# Re-run M1 vs M0 incremental value under perturbations.
# We use the OOF predictions on subsets — equivalent to asking: does the
# advantage hold among these subjects?
def metric_pair(d, m0_col, m1_col):
    return {
        "n": len(d),
        "logloss_M0": log_loss(d["outcome"], d[m0_col], labels=[0, 1]),
        "logloss_M1": log_loss(d["outcome"], d[m1_col], labels=[0, 1]),
        "auc_M0": roc_auc_score(d["outcome"], d[m0_col]),
        "auc_M1": roc_auc_score(d["outcome"], d[m1_col]),
    }


stress_rows = []
m0c, m1c = "logreg__M0_baseline", "logreg__M1_augmented"

stress_rows.append({"scenario": "all_data",
                    **metric_pair(df, m0c, m1c)})

stress_rows.append({"scenario": "drop_top_engagement_tier_5",
                    **metric_pair(df.query("engagement_level != 5"), m0c, m1c)})

stress_rows.append({"scenario": "wearable_users_only",
                    **metric_pair(df.query("wearable == 1"), m0c, m1c)})

stress_rows.append({"scenario": "non_wearable_only",
                    **metric_pair(df.query("wearable == 0"), m0c, m1c)})

stress_rows.append({"scenario": "age_45_64",
                    **metric_pair(df.query("age_bin == '45-64'"), m0c, m1c)})

stress_rows.append({"scenario": "age_65_plus",
                    **metric_pair(df.query("age_bin == '65+'"), m0c, m1c)})

# Bootstrap stability of incremental value
rng = np.random.default_rng(42)
boot = []
for _ in range(500):
    idx = rng.integers(0, len(df), size=len(df))
    sub = df.iloc[idx]
    boot.append(log_loss(sub["outcome"], sub[m1c], labels=[0, 1])
                - log_loss(sub["outcome"], sub[m0c], labels=[0, 1]))
boot = np.array(boot)
stress_rows.append({"scenario": "bootstrap_M1-M0_logloss_mean",
                    "n": 500,
                    "logloss_M0": np.nan, "logloss_M1": np.nan,
                    "auc_M0": np.nan, "auc_M1": np.nan,
                    "delta_logloss_mean": float(boot.mean()),
                    "delta_logloss_p2.5": float(np.quantile(boot, 0.025)),
                    "delta_logloss_p97.5": float(np.quantile(boot, 0.975))})

stress_df = pd.DataFrame(stress_rows)
for s in ["logloss", "auc"]:
    stress_df[f"delta_{s}"] = stress_df[f"{s}_M1"] - stress_df[f"{s}_M0"]
cols = ["scenario", "n", "logloss_M0", "logloss_M1", "delta_logloss",
        "auc_M0", "auc_M1", "delta_auc",
        "delta_logloss_mean", "delta_logloss_p2.5", "delta_logloss_p97.5"]
stress_df = stress_df.reindex(columns=[c for c in cols if c in stress_df.columns])
stress_df.to_csv(OUT_T / "phase4_stress_tests.csv", index=False)
print("\nStress tests (delta = M1 - M0; positive delta_logloss is bad):")
print(stress_df.round(5).to_string(index=False))


# ---------------------------------------------------------------------------
# 5. Calibration intercept / slope summary table per subgroup
# ---------------------------------------------------------------------------
import statsmodels.api as sm


def calib_si(y, p):
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    X = sm.add_constant(np.log(p / (1 - p)))
    res = sm.Logit(y, X).fit(disp=False)
    pp = np.asarray(res.params)
    return float(pp[0]), float(pp[1])


cs_rows = []
for col in PROB_COLS:
    a, b = calib_si(df["outcome"].values, df[col].values)
    cs_rows.append({"model_spec": col, "intercept": a, "slope": b})
pd.DataFrame(cs_rows).to_csv(OUT_T / "phase4_calibration_si.csv", index=False)
print("\nCalibration intercept (≈0) and slope (≈1) per spec:")
print(pd.DataFrame(cs_rows).round(4).to_string(index=False))

print("\nPhase 4 complete.")
