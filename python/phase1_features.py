"""
Phase 1: Feature engineering pipeline.

Inputs (from data/):
  participant_data_{train,test}.csv
  wearable_data_{train,test}.csv
  diagnosis_records_{train,test}.csv

Outputs (to data/processed/):
  features_{train,test}.parquet  -- one row per ID, fully engineered
  diagnosis_buckets.csv          -- normalization map used (for transparency)

Design notes:
  - Wearable adoption is self-selected. We engineer engagement features
    as both clinical signals (HR/SpO2 means/std) AND adoption proxies
    (days_with_data, app_crash_rate, etc.) so Phase 3 can disentangle.
  - Diagnosis text spans 2018-2025 (post-baseline). We treat first-observed
    diagnosis date as a feature, but exclude post-baseline diagnosis
    counts from the *baseline* (M0) underwriter feature set.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd

DATA = Path("data")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Diagnosis text normalization
# ---------------------------------------------------------------------------

DIAG_BUCKETS = {
    "hypertension": [
        r"^hbp$", r"high blood pressure", r"hypertension", r"primary hypertension",
    ],
    "diabetes": [r"^dmii$", r"hyperglycemia", r"diabetes", r"^t2dm$", r"^t1dm$"],
    "sleep_apnea": [r"sleep apnea", r"^osa$", r"^csa$", r"^sdb$"],
    "cardiac_other": [r"afib", r"atrial fib", r"arrhyth", r"chf", r"heart fail"],
    "cancer": [r"cancer", r"carcinoma", r"tumor", r"oncolog", r"leuk", r"lymph"],
    "respiratory_infection": [
        r"^flu$", r"influenza", r"^covid", r"sinus infection", r"bronchitis",
        r"strep", r"tonsillitis", r"pneumonia",
    ],
    "ent_eye": [r"ear infection", r"conjucti", r"conjuncti"],
    "gi": [r"ibs", r"diarrhea", r"food poisoning", r"stomach ache"],
    "skin": [r"psoriasis", r"eczema", r"dermatitis"],
    "neuro_pain": [
        r"migraine", r"concussion", r"chonic fatigue", r"chronic fatigue",
        r"^fm$", r"fibro", r"arthritis",
    ],
    "dementia_neurodegen": [r"dementia", r"alzheimer", r"parkinson"],
}


def bucket_diagnosis(raw: str) -> str:
    if not isinstance(raw, str):
        return "unknown"
    s = raw.strip().lower()
    if s in {"", "n/a", "na", "none"}:
        return "unknown"
    for bucket, patterns in DIAG_BUCKETS.items():
        for pat in patterns:
            if re.search(pat, s):
                return bucket
    return "other"


# ---------------------------------------------------------------------------
# Wearable panel aggregation
# ---------------------------------------------------------------------------

def aggregate_wearable(wear: pd.DataFrame) -> pd.DataFrame:
    """One row per ID. Engagement-clinical features + adoption proxies."""
    wear = wear.copy()
    wear["timestamp"] = pd.to_datetime(wear["timestamp"], errors="coerce")

    # Indicator: did this row carry any clinical measurement?
    wear["has_measurement"] = wear[
        ["sleep_duration", "RHR_U", "SpO2_U", "MaxHR", "steps"]
    ].notna().any(axis=1)

    # Static (per-user) attributes — take first non-null
    static_cols = [
        "dominant_hand", "state", "region", "hospital_state",
        "household_size", "altitude", "os",
    ]
    static = (
        wear.sort_values("timestamp")
        .groupby("ID")[static_cols]
        .first()
        .reset_index()
    )

    # Adoption / engagement proxies
    adoption = (
        wear.groupby("ID")
        .agg(
            n_days_recorded=("timestamp", "nunique"),
            n_days_with_measurement=("has_measurement", "sum"),
            app_crash_rate=("app_crash", "mean"),
            mean_battery=("battery_level", "mean"),
            first_record_date=("timestamp", "min"),
            last_record_date=("timestamp", "max"),
        )
        .reset_index()
    )
    adoption["recording_span_days"] = (
        adoption["last_record_date"] - adoption["first_record_date"]
    ).dt.days.fillna(0)
    adoption["measurement_density"] = (
        adoption["n_days_with_measurement"]
        / adoption["recording_span_days"].replace(0, np.nan)
    ).fillna(0)

    # Clinical numeric aggregates (mean + std + IQR proxies)
    num_cols = [
        "n_wakeups", "sleep_duration", "sleep_light", "sleep_deep", "sleep_rem",
        "exercise_duration", "steps",
        "SpO2_U", "Sp02_L", "RHR_U", "RHR_L", "MaxHR", "skin_temp",
    ]
    g = wear.groupby("ID")[num_cols]
    means = g.mean().add_suffix("_mean")
    stds = g.std().add_suffix("_std")
    p10 = g.quantile(0.10).add_suffix("_p10")
    p90 = g.quantile(0.90).add_suffix("_p90")
    clinical = pd.concat([means, stds, p10, p90], axis=1).reset_index()

    # Event-rate features (fraction of days with the condition)
    event_cols = ["snoring", "afib_daily", "exercise"]
    events = wear.groupby("ID")[event_cols].mean().add_prefix("rate_").reset_index()

    # Derived: heart-rate range, SpO2 range, sleep efficiency
    clinical["RHR_range_mean"] = clinical["RHR_U_mean"] - clinical["RHR_L_mean"]
    clinical["SpO2_range_mean"] = clinical["SpO2_U_mean"] - clinical["Sp02_L_mean"]
    sleep_total = (
        clinical["sleep_light_mean"]
        + clinical["sleep_deep_mean"]
        + clinical["sleep_rem_mean"]
    )
    clinical["sleep_deep_frac"] = clinical["sleep_deep_mean"] / sleep_total.replace(0, np.nan)
    clinical["sleep_rem_frac"] = clinical["sleep_rem_mean"] / sleep_total.replace(0, np.nan)

    out = (
        adoption.merge(clinical, on="ID", how="outer")
        .merge(events, on="ID", how="outer")
        .merge(static, on="ID", how="outer")
    )
    out = out.drop(columns=["first_record_date", "last_record_date"])
    return out


# ---------------------------------------------------------------------------
# Diagnosis aggregation
# ---------------------------------------------------------------------------

def aggregate_diagnosis(diag: pd.DataFrame) -> pd.DataFrame:
    """One row per ID with diagnosis-bucket counts and provider variety."""
    diag = diag.copy()
    diag["date"] = pd.to_datetime(diag["date"], errors="coerce")
    diag["bucket"] = diag["diagnosis"].map(bucket_diagnosis)

    # Counts per bucket (wide)
    counts = (
        diag.pivot_table(
            index="ID", columns="bucket", values="diagnosis",
            aggfunc="count", fill_value=0,
        )
        .add_prefix("dx_count_")
        .reset_index()
    )

    # Variety + earliest/latest visit
    variety = (
        diag.groupby("ID")
        .agg(
            dx_total=("diagnosis", "count"),
            dx_distinct_buckets=("bucket", "nunique"),
            dx_distinct_providers=("HCP", "nunique"),
            dx_distinct_facilities=("HOSPITAL_CLINIC", "nunique"),
            dx_first_date=("date", "min"),
            dx_last_date=("date", "max"),
        )
        .reset_index()
    )
    variety["dx_active_span_days"] = (
        variety["dx_last_date"] - variety["dx_first_date"]
    ).dt.days
    variety = variety.drop(columns=["dx_first_date", "dx_last_date"])

    return counts.merge(variety, on="ID", how="outer")


# ---------------------------------------------------------------------------
# Master assembly
# ---------------------------------------------------------------------------

BASELINE_PARTICIPANT_COLS = [
    "ID", "age", "age_bin", "sex", "height", "weight", "bmi",
    "family_medical_diabetes", "family_medical_cancer",
    "family_medical_dementia", "family_medical_heart_disease",
    "smoker", "weekly_exercise", "alcohol_consumption_frequency",
    "diagnosed_diabetes", "diagnosed_lung_cancer", "diagnosed_other_cancer",
    "diagnosed_sleep_apnea", "diagnosed_afib", "diagnosed_hypertension",
    "diagnosed_dementia",
]
WEARABLE_FLAGS = ["wearable", "engagement_level"]


def build_split(split: str) -> pd.DataFrame:
    p = pd.read_csv(DATA / f"participant_data_{split}.csv")
    w = pd.read_csv(DATA / f"wearable_data_{split}.csv")
    d = pd.read_csv(DATA / f"diagnosis_records_{split}.csv")

    # Drop columns with massive missingness / type ambiguity for now
    p = p.drop(columns=[c for c in ["family_medical_diabetes_type",
                                     "diagnosed_diabetes_type"] if c in p.columns])

    base_cols = BASELINE_PARTICIPANT_COLS + WEARABLE_FLAGS
    if "outcome" in p.columns:
        base_cols = base_cols + ["outcome"]
    base = p[base_cols].copy()

    wear_agg = aggregate_wearable(w)
    diag_agg = aggregate_diagnosis(d)

    df = base.merge(wear_agg, on="ID", how="left").merge(diag_agg, on="ID", how="left")

    # Cohort label for downstream stratification / propensity work
    def cohort(row):
        if row["wearable"] == 0:
            return "non_wearable"
        if row["engagement_level"] <= 2:
            return "wearable_low_engagement"
        return "wearable_high_engagement"

    df["cohort"] = df.apply(cohort, axis=1)

    # Fill diagnosis count cols with 0 where missing (no diagnosis = absence)
    dx_cols = [c for c in df.columns if c.startswith("dx_count_") or c.startswith("dx_")]
    df[dx_cols] = df[dx_cols].fillna(0)

    return df


def main():
    train = build_split("train")
    test = build_split("test")

    # Save
    train.to_parquet(OUT / "features_train.parquet", index=False)
    test.to_parquet(OUT / "features_test.parquet", index=False)

    # Bucket transparency table
    diag_train = pd.read_csv(DATA / "diagnosis_records_train.csv")
    diag_train["bucket"] = diag_train["diagnosis"].map(bucket_diagnosis)
    bucket_map = (
        diag_train.dropna(subset=["diagnosis"])
        .groupby("bucket")["diagnosis"]
        .agg(lambda s: ", ".join(sorted({str(x) for x in s})[:15]))
        .reset_index()
        .rename(columns={"diagnosis": "example_terms"})
    )
    bucket_map.to_csv(OUT / "diagnosis_buckets.csv", index=False)

    print(f"train: {train.shape}, test: {test.shape}")
    print(f"\ncohort distribution (train):\n{train['cohort'].value_counts()}")
    print(f"\nmortality by cohort (train):\n"
          f"{train.groupby('cohort')['outcome'].agg(['count','mean']).round(4)}")
    print(f"\nmortality by engagement_level (train):\n"
          f"{train.groupby('engagement_level')['outcome'].agg(['count','mean']).round(4)}")
    print(f"\ncolumns ({train.shape[1]}): {list(train.columns)}")


if __name__ == "__main__":
    main()
