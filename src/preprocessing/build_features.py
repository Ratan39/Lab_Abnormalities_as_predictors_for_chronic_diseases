from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd


# -----------------------------------------------------------
# Lab Mapping Dictionary (same idea as in build_patient_table.py)
# -----------------------------------------------------------

LAB_MAPPING: Dict[str, str] = {
    # Glucose / diabetes
    "Glucose [Mass/volume] in Blood": "glucose_latest",
    "Glucose": "glucose_latest",
    "Hemoglobin A1c/Hemoglobin.total in Blood": "hba1c_latest",
    "Hemoglobin A1c": "hba1c_latest",

    # Creatinine / kidney
    "Creatinine": "creatinine_latest",
    "Creatinine [Mass/volume] in Serum or Plasma": "creatinine_latest",
    "Creatinine [Mass/volume] in Blood": "creatinine_latest",

    # eGFR
    "Glomerular filtration rate/1.73 sq M.predicted": "egfr_latest",
    "Glomerular filtration rate/1.73 sq M.predicted [Volume Rate/Area] in Serum or Plasma by Creatinine-based formula (MDRD)": "egfr_latest",

    # BUN / Urea
    "Urea Nitrogen": "bun_latest",
    "Urea nitrogen [Mass/volume] in Serum or Plasma": "bun_latest",
    "Urea nitrogen [Mass/volume] in Blood": "bun_latest",

    # Lipids
    "Cholesterol in HDL [Mass/volume] in Serum or Plasma": "hdl_latest",
    "Low Density Lipoprotein Cholesterol": "ldl_latest",
    "Triglycerides": "triglycerides_latest",
    "Cholesterol [Mass/volume] in Serum or Plasma": "cholesterol_total_latest",

    # Anemia-related
    "Hemoglobin [Mass/volume] in Blood": "hemoglobin_latest",
    "Hematocrit [Volume Fraction] of Blood": "hematocrit_latest",
    "Hematocrit [Volume Fraction] of Blood by Automated count": "hematocrit_latest",
    "RBC Distribution Width": "rdw_latest",
    "Red blood cells [#/volume] in Blood": "rbc_latest",

    # Liver-related (we might drop some later in modeling, but it's fine to compute them here)
    "AST": "ast_latest",
    "ALT": "alt_latest",
    "AST (Elevated)": "ast_latest",
    "ALT (Elevated)": "alt_latest",
    "Bilirubin.total [Mass/volume] in Serum or Plasma": "bilirubin_latest",
    "Albumin [Mass/volume] in Serum or Plasma": "albumin_latest",
    "Albumin": "albumin_latest",
    "Protein [Mass/volume] in Serum or Plasma": "protein_latest",
}


# -----------------------------------------------------------
# Build lab features from df_obs (latest per patient per lab)
# -----------------------------------------------------------

def build_lab_features_from_obs(df_obs: pd.DataFrame) -> pd.DataFrame:
    """
    Take the observation DataFrame (as returned by parse_patient_bundle),
    and produce a wide table:

        patient_id, glucose_latest, hba1c_latest, creatinine_latest, ...

    Using the *latest* available value per lab per patient.
    """

    required_cols = {"patient_id", "code_display", "value_quantity", "effective_datetime"}
    missing = required_cols - set(df_obs.columns)
    if missing:
        raise ValueError(f"df_obs is missing required columns: {missing}")

    # Keep only labs we care about
    obs = df_obs[df_obs["code_display"].isin(LAB_MAPPING.keys())].copy()
    if obs.empty:
        # No matching labs; return an empty frame with just patient_id (if present)
        # We'll handle missing labs later in the pipeline.
        return pd.DataFrame(columns=["patient_id"])

    # Map to unified feature names
    obs["feature_name"] = obs["code_display"].map(LAB_MAPPING)

    # Convert date
    if not np.issubdtype(obs["effective_datetime"].dtype, np.datetime64):
        obs["effective_datetime"] = pd.to_datetime(obs["effective_datetime"], errors="coerce")

    # Sort by most recent
    obs = obs.sort_values(
        ["patient_id", "feature_name", "effective_datetime"],
        ascending=[True, True, False]
    )

    # Keep only the latest per patient + feature
    obs_latest = obs.drop_duplicates(
        subset=["patient_id", "feature_name"],
        keep="first"
    )

    # Pivot to wide format
    lab_features = obs_latest.pivot_table(
        index="patient_id",
        columns="feature_name",
        values="value_quantity",
        aggfunc="first"
    ).reset_index()

    # Ensure numeric labs
    for col in lab_features.columns:
        if col != "patient_id":
            lab_features[col] = pd.to_numeric(lab_features[col], errors="coerce")

    return lab_features


# -----------------------------------------------------------
# Build demographics (age, sex) from df_patients
# -----------------------------------------------------------

def build_demographics_from_patients(
    df_patients: pd.DataFrame,
    reference_date: str = "2025-01-01"
) -> pd.DataFrame:
    """
    From the patient table, compute age and sex.

    df_patients is expected to have columns:
        - patient_id
        - gender
        - birth_date

    We return:
        patient_id, age, sex
    """

    required_cols = {"patient_id", "gender", "birth_date"}
    missing = required_cols - set(df_patients.columns)
    if missing:
        raise ValueError(f"df_patients is missing required columns: {missing}")

    patients = df_patients.copy()

    patients["birth_date"] = pd.to_datetime(patients["birth_date"], errors="coerce")
    ref_date = pd.to_datetime(reference_date)

    # Compute age in years
    patients["age"] = (ref_date - patients["birth_date"]).dt.days // 365

    # Original: single-letter M/F
    patients["sex"] = patients["gender"].str[:1].str.upper()

    # 🔧 NEW: map to numeric (adjust mapping if your training used the opposite)
    sex_map = {"M": 1, "F": 0}
    patients["sex"] = patients["sex"].map(sex_map).astype("float32")

    demo = patients[["patient_id", "age", "sex"]].copy()
    return demo



# -----------------------------------------------------------
# Combine into a single feature row per patient
# -----------------------------------------------------------

def build_feature_table_for_bundle(
    df_patients: pd.DataFrame,
    df_obs: pd.DataFrame,
    reference_date: str = "2025-01-01"
) -> pd.DataFrame:
    """
    Main entry point:

    Given:
        df_patients, df_obs as returned by parse_patient_bundle(...)
    Returns:
        A DataFrame with one row per patient and columns:

            patient_id, age, sex, <lab features...>

    In your app, you will typically use only the first row, since
    the uploaded file should represent a single patient.
    """

    demo = build_demographics_from_patients(df_patients, reference_date=reference_date)
    labs = build_lab_features_from_obs(df_obs)

    if labs.empty:
        # No labs found; still return age/sex so the pipeline can decide what to do.
        feature_table = demo.copy()
    else:
        feature_table = demo.merge(labs, on="patient_id", how="left")

    return feature_table
