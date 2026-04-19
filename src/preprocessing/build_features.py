from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd

# -----------------------------------------------------------
# Lab Mapping Dictionary
# -----------------------------------------------------------

LAB_MAPPING: Dict[str, str] = {
    "Glucose [Mass/volume] in Blood": "glucose_latest",
    "Glucose": "glucose_latest",
    "Hemoglobin A1c/Hemoglobin.total in Blood": "hba1c_latest",
    "Hemoglobin A1c": "hba1c_latest",
    "Creatinine": "creatinine_latest",
    "Creatinine [Mass/volume] in Serum or Plasma": "creatinine_latest",
    "Creatinine [Mass/volume] in Blood": "creatinine_latest",
    "Glomerular filtration rate/1.73 sq M.predicted": "egfr_latest",
    "Glomerular filtration rate/1.73 sq M.predicted [Volume Rate/Area] in Serum or Plasma by Creatinine-based formula (MDRD)": "egfr_latest",
    "Urea Nitrogen": "bun_latest",
    "Urea nitrogen [Mass/volume] in Serum or Plasma": "bun_latest",
    "Urea nitrogen [Mass/volume] in Blood": "bun_latest",
    "Cholesterol in HDL [Mass/volume] in Serum or Plasma": "hdl_latest",
    "Low Density Lipoprotein Cholesterol": "ldl_latest",
    "Triglycerides": "triglycerides_latest",
    "Cholesterol [Mass/volume] in Serum or Plasma": "cholesterol_total_latest",
    "Hemoglobin [Mass/volume] in Blood": "hemoglobin_latest",
    "Hematocrit [Volume Fraction] of Blood": "hematocrit_latest",
    "Hematocrit [Volume Fraction] of Blood by Automated count": "hematocrit_latest",
    "RBC Distribution Width": "rdw_latest",
    "Red blood cells [#/volume] in Blood": "rbc_latest",
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
# 1. Build lab features from df_obs
# -----------------------------------------------------------

def build_lab_features_from_obs(df_obs: pd.DataFrame) -> pd.DataFrame:
    """
    Produces a wide table of lab results. 
    Forced datetime conversion at the start prevents the NumPy 2.x/Python 3.13 crash.
    """
    if df_obs is None or df_obs.empty:
        return pd.DataFrame()

    # Create a local copy to avoid SettingWithCopy warnings
    df_obs = df_obs.copy()

    # ✅ STEP 1: FORCE CONVERSION IMMEDIATELY
    # This prevents the NumPy 'issubdtype' crash during drop_duplicates
    if "effective_datetime" in df_obs.columns:
        df_obs["effective_datetime"] = pd.to_datetime(df_obs["effective_datetime"], errors="coerce")
    
    # ✅ STEP 2: FILTER
    obs = df_obs[df_obs["code_display"].isin(LAB_MAPPING.keys())].copy()
    if obs.empty:
        return pd.DataFrame(columns=["patient_id"])

    # ✅ STEP 3: MAP
    obs["feature_name"] = obs["code_display"].map(LAB_MAPPING)

    # ✅ STEP 4: SORT AND DROP DUPLICATES
    # The sort and drop will now work because 'effective_datetime' is a proper datetime64 type
    obs = obs.sort_values(
        ["patient_id", "feature_name", "effective_datetime"],
        ascending=[True, True, False]
    )

    obs_latest = obs.drop_duplicates(
        subset=["patient_id", "feature_name"],
        keep="first"
    )

    # ✅ STEP 5: PIVOT
    lab_features = obs_latest.pivot_table(
        index="patient_id",
        columns="feature_name",
        values="value_quantity",
        aggfunc="first"
    ).reset_index()

    # Ensure all lab columns are numeric
    for col in lab_features.columns:
        if col != "patient_id":
            lab_features[col] = pd.to_numeric(lab_features[col], errors="coerce")

    return lab_features

# -----------------------------------------------------------
# 2. Build demographics from df_patients
# -----------------------------------------------------------

def build_demographics_from_patients(
    df_patients: pd.DataFrame,
    reference_date: str = "2025-01-01"
) -> pd.DataFrame:
    required_cols = {"patient_id", "gender", "birth_date"}
    if not required_cols.issubset(df_patients.columns):
        return pd.DataFrame(columns=["patient_id", "age", "sex"])

    patients = df_patients.copy()
    patients["birth_date"] = pd.to_datetime(patients["birth_date"], errors="coerce")
    ref_date = pd.to_datetime(reference_date)

    # Compute age
    patients["age"] = (ref_date - patients["birth_date"]).dt.days // 365

    # Map sex to numeric (1=M, 0=F)
    sex_map = {"M": 1.0, "F": 0.0}
    patients["sex"] = patients["gender"].str[:1].str.upper().map(sex_map).fillna(0.0).astype("float32")

    return patients[["patient_id", "age", "sex"]]

# -----------------------------------------------------------
# 3. Main Orchestrator for Streamlit
# -----------------------------------------------------------

def build_feature_table_for_bundle(
    df_patients: pd.DataFrame,
    df_obs: pd.DataFrame,
    reference_date: str = "2025-01-01"
) -> pd.DataFrame:
    """
    Main entry point for the Streamlit app.
    """
    demo = build_demographics_from_patients(df_patients, reference_date=reference_date)
    labs = build_lab_features_from_obs(df_obs)

    if labs.empty:
        return demo.copy()
    
    # Final merge
    feature_table = demo.merge(labs, on="patient_id", how="left")
    return feature_table
