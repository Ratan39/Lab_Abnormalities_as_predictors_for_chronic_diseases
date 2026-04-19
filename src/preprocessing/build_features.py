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

    NumPy 2.x / Python 3.13 compatibility notes
    --------------------------------------------
    * pivot_table() calls numpy.issubdtype() internally and crashes when any
      column still carries object dtype.  We use plain pivot() instead, which
      skips that code path entirely.
    * Datetime conversion must happen on the filtered `obs` copy (not the
      original df_obs) because a boolean-slice + .copy() can re-materialise
      the original object dtype on some pandas/NumPy combos.
    """
    if df_obs is None or df_obs.empty:
        return pd.DataFrame()

    # STEP 1: filter to only the lab codes we care about
    obs = df_obs[df_obs["code_display"].isin(LAB_MAPPING.keys())].copy()
    if obs.empty:
        return pd.DataFrame(columns=["patient_id"])

    # STEP 2: force datetime dtype on the filtered copy
    if "effective_datetime" in obs.columns:
        obs["effective_datetime"] = pd.to_datetime(
            obs["effective_datetime"], errors="coerce"
        )

    # STEP 3: map display name -> feature column name
    obs["feature_name"] = obs["code_display"].map(LAB_MAPPING)

    # STEP 4: coerce value to numeric BEFORE pivoting
    # Ensures the pivot column is float64, never object, sidestepping
    # any remaining issubdtype probes inside pandas/NumPy internals.
    obs["value_quantity"] = pd.to_numeric(obs["value_quantity"], errors="coerce")

    # STEP 5: keep only the most-recent row per (patient, feature)
    obs = obs.sort_values(
        ["patient_id", "feature_name", "effective_datetime"],
        ascending=[True, True, False],
        na_position="last",
    )
    obs_latest = obs.drop_duplicates(subset=["patient_id", "feature_name"], keep="first")

    # STEP 6: pivot with plain pivot() — avoids pivot_table's aggfunc/issubdtype path
    # drop_duplicates above guarantees at most one value per (patient, feature),
    # so no aggregation is needed and pivot() is safe.
    lab_features = (
        obs_latest[["patient_id", "feature_name", "value_quantity"]]
        .pivot(index="patient_id", columns="feature_name", values="value_quantity")
        .reset_index()
    )

    # Clean up the column index name added by pivot
    lab_features.columns.name = None

    return lab_features


# -----------------------------------------------------------
# 2. Build demographics from df_patients
# -----------------------------------------------------------

def build_demographics_from_patients(
    df_patients: pd.DataFrame,
    reference_date: str = "2025-01-01",
) -> pd.DataFrame:
    required_cols = {"patient_id", "gender", "birth_date"}
    if not required_cols.issubset(df_patients.columns):
        return pd.DataFrame(columns=["patient_id", "age", "sex"])

    patients = df_patients.copy()
    patients["birth_date"] = pd.to_datetime(patients["birth_date"], errors="coerce")
    ref_date = pd.to_datetime(reference_date)

    patients["age"] = (ref_date - patients["birth_date"]).dt.days // 365

    sex_map = {"M": 1.0, "F": 0.0}
    patients["sex"] = (
        patients["gender"].str[:1].str.upper().map(sex_map).fillna(0.0).astype("float32")
    )

    return patients[["patient_id", "age", "sex"]]


# -----------------------------------------------------------
# 3. Main Orchestrator for Streamlit
# -----------------------------------------------------------

def build_feature_table_for_bundle(
    df_patients: pd.DataFrame,
    df_obs: pd.DataFrame,
    reference_date: str = "2025-01-01",
) -> pd.DataFrame:
    """Main entry point for the Streamlit app."""
    demo = build_demographics_from_patients(df_patients, reference_date=reference_date)
    labs = build_lab_features_from_obs(df_obs)

    if labs.empty:
        return demo.copy()

    feature_table = demo.merge(labs, on="patient_id", how="left")
    return feature_table
