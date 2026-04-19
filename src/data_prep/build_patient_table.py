from pathlib import Path
import pandas as pd
import numpy as np

# -----------------------------------------------------------
# Base directories
# -----------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_INTERIM.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------
# Lab Mapping Dictionary
# -----------------------------------------------------------

LAB_MAPPING = {
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

    # Liver
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
# Build Lab Features Table
# -----------------------------------------------------------

def build_lab_features_from_obs(obs: pd.DataFrame) -> pd.DataFrame:
    """
    Called by the Streamlit app and the batch pipeline.
    This replaces the problematic NumPy check with a robust Pandas check.
    """
    if obs is None or obs.empty:
        return pd.DataFrame()

    obs = obs.copy()
    
    # Filter by labs we care about
    obs = obs[obs["code_display"].isin(LAB_MAPPING.keys())].copy()
    if obs.empty:
        return pd.DataFrame()

    # Map to unified names
    obs["feature_name"] = obs["code_display"].map(LAB_MAPPING)

    # UPDATED: Robust datetime conversion for Python 3.13 compatibility
    if not pd.api.types.is_datetime64_any_dtype(obs["effective_datetime"]):
        obs["effective_datetime"] = pd.to_datetime(obs["effective_datetime"], errors='coerce')

    # Sort by most recent
    obs = obs.sort_values(
        ["patient_id", "feature_name", "effective_datetime"],
        ascending=[True, True, False]
    )

    # Keep only latest per patient + feature
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

    return lab_features

# -----------------------------------------------------------
# Build Demographics Table
# -----------------------------------------------------------

def build_demographics_table(df_patients: pd.DataFrame, reference_date: str = "2025-01-01") -> pd.DataFrame:
    patients = df_patients.copy()

    # Keep only columns we care about
    cols = ["patient_id", "gender", "birth_date"]
    patients = patients[[c for c in cols if c in patients.columns]].copy()

    # Convert dates
    patients["birth_date"] = pd.to_datetime(patients["birth_date"], errors='coerce')
    ref_date = pd.to_datetime(reference_date)

    # Compute age in years
    patients["age"] = (ref_date - patients["birth_date"]).dt.days // 365

    # Clean gender
    if "gender" in patients.columns:
        patients["sex"] = patients["gender"].str[0].str.upper()
    else:
        patients["sex"] = "U"

    return patients[["patient_id", "age", "sex"]]

# -----------------------------------------------------------
# Final Orchestrator for Streamlit
# -----------------------------------------------------------

def build_feature_table_for_bundle(df_patients, df_obs, reference_date="2025-01-01"):
    """
    This is the main function called by app/test.py
    """
    demo = build_demographics_table(df_patients, reference_date)
    labs = build_lab_features_from_obs(df_obs)

    if labs.empty:
        return demo

    # Merge demographics with latest lab results
    feature_table = demo.merge(labs, on="patient_id", how="left")
    
    # Ensure 'sex' is numeric for the model if needed (1=M, 0=F)
    if "sex" in feature_table.columns:
        feature_table["sex"] = feature_table["sex"].map({"M": 1, "F": 0}).fillna(0)

    return feature_table
