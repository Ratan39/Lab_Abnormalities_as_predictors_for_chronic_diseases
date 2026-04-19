from pathlib import Path
import pandas as pd
import numpy as np

# -----------------------------------------------------------
# Base directories
# -----------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_INTERIM.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------
# Lab Mapping Dictionary
# -----------------------------------------------------------

LAB_MAPPING = {
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
# 1. Build Lab Features (FIXED for Python 3.13)
# -----------------------------------------------------------

def build_lab_features_from_obs(obs: pd.DataFrame) -> pd.DataFrame:
    """
    Standardized function name to match app traceback.
    Replaces problematic np.issubdtype check.
    """
    if obs is None or obs.empty:
        return pd.DataFrame()

    # Filter to relevant labs
    obs = obs[obs["code_display"].isin(LAB_MAPPING.keys())].copy()
    if obs.empty:
        return pd.DataFrame()

    # Map lab names
    obs["feature_name"] = obs["code_display"].map(LAB_MAPPING)

    # FIXED: Robust date conversion
    # Instead of np.issubdtype, we use the Pandas-native check
    if not pd.api.types.is_datetime64_any_dtype(obs["effective_datetime"]):
        obs["effective_datetime"] = pd.to_datetime(obs["effective_datetime"], errors='coerce')

    # Sort and keep latest result per patient per lab
    obs = obs.sort_values(
        ["patient_id", "feature_name", "effective_datetime"],
        ascending=[True, True, False]
    )

    obs_latest = obs.drop_duplicates(
        subset=["patient_id", "feature_name"],
        keep="first"
    )

    # Pivot to wide format (one row per patient)
    lab_features = obs_latest.pivot_table(
        index="patient_id",
        columns="feature_name",
        values="value_quantity",
        aggfunc="first"
    ).reset_index()

    return lab_features

# -----------------------------------------------------------
# 2. Build Demographics
# -----------------------------------------------------------

def build_demographics_table(df_patients: pd.DataFrame, reference_date: str = "2025-01-01") -> pd.DataFrame:
    if df_patients.empty:
        return pd.DataFrame(columns=["patient_id", "age", "sex"])
    
    patients = df_patients.copy()
    ref_date = pd.to_datetime(reference_date)

    # Basic cleaning
    patients["birth_date"] = pd.to_datetime(patients["birth_date"], errors='coerce')
    patients["age"] = (ref_date - patients["birth_date"]).dt.days // 365
    
    # Handle sex encoding (M=1, F=0)
    if "gender" in patients.columns:
        patients["sex"] = patients["gender"].str[0].str.upper().map({"M": 1, "F": 0}).fillna(0)
    else:
        patients["sex"] = 0

    return patients[["patient_id", "age", "sex"]]

# -----------------------------------------------------------
# 3. Main Orchestrator (Called by app/test.py)
# -----------------------------------------------------------

def build_feature_table_for_bundle(df_patients, df_obs, reference_date="2025-01-01"):
    """
    Orchestrates the creation of the final feature table.
    """
    demo_df = build_demographics_table(df_patients, reference_date)
    labs_df = build_lab_features_from_obs(df_obs)

    if labs_df.empty:
        # Return demo info even if no labs are found
        return demo_df

    # Merge results
    feature_table = demo_df.merge(labs_df, on="patient_id", how="left")
    
    return feature_table
