from __future__ import annotations
from typing import Dict
import math
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
# helpers
# -----------------------------------------------------------

def _to_py_str(v) -> str:
    """Unconditionally produce a native Python str — works on numpy.str_, bytes, anything."""
    return str.__new__(str, v) if type(v) is str else str(v)

def _to_py_float(v) -> float:
    """Unconditionally produce a native Python float (or math.nan for missing)."""
    if v is None:
        return math.nan
    t = type(v)
    if t is float:
        return v
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan

# -----------------------------------------------------------
# 1. Build lab features from df_obs
# -----------------------------------------------------------

def build_lab_features_from_obs(df_obs: pd.DataFrame) -> pd.DataFrame:
    """
    Produces a wide (one-row-per-patient) table of latest lab values.

    NumPy 2.x / Python 3.13: numpy.issubdtype() is triggered by ANY
    operation that hashes, compares, or introspects a NumPy scalar —
    including dict/set lookups and even type() checks in some builds.

    The only safe approach: cast every value to a *native* Python type
    via str() / float() (not .tolist(), which can still return NumPy
    scalars in certain pandas/NumPy version combos) before touching
    any Python collection.
    """
    if df_obs is None or df_obs.empty:
        return pd.DataFrame()

    # -- filter ---------------------------------------------------------------
    obs = df_obs[df_obs["code_display"].isin(LAB_MAPPING.keys())].copy()
    if obs.empty:
        return pd.DataFrame(columns=["patient_id"])

    # -- type-safe conversions on the filtered copy ---------------------------
    if "effective_datetime" in obs.columns:
        obs["effective_datetime"] = pd.to_datetime(
            obs["effective_datetime"], errors="coerce"
        )
    obs["value_quantity"] = pd.to_numeric(obs["value_quantity"], errors="coerce")
    obs["feature_name"]   = obs["code_display"].map(LAB_MAPPING)

    # Convert date
    obs["effective_datetime"] = pd.to_datetime(obs["effective_datetime"], errors="coerce", utc=True)


    # -- build wide dict using only native Python scalars ---------------------
    # We iterate over the raw values arrays directly (avoiding itertuples /
    # iterrows which still box values as NumPy scalars), then EXPLICITLY cast
    # each element to str / float so no NumPy type ever enters a Python
    # collection or comparison.
    pid_arr  = obs["patient_id"].values
    feat_arr = obs["feature_name"].values
    val_arr  = obs["value_quantity"].values

    wide: dict = {}   # { str -> { str -> float } }
    seen: set  = set()

    for i in range(len(pid_arr)):
        pid  = _to_py_str(pid_arr[i])    # guaranteed native Python str
        feat = _to_py_str(feat_arr[i])   # guaranteed native Python str
        val  = _to_py_float(val_arr[i])  # guaranteed native Python float

        key = pid + "\x00" + feat        # string concat — no tuple hashing
        if key in seen:
            continue
        seen.add(key)

        if pid not in wide:
            wide[pid] = {}
        wide[pid][feat] = val

    if not wide:
        return pd.DataFrame(columns=["patient_id"])

    # -- construct DataFrame from pure-Python dict ----------------------------
    lab_features = pd.DataFrame.from_dict(wide, orient="index")
    lab_features.index.name = "patient_id"
    lab_features = lab_features.reset_index()

    for col in lab_features.columns:
        if col != "patient_id":
            lab_features[col] = pd.to_numeric(lab_features[col], errors="coerce")

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
