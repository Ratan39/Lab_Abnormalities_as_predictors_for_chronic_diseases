from __future__ import annotations
from typing import Dict
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
    Produces a wide (one-row-per-patient) table of latest lab values.

    NumPy 2.x / Python 3.13 issue: numpy.issubdtype() is triggered by ANY
    operation that hashes or compares NumPy scalar types (numpy.str_,
    numpy.float64, etc.) — including set membership, dict lookup, and all
    reshape ops (pivot, pivot_table, unstack).

    Fix: extract only the three needed columns as plain Python lists via
    .tolist() BEFORE any looping or collection operations, so every value
    is a native Python str/float/None and NumPy scalars never appear.
    """
    if df_obs is None or df_obs.empty:
        return pd.DataFrame()

    # -- filter ---------------------------------------------------------------
    obs = df_obs[df_obs["code_display"].isin(LAB_MAPPING.keys())].copy()
    if obs.empty:
        return pd.DataFrame(columns=["patient_id"])

    # -- type-safe conversions ------------------------------------------------
    if "effective_datetime" in obs.columns:
        obs["effective_datetime"] = pd.to_datetime(
            obs["effective_datetime"], errors="coerce"
        )
    obs["value_quantity"] = pd.to_numeric(obs["value_quantity"], errors="coerce")
    obs["feature_name"]   = obs["code_display"].map(LAB_MAPPING)

    # -- sort most-recent first -----------------------------------------------
    obs = obs.sort_values(
        ["patient_id", "feature_name", "effective_datetime"],
        ascending=[True, True, False],
        na_position="last",
    )

    # -- extract as plain Python lists (kills ALL NumPy scalar types) ---------
    # .tolist() on a pandas Series converts every element to a native Python
    # type: numpy.str_ -> str, numpy.float64 -> float, NaT -> None, etc.
    # After this point there are zero NumPy objects in play.
    patient_ids   = obs["patient_id"].tolist()      # list[str]
    feature_names = obs["feature_name"].tolist()    # list[str]
    values        = obs["value_quantity"].tolist()  # list[float | None]

    # -- build wide dict with plain Python types only -------------------------
    wide: dict[str, dict[str, float]] = {}
    seen: set[tuple[str, str]] = set()

    for pid, feat, val in zip(patient_ids, feature_names, values):
        # pid and feat are now guaranteed native Python str — safe to hash
        key = (pid, feat)
        if key in seen:
            continue
        seen.add(key)
        if pid not in wide:
            wide[pid] = {}
        wide[pid][feat] = val   # val is native Python float or None

    if not wide:
        return pd.DataFrame(columns=["patient_id"])

    # -- construct DataFrame from pure-Python dict ----------------------------
    lab_features = pd.DataFrame.from_dict(wide, orient="index")
    lab_features.index.name = "patient_id"
    lab_features = lab_features.reset_index()

    # Ensure lab columns are float64
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
        obs["effective_datetime"] = pd.to_datetime(
            obs["effective_datetime"], errors="coerce"
        )
    obs["value_quantity"] = pd.to_numeric(obs["value_quantity"], errors="coerce")
    obs["feature_name"]   = obs["code_display"].map(LAB_MAPPING)

    # -- sort so the most-recent row comes first per (patient, feature) -------
    obs = obs.sort_values(
        ["patient_id", "feature_name", "effective_datetime"],
        ascending=[True, True, False],
        na_position="last",
    )

    # -- build wide table via pure-Python dict — no NumPy reshape at all ------
    # { patient_id: { feature_name: value, ... }, ... }
    wide: dict[str, dict[str, float]] = {}
    seen: set[tuple] = set()

    for row in obs.itertuples(index=False):
        pid     = row.patient_id
        feat    = row.feature_name
        val     = row.value_quantity          # already float / NaN
        key     = (pid, feat)
        if key in seen:
            continue                          # keep first (= most recent)
        seen.add(key)
        if pid not in wide:
            wide[pid] = {}
        wide[pid][feat] = val

    if not wide:
        return pd.DataFrame(columns=["patient_id"])

    # -- assemble DataFrame from the dict directly ----------------------------
    lab_features = pd.DataFrame.from_dict(wide, orient="index")
    lab_features.index.name = "patient_id"
    lab_features = lab_features.reset_index()

    # Guarantee all lab columns are float64 (not object)
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
