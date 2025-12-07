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
# Load all datasets
# -----------------------------------------------------------

# Observations (multiple files)
obs_files = sorted(DATA_RAW.glob("observations_*.parquet"))
if not obs_files:
    raise ValueError("No observation files found: expected observations_*.parquet")

df_obs = pd.concat((pd.read_parquet(f) for f in obs_files), ignore_index=True)
print(f"Loaded observations: {len(df_obs)} rows from {len(obs_files)} files")

# Conditions (multiple files)
cond_files = sorted(DATA_RAW.glob("conditions_*.parquet"))
if not cond_files:
    raise ValueError("No condition files found: expected conditions_*.parquet")

df_conditions = pd.concat((pd.read_parquet(f) for f in cond_files), ignore_index=True)
print(f"Loaded conditions: {len(df_conditions)} rows from {len(cond_files)} files")

# Patients (multiple files)
patient_files = sorted(DATA_RAW.glob("patients_*.parquet"))
if not patient_files:
    raise ValueError("No patient files found: expected patients_*.parquet")

df_patients = pd.concat((pd.read_parquet(f) for f in patient_files), ignore_index=True)
print(f"Loaded patients: {len(df_patients)} rows from {len(patient_files)} files")

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

    # Liver (still useful as raw labs, even if we don't have a liver label)
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

def build_lab_features_table() -> pd.DataFrame:
    # Filter by labs we care about
    obs = df_obs[df_obs["code_display"].isin(LAB_MAPPING.keys())].copy()
    if obs.empty:
        raise ValueError("No observations matched LAB_MAPPING. Check your code_display strings.")

    # Map to unified names
    obs["feature_name"] = obs["code_display"].map(LAB_MAPPING)

    # Convert date
    if not np.issubdtype(obs["effective_datetime"].dtype, np.datetime64):
        obs["effective_datetime"] = pd.to_datetime(obs["effective_datetime"])

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

    print(f"Lab features shape: {lab_features.shape}")
    return lab_features

# -----------------------------------------------------------
# Build Demographics Table
# -----------------------------------------------------------

def build_demographics_table(reference_date: str = "2025-01-01") -> pd.DataFrame:
    """
    reference_date: used to compute age, in YYYY-MM-DD format.
    Uses the global df_patients loaded at module import.
    """
    patients = df_patients.copy()
    print(f"Building demographics from patients (rows: {len(patients)})")

    # Keep only columns we care about
    cols = ["patient_id", "gender", "birth_date"]
    missing = [c for c in cols if c not in patients.columns]
    if missing:
        raise ValueError(f"Missing columns in patients data: {missing}")
    patients = patients[cols].copy()

    # Convert dates
    patients["birth_date"] = pd.to_datetime(patients["birth_date"])
    ref_date = pd.to_datetime(reference_date)

    # Compute age in years
    patients["age"] = (ref_date - patients["birth_date"]).dt.days // 365

    # Clean gender to simple categories
    patients["sex"] = patients["gender"].str[0].str.upper()  # e.g., "M" or "F"

    demo = patients[["patient_id", "age", "sex"]].copy()
    print(f"Demographics shape: {demo.shape}")
    return demo

# -----------------------------------------------------------
# Build Label Table (4 diseases: CKD, CVD, Anemia, Prediabetes)
# -----------------------------------------------------------

def build_label_table() -> pd.DataFrame:
    cond = df_conditions.copy()
    cond["code_display_lower"] = cond["code_display"].str.lower().fillna("")

    def has_any_keyword(series: pd.Series, keywords: list[str]) -> pd.Series:
        pattern = "|".join(k.lower() for k in keywords)
        return series.str.contains(pattern, na=False)

    # CKD (based on your actual condition names)
    ckd_keywords = [
        "chronic kidney disease",
        "ckd",
        "ckd stage",
        "renal failure",
        "renal insufficiency",
        "end-stage renal disease",
        "esrd",
        "renal transplant",
        "kidney transplant",
        "awaiting transplantation of kidney",
        "kidney transplant failure",
        "renal dysplasia",
        "disorder of kidney due to diabetes",
        "diabetic nephropathy",
    ]

    # Cardiovascular disease (CVD)
    cvd_keywords = [
        "essential hypertension",
        "hypertension",
        "ischemic heart disease",
        "myocardial infarction",
        "history of myocardial infarction",
        "history of coronary artery bypass grafting",
        "coronary artery",
        "heart failure",
        "chronic congestive heart failure",
        "congestive heart failure",
        "acute st segment elevation myocardial infarction",
        "acute non-st segment elevation myocardial infarction",
        "ischemic",
        "angina",
        "cardiac",
        "stroke",  # in case stroke codes are present
    ]

    # Anemia
    anemia_keywords = [
        "anemia",
        "anaemia",
        "iron deficiency",
    ]

    # Prediabetes / hyperglycemia
    predm_keywords = [
        "prediabetes",
        "hyperglycemia",
        "impaired fasting glucose",
        "impaired glucose tolerance",
        "abnormal glucose",
    ]

    cond["label_ckd"] = has_any_keyword(cond["code_display_lower"], ckd_keywords).astype(int)
    cond["label_cvd"] = has_any_keyword(cond["code_display_lower"], cvd_keywords).astype(int)
    cond["label_anemia"] = has_any_keyword(cond["code_display_lower"], anemia_keywords).astype(int)
    cond["label_predm"] = has_any_keyword(cond["code_display_lower"], predm_keywords).astype(int)

    label_cols = ["label_ckd", "label_cvd", "label_anemia", "label_predm"]

    # Aggregate to patient-level: if patient has ANY row with label=1 → label=1
    label_table = (
        cond.groupby("patient_id")[label_cols]
        .max()
        .reset_index()
    )

    print(f"Label table shape: {label_table.shape}")
    return label_table

# -----------------------------------------------------------
# Build Final Patient Table
# -----------------------------------------------------------

def build_patient_table(output_path: Path | None = None) -> pd.DataFrame:
    if output_path is None:
        output_path = DATA_INTERIM / "patient_table.parquet"

    demo = build_demographics_table()
    labs = build_lab_features_table()
    labels = build_label_table()

    # Start from labels (so every row has at least label info)
    patient_table = labels.merge(demo, on="patient_id", how="left")
    patient_table = patient_table.merge(labs, on="patient_id", how="left")

    print(f"Final patient_table shape: {patient_table.shape}")
    print(f"Saving patient_table to {output_path}")
    patient_table.to_parquet(output_path, index=False)

    return patient_table

if __name__ == "__main__":
    build_patient_table()
