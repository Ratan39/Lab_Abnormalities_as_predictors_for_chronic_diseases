from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import json


# -----------------------------------------------------------
# Project paths
# -----------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLUSTER_DIR = PROJECT_ROOT / "models" / "clustering"
PREDICTION_DIR = PROJECT_ROOT / "models" / "prediction"

IMPUTER_PATH = CLUSTER_DIR / "imputer.joblib"
SCALER_PATH = CLUSTER_DIR / "scaler.joblib"
PCA_PATH = CLUSTER_DIR / "pca_14components.joblib"
KMEANS_PATH = CLUSTER_DIR / "kmeans_k4.joblib"

FEATURE_COLS_PATH = PREDICTION_DIR / "feature_columns.json"


# -----------------------------------------------------------
# Clustering feature columns (exactly as used in training)
# -----------------------------------------------------------

CLUSTER_FEATURE_COLUMNS: List[str] = [
    "age",
    "sex",
    "albumin_latest",
    "alt_latest",
    "ast_latest",
    "bilirubin_latest",
    "bun_latest",
    "cholesterol_total_latest",
    "creatinine_latest",
    "egfr_latest",
    "glucose_latest",
    "hba1c_latest",
    "hdl_latest",
    "hematocrit_latest",
    "hemoglobin_latest",
    "ldl_latest",
    "protein_latest",
    "rdw_latest",
    "triglycerides_latest",
]


# -----------------------------------------------------------
# Load imputer, scaler, PCA, KMeans, and model feature columns
# -----------------------------------------------------------

imputer = joblib.load(IMPUTER_PATH)
scaler = joblib.load(SCALER_PATH)
pca = joblib.load(PCA_PATH)
kmeans = joblib.load(KMEANS_PATH)

with open(FEATURE_COLS_PATH, "r") as f:
    MODEL_FEATURE_COLUMNS: List[str] = json.load(f)


# -----------------------------------------------------------
# Helper: prepare features for clustering
# -----------------------------------------------------------

def _prepare_features_for_clustering(feature_table: pd.DataFrame) -> pd.DataFrame:
    """
    Turn the feature_table (patient_id, age, sex, labs...) into
    the numeric feature matrix used for imputer/scaler/PCA/KMeans.

    We:
      - drop patient_id
      - encode sex as 0/1 if needed
      - coerce numeric
      - add any missing clustering columns as NaN
      - reorder to CLUSTER_FEATURE_COLUMNS (exact training order)
    """

    if feature_table.empty:
        raise ValueError("feature_table is empty; cannot compute cluster.")

    X = feature_table.copy()

    # Drop patient_id if present
    if "patient_id" in X.columns:
        X = X.drop(columns=["patient_id"])

    # Drop cluster if somehow present (shouldn't be yet)
    if "cluster" in X.columns:
        X = X.drop(columns=["cluster"])

    # Encode sex if present as string
    if "sex" in X.columns and X["sex"].dtype == object:
        X["sex"] = (
            X["sex"].str.upper()
            .map({"M": 1, "F": 0})
            .astype("float64")
        )

    # Coerce all existing columns to numeric (non-numeric → NaN)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Ensure all clustering features exist (missing ones → NaN)
    for col in CLUSTER_FEATURE_COLUMNS:
        if col not in X.columns:
            X[col] = np.nan

    # Reorder to EXACT same order as at training time
    X = X[CLUSTER_FEATURE_COLUMNS].copy()

    return X


# -----------------------------------------------------------
# Main function: add cluster + align to model feature columns
# -----------------------------------------------------------

def add_cluster_and_align_for_models(
    feature_table: pd.DataFrame,
    model_feature_columns: List[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Given:
        feature_table: output of build_feature_table_for_bundle(...)
                       (one row per patient, with patient_id, age, sex, labs...)

    Returns:
        feature_table_with_cluster: original table plus 'cluster' column
        X_ready: DataFrame with columns in the exact order required by
                 the XGBoost models (MODEL_FEATURE_COLUMNS), including 'cluster'.
    """

    if model_feature_columns is None:
        model_feature_columns = MODEL_FEATURE_COLUMNS

    if feature_table.empty:
        raise ValueError("feature_table is empty; cannot prepare model input.")

    # 1) Prepare matrix for clustering
    X_cluster = _prepare_features_for_clustering(feature_table)

    # 2) Apply imputer -> scaler -> PCA -> KMeans
    X_imputed = imputer.transform(X_cluster)
    X_scaled = scaler.transform(X_imputed)
    X_pca = pca.transform(X_scaled)
    cluster_labels = kmeans.predict(X_pca)

    # 3) Add 'cluster' back to feature_table copy
    ft_with_cluster = feature_table.copy()
    ft_with_cluster["cluster"] = cluster_labels.astype(int)

    # 4) Ensure all model feature columns exist (for XGBoost models)
    for col in model_feature_columns:
        if col not in ft_with_cluster.columns:
            ft_with_cluster[col] = np.nan

    # 5) Reorder into exact feature order for models
    X_ready = ft_with_cluster[model_feature_columns].copy()

    return ft_with_cluster, X_ready
