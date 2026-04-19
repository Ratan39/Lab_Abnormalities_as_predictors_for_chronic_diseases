from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import json
import sys

import joblib
import numpy as np
import pandas as pd


# -----------------------------------------------------------
# Project paths
# -----------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLUSTER_DIR    = PROJECT_ROOT / "models" / "clustering"
PREDICTION_DIR = PROJECT_ROOT / "models" / "prediction"

IMPUTER_PATH = CLUSTER_DIR / "imputer.joblib"
SCALER_PATH  = CLUSTER_DIR / "scaler.joblib"
PCA_PATH     = CLUSTER_DIR / "pca_14components.joblib"
KMEANS_PATH  = CLUSTER_DIR / "kmeans_k4.joblib"

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
# Safe loader — clear error message if artifact is missing/stale
# -----------------------------------------------------------

def _load_artifact(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {path}\n"
            f"Run retrain_clustering_artifacts.py to regenerate it."
        )
    try:
        return joblib.load(path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load {path.name} — likely saved on a different "
            f"Python/sklearn version (you are on Python "
            f"{sys.version_info.major}.{sys.version_info.minor}).\n"
            f"Run retrain_clustering_artifacts.py to fix this.\n"
            f"Original error: {e}"
        ) from e


# -----------------------------------------------------------
# Load artifacts at import time
# -----------------------------------------------------------

imputer = _load_artifact(IMPUTER_PATH)
scaler  = _load_artifact(SCALER_PATH)
pca     = _load_artifact(PCA_PATH)
kmeans  = _load_artifact(KMEANS_PATH)

with open(FEATURE_COLS_PATH, "r") as f:
    MODEL_FEATURE_COLUMNS: List[str] = json.load(f)


# -----------------------------------------------------------
# Helper: prepare features for clustering
# -----------------------------------------------------------

def _prepare_features_for_clustering(feature_table: pd.DataFrame) -> pd.DataFrame:
    if feature_table.empty:
        raise ValueError("feature_table is empty; cannot compute cluster.")

    X = feature_table.copy()

    if "patient_id" in X.columns:
        X = X.drop(columns=["patient_id"])
    if "cluster" in X.columns:
        X = X.drop(columns=["cluster"])

    if "sex" in X.columns and X["sex"].dtype == object:
        X["sex"] = (
            X["sex"].str.upper()
            .map({"M": 1, "F": 0})
            .astype("float64")
        )

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    for col in CLUSTER_FEATURE_COLUMNS:
        if col not in X.columns:
            X[col] = np.nan

    return X[CLUSTER_FEATURE_COLUMNS].copy()


# -----------------------------------------------------------
# Main function: add cluster + align to model feature columns
# -----------------------------------------------------------

def add_cluster_and_align_for_models(
    feature_table: pd.DataFrame,
    model_feature_columns: List[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if model_feature_columns is None:
        model_feature_columns = MODEL_FEATURE_COLUMNS

    if feature_table.empty:
        raise ValueError("feature_table is empty; cannot prepare model input.")

    X_cluster      = _prepare_features_for_clustering(feature_table)
    X_imputed      = imputer.transform(X_cluster)
    X_scaled       = scaler.transform(X_imputed)
    X_pca          = pca.transform(X_scaled)
    cluster_labels = kmeans.predict(X_pca)

    ft_with_cluster = feature_table.copy()
    ft_with_cluster["cluster"] = cluster_labels.astype(int)

    for col in model_feature_columns:
        if col not in ft_with_cluster.columns:
            ft_with_cluster[col] = np.nan

    X_ready = ft_with_cluster[model_feature_columns].copy()

    return ft_with_cluster, X_ready