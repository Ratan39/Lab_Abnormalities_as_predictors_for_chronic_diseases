from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple

import json
import pandas as pd
from xgboost import XGBClassifier


# -----------------------------------------------------------
# Paths to model directory and files
# -----------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICTION_DIR = PROJECT_ROOT / "models" / "prediction"

CKD_MODEL_PATH = PREDICTION_DIR / "xgb_ckd_model.json"
CVD_MODEL_PATH = PREDICTION_DIR / "xgb_cvd_model.json"
ANEMIA_MODEL_PATH = PREDICTION_DIR / "xgb_anemia_model.json"
PREDM_MODEL_PATH = PREDICTION_DIR / "xgb_predm_model.json"
FEATURE_COLS_PATH = PREDICTION_DIR / "feature_columns.json"


# -----------------------------------------------------------
# Load feature column order (for safety / checking)
# -----------------------------------------------------------

with open(FEATURE_COLS_PATH, "r") as f:
    MODEL_FEATURE_COLUMNS = json.load(f)


# -----------------------------------------------------------
# Model loading
# -----------------------------------------------------------

def load_all_models() -> Dict[str, XGBClassifier]:
    """
    Load all 4 XGBoost models from disk.

    Returns:
        {
          "ckd":    XGBClassifier,
          "cvd":    XGBClassifier,
          "anemia": XGBClassifier,
          "predm":  XGBClassifier
        }
    """
    models: Dict[str, XGBClassifier] = {}

    m_ckd = XGBClassifier()
    m_ckd.load_model(CKD_MODEL_PATH)
    models["ckd"] = m_ckd

    m_cvd = XGBClassifier()
    m_cvd.load_model(CVD_MODEL_PATH)
    models["cvd"] = m_cvd

    m_anemia = XGBClassifier()
    m_anemia.load_model(ANEMIA_MODEL_PATH)
    models["anemia"] = m_anemia

    m_predm = XGBClassifier()
    m_predm.load_model(PREDM_MODEL_PATH)
    models["predm"] = m_predm

    return models


# -----------------------------------------------------------
# Prediction helpers
# -----------------------------------------------------------

def _predict_proba_single(model: XGBClassifier, X: pd.DataFrame) -> float:
    """
    Predict the positive class probability for a single-row DataFrame X.

    Assumes X has the same feature order as during training.
    """
    if X.shape[0] != 1:
        raise ValueError(f"Expected X with 1 row, got {X.shape[0]} rows.")
    proba = model.predict_proba(X)[:, 1][0]
    return float(proba)


def predict_all_diseases(
    X_ready: pd.DataFrame,
    models: Dict[str, XGBClassifier],
    threshold: float = 0.5,
) -> Dict[str, Dict[str, Any]]:
    """
    Run all 4 disease models on a single patient row.

    Inputs:
        X_ready : DataFrame with 1 row and columns in the exact order used during training
        models  : dict from load_all_models()
        threshold : probability cutoff for binary label (default 0.5)

    Returns:
        {
          "ckd":    {"prob": 0.12, "label": 0},
          "cvd":    {"prob": 0.47, "label": 0},
          "anemia": {"prob": 0.09, "label": 0},
          "predm":  {"prob": 0.62, "label": 1},
        }
    """

    results: Dict[str, Dict[str, Any]] = {}

    p_ckd = _predict_proba_single(models["ckd"], X_ready)
    p_cvd = _predict_proba_single(models["cvd"], X_ready)
    p_anemia = _predict_proba_single(models["anemia"], X_ready)
    p_predm = _predict_proba_single(models["predm"], X_ready)

    results["ckd"] = {
        "prob": p_ckd,
        "label": int(p_ckd >= threshold),
    }
    results["cvd"] = {
        "prob": p_cvd,
        "label": int(p_cvd >= threshold),
    }
    results["anemia"] = {
        "prob": p_anemia,
        "label": int(p_anemia >= threshold),
    }
    results["predm"] = {
        "prob": p_predm,
        "label": int(p_predm >= threshold),
    }

    return results
