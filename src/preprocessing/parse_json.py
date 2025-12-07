import re
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd


# -----------------------------
# Helpers (Python versions of your Glue helpers)
# -----------------------------

UUID_PATTERN = re.compile(r"urn:uuid:([A-Za-z0-9-]+)")


def ref_to_uuid(ref: Optional[str]) -> Optional[str]:
    """
    Extract the UUID from a FHIR reference like 'urn:uuid:<id>'.
    Returns None if it doesn't match.
    """
    if not ref:
        return None
    m = UUID_PATTERN.search(ref)
    return m.group(1) if m else None


def get_first_or_self(x):
    """
    If x is a list, return x[0] (or None if empty).
    If x is a dict or anything else, return as-is.
    """
    if isinstance(x, list):
        return x[0] if x else None
    return x


def cc_coding(obj: Any) -> Optional[Dict[str, Any]]:
    """
    CodeableConcept helper:
    Given a CodeableConcept that might be:
      - { "coding": [ {...} ], "text": ... } OR
      - [ { "coding": [ {...} ], "text": ... }, ... ]
    return the FIRST coding dict if possible.
    """
    if obj is None:
        return None

    # Normalize list vs scalar
    obj = get_first_or_self(obj)
    if not isinstance(obj, dict):
        return None

    coding = obj.get("coding")
    if coding is None:
        return None

    coding = get_first_or_self(coding)
    if isinstance(coding, dict):
        return coding

    return None


def cc_attr(obj: Any, attr: str) -> Optional[str]:
    """
    Get CodeableConcept->coding[0]->attr (like 'system', 'code', 'display')
    in a way that tolerates lists or single objects.
    """
    coding = cc_coding(obj)
    if coding is None:
        return None
    return coding.get(attr)


def cc_text(obj: Any) -> Optional[str]:
    """
    Get CodeableConcept->text (also tolerant of list vs single).
    """
    if obj is None:
        return None
    obj = get_first_or_self(obj)
    if isinstance(obj, dict):
        return obj.get("text")
    return None


# -----------------------------
# Main parse function
# -----------------------------

def parse_patient_bundle(bundle: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse a FHIR Bundle (like Synthea output) containing multiple resources
    (Patient, Observation, Condition, Encounter, ...).

    Returns:
        df_patients   : columns ['patient_id', 'gender', 'birth_date', ...]
        df_obs        : observation rows with numeric lab values
        df_conditions : condition rows (if any). Not needed for prediction.
    """

    entries: List[Dict[str, Any]] = bundle.get("entry", [])
    resources: List[Dict[str, Any]] = [e.get("resource", {}) for e in entries]

    patients_rows: List[Dict[str, Any]] = []
    obs_rows: List[Dict[str, Any]] = []
    cond_rows: List[Dict[str, Any]] = []

    for res in resources:
        rtype = res.get("resourceType")

        # ------------- PATIENT -------------
        if rtype == "Patient":
            patient_id = res.get("id")
            gender = res.get("gender")
            birth_date = res.get("birthDate")

            # address fields are not strictly needed for the models
            addr = None
            address_list = res.get("address")
            if isinstance(address_list, list) and address_list:
                addr = address_list[0]

            patients_rows.append(
                {
                    "patient_id": patient_id,
                    "gender": gender,
                    "birth_date": birth_date,
                    # keep optional fields if you want them later
                    "address_city": (addr or {}).get("city"),
                    "address_state": (addr or {}).get("state"),
                    "address_postal": (addr or {}).get("postalCode"),
                    "country": (addr or {}).get("country"),
                }
            )

        # ------------- CONDITION -------------
        elif rtype == "Condition":
            condition_id = res.get("id")
            subj_ref = res.get("subject", {}).get("reference")
            enc_ref = res.get("encounter", {}).get("reference")

            code_cc = res.get("code")
            clinical_status_cc = res.get("clinicalStatus")
            verification_status_cc = res.get("verificationStatus")

            cond_rows.append(
                {
                    "condition_id": condition_id,
                    "patient_id": ref_to_uuid(subj_ref),
                    "encounter_id": ref_to_uuid(enc_ref),
                    "code_system": cc_attr(code_cc, "system"),
                    "code": cc_attr(code_cc, "code"),
                    "code_display": cc_attr(code_cc, "display"),
                    "clinical_status": cc_attr(clinical_status_cc, "code"),
                    "verification_status": cc_attr(verification_status_cc, "code"),
                    "onset_datetime": res.get("onsetDateTime"),
                    "recorded_datetime": res.get("recordedDate"),
                }
            )

        # ------------- OBSERVATION -------------
        elif rtype == "Observation":
            observation_id = res.get("id")
            subj_ref = res.get("subject", {}).get("reference")
            enc_ref = res.get("encounter", {}).get("reference")
            status = res.get("status")

            category_cc = res.get("category")
            code_cc = res.get("code")

            # valueQuantity may have shape:
            # { "value": ..., "unit": ... } in Synthea FHIR
            vq = res.get("valueQuantity") or {}
            value_quantity = vq.get("value")
            value_unit = vq.get("unit")

            # For "valueString" / others, we keep a generic textual representation
            value_string = None
            if res.get("valueString") is not None:
                value_string = str(res["valueString"])
            elif res.get("valueInteger") is not None:
                value_string = str(res["valueInteger"])
            elif res.get("valueBoolean") is not None:
                value_string = str(res["valueBoolean"])
            elif res.get("valueDateTime") is not None:
                value_string = str(res["valueDateTime"])
            # (you can expand this if needed, but it's not used for lab predictors)

            obs_rows.append(
                {
                    "observation_id": observation_id,
                    "patient_id": ref_to_uuid(subj_ref),
                    "encounter_id": ref_to_uuid(enc_ref),
                    "status": status,
                    "category": cc_text(category_cc) or cc_attr(category_cc, "display"),
                    "code_system": cc_attr(code_cc, "system"),
                    "code": cc_attr(code_cc, "code"),
                    "code_display": cc_attr(code_cc, "display"),
                    "value_quantity": value_quantity,
                    "value_unit": value_unit,
                    "value_string": value_string,
                    "effective_datetime": res.get("effectiveDateTime"),
                }
            )

        # you can add Encounter parsing here if you ever need it

    # -----------------------------
    # Build DataFrames
    # -----------------------------
    df_patients = pd.DataFrame(patients_rows).drop_duplicates(subset=["patient_id"]) if patients_rows else pd.DataFrame(
        columns=["patient_id", "gender", "birth_date"]
    )

    df_obs = pd.DataFrame(obs_rows).drop_duplicates(subset=["observation_id"]) if obs_rows else pd.DataFrame(
        columns=[
            "observation_id",
            "patient_id",
            "encounter_id",
            "status",
            "category",
            "code_system",
            "code",
            "code_display",
            "value_quantity",
            "value_unit",
            "value_string",
            "effective_datetime",
        ]
    )

    df_conditions = pd.DataFrame(cond_rows).drop_duplicates(subset=["condition_id"]) if cond_rows else pd.DataFrame(
        columns=[
            "condition_id",
            "patient_id",
            "encounter_id",
            "code_system",
            "code",
            "code_display",
            "clinical_status",
            "verification_status",
            "onset_datetime",
            "recorded_datetime",
        ]
    )

    return df_patients, df_obs, df_conditions
