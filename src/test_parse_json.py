import json
from pathlib import Path

from src.preprocessing.parse_json import parse_patient_bundle


def main():
    # 1. Point this to one of your raw JSON bundle files
    json_path = Path("/Users/sai/Desktop/Aaron697_Batz141_f339516d-a3de-8d31-72fe-b409d2f99126.json")

    print(f"Loading JSON from: {json_path}")
    with open(json_path, "r") as f:
        bundle = json.load(f)

    # 2. Parse into DataFrames
    df_patients, df_obs, df_conditions = parse_patient_bundle(bundle)

    print("\n=== Patients ===")
    print(df_patients.head())
    print("Shape:", df_patients.shape)

    print("\n=== Observations ===")
    print(df_obs[["patient_id", "code_display", "value_quantity", "effective_datetime"]].head())
    print("Shape:", df_obs.shape)

    print("\n=== Conditions ===")
    print(df_conditions[["patient_id", "code_display", "onset_datetime"]].head())
    print("Shape:", df_conditions.shape)


if __name__ == "__main__":
    main()
