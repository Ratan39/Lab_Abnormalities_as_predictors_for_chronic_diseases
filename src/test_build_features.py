import json
from pathlib import Path

from src.preprocessing.parse_json import parse_patient_bundle
from src.preprocessing.build_features import build_feature_table_for_bundle


def main():
    json_path = Path("/Users/sai/Desktop/Aaron697_Batz141_f339516d-a3de-8d31-72fe-b409d2f99126.json")

    with open(json_path, "r") as f:
        bundle = json.load(f)

    df_patients, df_obs, df_conditions = parse_patient_bundle(bundle)

    print("Patients shape:", df_patients.shape)
    print("Observations shape:", df_obs.shape)

    feature_table = build_feature_table_for_bundle(df_patients, df_obs, reference_date="2025-01-01")

    print("\n=== Feature table ===")
    print(feature_table.head())
    print("Columns:\n", feature_table.columns.tolist())

    # For your app, you will usually do:
    # feature_row = feature_table.iloc[[0]]  # keep as DataFrame with 1 row


if __name__ == "__main__":
    main()
