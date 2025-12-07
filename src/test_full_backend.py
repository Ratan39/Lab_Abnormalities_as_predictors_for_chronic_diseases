import json
from pathlib import Path

from src.preprocessing.parse_json import parse_patient_bundle
from src.preprocessing.build_features import build_feature_table_for_bundle
from src.preprocessing.clustering_transform import add_cluster_and_align_for_models


def main():
    json_path = Path("/Users/sai/Desktop/Aaron697_Batz141_f339516d-a3de-8d31-72fe-b409d2f99126.json")

    with open(json_path, "r") as f:
        bundle = json.load(f)

    # 1) Parse raw bundle
    df_patients, df_obs, df_conditions = parse_patient_bundle(bundle)
    print("Patients shape:", df_patients.shape)
    print("Observations shape:", df_obs.shape)

    # 2) Build feature table (age, sex, labs...)
    feature_table = build_feature_table_for_bundle(df_patients, df_obs, reference_date="2025-01-01")
    print("\n=== Feature table ===")
    print(feature_table.head())

    # 3) Add cluster + align for models
    ft_with_cluster, X_ready = add_cluster_and_align_for_models(feature_table)

    print("\n=== Feature table with cluster ===")
    print(ft_with_cluster.head())

    print("\n=== X_ready for models ===")
    print(X_ready.head())
    print("X_ready columns:", X_ready.columns.tolist())
    print("Number of features:", X_ready.shape[1])


if __name__ == "__main__":
    main()
