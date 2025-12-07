import json
from pathlib import Path

from src.preprocessing.parse_json import parse_patient_bundle
from src.preprocessing.build_features import build_feature_table_for_bundle
from src.preprocessing.clustering_transform import add_cluster_and_align_for_models
from src.models.inference import load_all_models, predict_all_diseases


def main():
    # 1. Load one of your patient JSON bundles
    json_path = Path("/Users/sai/Desktop/Aaron697_Batz141_f339516d-a3de-8d31-72fe-b409d2f99126.json")

    with open(json_path, "r") as f:
        bundle = json.load(f)

    # 2. Parse JSON → dfs
    df_patients, df_obs, df_conditions = parse_patient_bundle(bundle)
    print("Patients shape:", df_patients.shape)
    print("Observations shape:", df_obs.shape)

    # 3. Build feature table (1 row for this patient)
    feature_table = build_feature_table_for_bundle(df_patients, df_obs, reference_date="2025-01-01")
    print("\n=== Feature table ===")
    print(feature_table.head())

    # 4. Add cluster + align for models
    ft_with_cluster, X_ready = add_cluster_and_align_for_models(feature_table)
    print("\n=== Feature table with cluster ===")
    print(ft_with_cluster.head())

    print("\n=== X_ready (model input) ===")
    print(X_ready.head())
    print("Num features:", X_ready.shape[1])

    # 5. Load models
    models = load_all_models()

    # 6. Predict all 4 diseases
    results = predict_all_diseases(X_ready, models, threshold=0.5)

    print("\n=== Prediction results ===")
    for disease, info in results.items():
        print(
            f"{disease}: prob={info['prob']:.3f}, "
            f"label={'Positive' if info['label'] == 1 else 'Negative'}"
        )


if __name__ == "__main__":
    main()
