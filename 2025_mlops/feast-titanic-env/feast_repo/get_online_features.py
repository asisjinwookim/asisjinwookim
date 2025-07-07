import pandas as pd
from feast import FeatureStore
import os

fs = FeatureStore(repo_path="feature_repo")

entity_rows = pd.DataFrame({"passenger_id": [1, 5, 10, 15]})

print("\n--- Getting Online Features ---")
features_to_get = [
    "passenger_features:Pclass",
    "passenger_features:Sex",
    "passenger_features:Age",
    "passenger_features:SibSp",
    "passenger_features:Parch",
    "passenger_features:Fare",
    "passenger_features:Embarked",
    "passenger_features:Survived",  # Survived도 온라인에서 가져올 수 있습니다.
    "passenger_features:age_fare_ratio",  # 경로 변경: passenger_features의 일반 필드로
]

online_features = fs.get_online_features(
    features=features_to_get,
    entity_rows=entity_rows,
).to_df()

print("\nOnline Features Retrieved (head):")
print(online_features)
print(f"\nTotal online features retrieved: {len(online_features)} rows.")
print("\nFeast online feature retrieval complete.")
