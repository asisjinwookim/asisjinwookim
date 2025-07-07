import pandas as pd
import os
from feast import FeatureStore
import numpy as np
import joblib

fs = FeatureStore(repo_path="feature_repo")

data_dir = "data"
test_csv_path = os.path.join(data_dir, "test.csv")
model_path = os.path.join(data_dir, "titanic_logistic_model.joblib")

try:
    test_df_raw = pd.read_csv(test_csv_path)
    print(f"Kaggle test.csv loaded from: {test_csv_path}")
except FileNotFoundError:
    print(
        f"Error: {test_csv_path} not found. Please download test.csv from Kaggle and place it in the '{data_dir}' directory."
    )
    exit(1)

entity_rows_for_test = pd.DataFrame({"passenger_id": test_df_raw["PassengerId"]})

print("\n--- Getting Features for Prediction (from test.csv PassengerIds) ---")

features_to_get_for_prediction = [
    "passenger_features:Pclass",
    "passenger_features:Sex",
    "passenger_features:Age",
    "passenger_features:SibSp",
    "passenger_features:Parch",
    "passenger_features:Fare",
    "passenger_features:Embarked",
    "passenger_features:age_fare_ratio",  # 경로 변경
]

prediction_features_df = fs.get_online_features(
    features=features_to_get_for_prediction,
    entity_rows=entity_rows_for_test,
).to_df()

print("\nPrediction Features Retrieved (head):")
print(prediction_features_df.head())
print(f"\nTotal prediction features retrieved: {len(prediction_features_df)} rows.")

# 모델 예측을 위한 데이터 전처리 (get_training_data.py와 동일하게 적용)
prediction_features_df = pd.get_dummies(
    prediction_features_df, columns=["Sex", "Embarked"], drop_first=True
)
prediction_features_df.set_index("passenger_id", inplace=True)

for col in [
    "Age",
    "Fare",
    "Pclass",
    "SibSp",
    "Parch",
    "Sex_male",
    "Embarked_Q",
    "Embarked_S",
    "age_fare_ratio",
]:
    if (
        col in prediction_features_df.columns
        and prediction_features_df[col].isnull().any()
    ):
        print(f"Warning: NaN found in {col} in prediction features. Filling with 0.")
        prediction_features_df[col].fillna(0, inplace=True)

try:
    model = joblib.load(model_path)
    print(f"\nTrained model loaded from: {model_path}")

    model_features = model.feature_names_in_.tolist()

    final_prediction_X = pd.DataFrame(columns=model_features)
    for col in model_features:
        if col in prediction_features_df.columns:
            final_prediction_X[col] = prediction_features_df[col]
        else:
            final_prediction_X[col] = 0

    final_prediction_X = final_prediction_X[model_features]

    predictions = model.predict(final_prediction_X)
    prediction_proba = model.predict_proba(final_prediction_X)[:, 1]

    results_df = pd.DataFrame(
        {
            "PassengerId": test_df_raw["PassengerId"],
            "Survived_Prediction": predictions,
            "Survived_Proba": prediction_proba,
        }
    )

    print("\nPrediction Results (head):")
    print(results_df.head())

    kaggle_submission_path = os.path.join(data_dir, "submission.csv")
    results_df[["PassengerId", "Survived_Prediction"]].rename(
        columns={"Survived_Prediction": "Survived"}
    ).to_csv(kaggle_submission_path, index=False)
    print(f"\nKaggle submission file saved to: {kaggle_submission_path}")

except FileNotFoundError:
    print(
        f"Error: Trained model not found at {model_path}. Please run get_training_data.py first."
    )
except Exception as e:
    print(f"An error occurred during prediction: {e}")

print("\nFeast prediction feature retrieval and model inference complete.")
