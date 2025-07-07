import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os
import numpy as np

# 학습 데이터 직접 로드 (Feast 사용 안 함)
data_dir = "feature_repo/data"
kaggle_train_csv_path = os.path.join(data_dir, "train.csv")

try:
    training_df = pd.read_csv(kaggle_train_csv_path)
    print(
        f"Kaggle Titanic train data loaded directly for training: {kaggle_train_csv_path}"
    )
except FileNotFoundError:
    print(
        f"Error: {kaggle_train_csv_path} not found. Please download train.csv from Kaggle and place it in the '{data_dir}' directory."
    )
    exit(1)

# 데이터 전처리 (example_repo.py의 전처리 로직과 일관되게)
training_df.rename(columns={"PassengerId": "passenger_id"}, inplace=True)
training_df["Age"].fillna(training_df["Age"].mean(), inplace=True)
training_df["Embarked"].fillna(training_df["Embarked"].mode()[0], inplace=True)
training_df["Fare"].fillna(training_df["Fare"].mean(), inplace=True)

# 온-디맨드 피처 'age_fare_ratio' 직접 계산 (Feast에서 가져오는 대신)
# 이 로직은 example_repo.py의 @on_demand_feature_view 로직과 동일해야 합니다.
training_df["age_fare_ratio"] = training_df.apply(
    lambda row: (
        row["Fare"] / row["Age"] if pd.notna(row["Age"]) and row["Age"] > 0 else 0.0
    ),
    axis=1,
).astype(np.float32)

# 문자열(Categorical) 피처를 원-핫 인코딩
training_df = pd.get_dummies(training_df, columns=["Sex", "Embarked"], drop_first=True)

# X (피처)와 y (라벨) 분리
# Survived가 라벨이므로 X에서 제외합니다.
X = training_df.drop(
    columns=["Survived", "Name", "Ticket", "Cabin"]
)  # 모델 학습에 필요 없는 컬럼도 제거
y = training_df["Survived"]

# PassengerId는 엔티티 키이므로, 학습 피처에서는 제거하거나 인덱스로 설정
X.set_index("passenger_id", inplace=True)

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

# 학습/검증 세트 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")

# 모델 학습 (Logistic Regression)
print("\n--- Training Logistic Regression Model ---")
model = LogisticRegression(
    solver="liblinear", random_state=42, max_iter=200
)  # max_iter 추가
model.fit(X_train, y_train)

# 모델 평가
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"\nModel Accuracy on validation set: {accuracy:.4f}")
print("\nClassification Report on validation set:")
print(classification_report(y_val, y_pred))

print("\nModel training complete directly from CSV.")

# 학습된 모델을 저장
import joblib

model_path = os.path.join(data_dir, "titanic_logistic_model.joblib")
joblib.dump(model, model_path)
print(f"Trained model saved to: {model_path}")
