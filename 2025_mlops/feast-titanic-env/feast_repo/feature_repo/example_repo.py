from datetime import timedelta, datetime
import pandas as pd
import numpy as np

from feast import FeatureView, Field, Entity, ValueType, FileSource

# from feast.on_demand_feature_view import on_demand_feature_view # 이 임포트 제거
from feast.types import Int64, String, Float32

passenger_entity = Entity(
    name="passenger_id",
    description="Titanic Passenger ID",
    value_type=ValueType.INT64,
)

import os

data_dir = "./data"
kaggle_train_csv_path = os.path.join(data_dir, "train.csv")

try:
    titanic_df = pd.read_csv(kaggle_train_csv_path)
    print(f"Kaggle Titanic train data loaded from: {kaggle_train_csv_path}")
except FileNotFoundError:
    print(
        f"Error: {kaggle_train_csv_path} not found. Please download train.csv from Kaggle and place it in the '{data_dir}' directory."
    )
    raise

titanic_df.rename(columns={"PassengerId": "passenger_id"}, inplace=True)

# Feast를 위한 'event_timestamp' 컬럼 추가
titanic_df["event_timestamp"] = datetime(2023, 1, 1, 0, 0, 0)

# 결측치 처리 (train.csv에 맞춰 Age는 평균, Embarked는 최빈값, Fare는 평균으로 채움)
titanic_df["Age"].fillna(titanic_df["Age"].mean(), inplace=True)
titanic_df["Embarked"].fillna(titanic_df["Embarked"].mode()[0], inplace=True)
titanic_df["Fare"].fillna(titanic_df["Fare"].mean(), inplace=True)

# 데이터 타입 명시적 변환
titanic_df["Pclass"] = titanic_df["Pclass"].astype(np.int64)
titanic_df["Age"] = titanic_df["Age"].astype(np.float32)
titanic_df["SibSp"] = titanic_df["SibSp"].astype(np.int64)
titanic_df["Parch"] = titanic_df["Parch"].astype(np.int64)
titanic_df["Fare"] = titanic_df["Fare"].astype(np.float32)
titanic_df["Survived"] = titanic_df["Survived"].astype(np.int64)


# 새로운 복합 피처 'age_fare_ratio'를 여기서 직접 계산
# ODFV 로직과 동일하게 벡터화된 연산을 사용
valid_age_condition = titanic_df["Age"].notna() & (titanic_df["Age"] > 0)
titanic_df["age_fare_ratio"] = (titanic_df["Fare"] / titanic_df["Age"]).where(
    valid_age_condition, 0.0
)
titanic_df["age_fare_ratio"] = titanic_df["age_fare_ratio"].astype(
    np.float32
)  # 최종 타입 명시


titanic_file_source = FileSource(
    path=kaggle_train_csv_path,  # 이 경로의 train.csv가 이제 age_fare_ratio 컬럼을 포함해야 함
    timestamp_field="event_timestamp",
)

# FeatureView 정의: age_fare_ratio를 일반 필드로 추가
passenger_features_fv = FeatureView(
    name="passenger_features",
    entities=[passenger_entity],
    ttl=timedelta(days=3650),
    source=titanic_file_source,
    schema=[
        Field(name="Pclass", dtype=Int64),
        Field(name="Sex", dtype=String),
        Field(name="Age", dtype=Float32),
        Field(name="SibSp", dtype=Int64),
        Field(name="Parch", dtype=Int64),
        Field(name="Fare", dtype=Float32),
        Field(name="Embarked", dtype=String),
        Field(name="Survived", dtype=Int64),
        Field(
            name="age_fare_ratio", dtype=Float32
        ),  # age_fare_ratio를 일반 필드로 추가
    ],
)
