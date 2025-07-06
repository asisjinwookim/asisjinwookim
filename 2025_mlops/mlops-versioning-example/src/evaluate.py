import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import os

if __name__ == "__main__":
    print("📊 모델 평가 시작")
    with mlflow.start_run(nested=True, run_name="Model_Evaluation"):
        df = pd.read_csv('data/raw_data.csv')
        X = df[['feature']]
        y_true = df['target']

        model_path = os.path.join("model", "linear_regression_model.pkl")
        if not os.path.exists(model_path):
            print(f"오류: 모델 파일 {model_path}을(를) 찾을 수 없습니다. 학습이 먼저 진행되어야 합니다.")
            exit(1)

        model = joblib.load(model_path)
        y_pred = model.predict(X)

        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)

        print(f"✅ MSE: {mse:.2f}")
        print(f"✅ R2 Score: {r2:.2f}")
        print("✅ MLflow: 지표가 기록되었습니다.")
