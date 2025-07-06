import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import mlflow
import yaml
import os

def load_params():
    params = {}
    try:
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load(f)
    except FileNotFoundError:
        print("경고: params.yaml 파일을 찾을 수 없습니다. 기본값을 사용합니다.")
        params = {'train': {'alpha': 0.0}}
    return params.get('train', {})

if __name__ == "__main__":
    params = load_params()
    alpha = params.get('alpha', 0.0)
    print(f"📈 모델 학습 시작 (alpha={alpha})")

    with mlflow.start_run(run_name=f"Model_Training_alpha_{alpha}"):
        mlflow.log_param("alpha", alpha)

        df = pd.read_csv('data/raw_data.csv')
        X = df[['feature']]
        y = df['target']

        model = LinearRegression()
        model.fit(X, y)

        model_dir = "model"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "linear_regression_model.pkl")
        joblib.dump(model, model_path)
        print(f"✅ 모델이 {model_path}에 저장되었습니다.")

        mlflow.log_artifact(model_path)
        print("✅ MLflow: 모델 아티팩트가 기록되었습니다.")
