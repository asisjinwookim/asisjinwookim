# [1] Initial Setup: Create Project Directory and Conda Environment
## 1. Project Directory Creation
mkdir mlops-versioning-example
cd mlops-versioning-example
echo "✅ Project directory 'mlops-versioning-example' created and entered."

## 2. Create environment.yaml for Conda
cat << EOF > environment.yaml
name: mlops-versioning-example
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - pandas
  - numpy
  - scikit-learn
  - dvc
  - mlflow
  - flask
  - pyyaml
EOF
echo "✅ environment.yaml created."

## 3. Create Conda Environment
echo "🔄 Creating Conda environment 'mlops-versioning-example'. This might take a few minutes..."
conda env create -f environment.yaml
echo "✅ Conda environment created."

## 4. Activate Conda Environment
echo "🔄 Activating Conda environment..."
conda activate mlops-versioning-example
echo "✅ Conda environment 'mlops-versioning-example' activated."
conda list > conda_packages.txt # Optional: Save package list for reference
echo "Conda environment packages listed in conda_packages.txt"

# [2] Create Core MLOps Files (Data, Scripts, Configs)
## 1. Create 'data' and 'src' directories
mkdir data src templates model
echo "✅ Directories 'data', 'src', 'templates', 'model' created."

## 2. Create data/raw_data.csv (Initial Version)
python -c "
import pandas as pd
import numpy as np
import os
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X + 1 + np.random.randn(100, 1) * 2
df = pd.DataFrame({'feature': X.flatten(), 'target': y.flatten()})
df.to_csv('data/raw_data.csv', index=False)
print('✅ data/raw_data.csv (초기 버전) 파일이 생성되었습니다.')
"

## 3. Create src/train.py
cat << 'EOF' > src/train.py
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
EOF
echo "✅ src/train.py created."

## 4. Create src/evaluate.py
cat << 'EOF' > src/evaluate.py
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
EOF
echo "✅ src/evaluate.py created."

## 5. Create params.yaml
cat << EOF > params.yaml
train:
  alpha: 0.0
EOF
echo "✅ params.yaml created."

## 6. Create .gitignore
cat << EOF > .gitignore
# .gitignore
# DVC
.dvc/cache/
data/*.csv
data/*.tsv
data/*.json

# Model artifacts
model/
*.pkl
*.bin

# MLflow
mlruns/

# Python
__pycache__/
.ipynb_checkpoints/
*.pyc
.env
venv/
EOF
echo "✅ .gitignore created."


# 3. Initialize Git & DVC, and Define Pipeline

## Ensure Conda environment is activated: (mlops-versioning-example)
## conda activate mlops-versioning-example

## 1. Git Initialization
git init
echo "✅ Git initialized."

## 2. Add .gitignore to Git
git add .gitignore
git commit -m "feat: Add .gitignore to exclude data, model, and MLflow logs"
echo "✅ .gitignore file committed."

## 3. DVC Initialization (uses local cache by default)
dvc init
echo "✅ DVC initialized with local cache."

## 4. Add data to DVC and commit its metafile to Git
dvc add data/raw_data.csv
echo "✅ data/raw_data.csv tracked by DVC. data/raw_data.csv.dvc file created."

git add data/raw_data.csv.dvc
git commit -m "feat: Track raw_data.csv with DVC (initial data version)"
echo "✅ Git: DVC metafile committed."

## 5. Manually define the DVC pipeline (dvc.yaml)
## THIS IS THE MODERN DVC WAY - no dvc run --no-exec
cat << 'EOF' > dvc.yaml
stages:
  train:
    cmd: python src/train.py
    deps:
    - src/train.py
    - data/raw_data.csv
    - params.yaml
    outs:
    - model/linear_regression_model.pkl
  evaluate:
    cmd: python src/evaluate.py
    deps:
    - src/evaluate.py
    - model/linear_regression_model.pkl
    - data/raw_data.csv
EOF
echo "✅ dvc.yaml (DVC pipeline definition) created."
echo "--- dvc.yaml content ---"
cat dvc.yaml
echo "----------------------"


## 6. Commit pipeline definition and source code to Git
git add dvc.yaml src/train.py src/evaluate.py params.yaml
git commit -m "feat: Define DVC pipeline and initial model training/evaluation scripts (v1)"
echo "✅ Git: DVC pipeline definition and initial code committed."

## 7. Run the DVC pipeline for the first time
echo "🚀 Running the DVC pipeline for the first time (model training and evaluation)..."
dvc repro
echo "✅ Pipeline run complete. Results logged to MLflow."


# 4. Setup Flask Web UI
## 1. Create app.py
cat << 'EOF' > app.py
import subprocess
import os
import json
from flask import Flask, render_template, request, redirect, url_for, flash
import yaml

app = Flask(__name__)
app.secret_key = 'your_super_secret_key_for_flash_messages'

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def run_command(command, cwd=PROJECT_ROOT):
    print(f"Executing command: {command}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True, cwd=cwd)
        if result.stderr:
            print(f"Command STDERR: {result.stderr}")
        return result.stdout.strip(), result.stderr.strip()
    except subprocess.CalledProcessError as e:
        error_msg = f"Error executing command '{e.cmd}':\nStdout: {e.stdout}\nStderr: {e.stderr}"
        print(error_msg)
        raise RuntimeError(error_msg)

def get_git_log():
    try:
        stdout, _ = run_command("git log --oneline --decorate=short")
        logs = []
        for line in stdout.splitlines():
            if line:
                parts = line.split(" ", 1)
                logs.append({"hash": parts[0], "message": parts[1] if len(parts) > 1 else ""})
        return logs
    except RuntimeError as e:
        print(f"Failed to get Git log: {e}")
        return []

def get_current_status():
    status = {
        "git_head": "N/A",
        "git_message": "N/A",
        "dvc_data_status": {},
        "params_content": "File not found or unreadable."
    }
    try:
        git_head, _ = run_command("git rev-parse HEAD")
        status["git_head"] = git_head

        git_msg, _ = run_command(f"git log -1 --pretty=%B {git_head}")
        status["git_message"] = git_msg.strip()

        dvc_file_path = os.path.join('data', 'raw_data.csv.dvc')
        if os.path.exists(dvc_file_path):
            with open(dvc_file_path, 'r') as f:
                dvc_meta = yaml.safe_load(f)
                if 'outs' in dvc_meta and dvc_meta['outs']:
                    dvc_data_hash = dvc_meta['outs'][0].get('md5', dvc_meta['outs'][0].get('hash', 'N/A'))
                    status["dvc_data_status"] = {"raw_data.csv": dvc_data_hash}
        else:
            status["dvc_data_status"] = {"raw_data.csv": "Not DVC tracked or file missing."}

        try:
            with open('params.yaml', 'r') as f:
                status["params_content"] = f.read()
        except FileNotFoundError:
            status["params_content"] = "params.yaml not found."

    except Exception as e:
        print(f"Error getting current status: {e}")
        flash(f"상태 정보를 가져오는 중 오류 발생: {e}", "error")
    return status

@app.route('/')
def index():
    status = get_current_status()
    git_logs = get_git_log()
    return render_template('index.html', status=status, git_logs=git_logs)

@app.route('/checkout', methods=['POST'])
def checkout():
    commit_hash = request.form['commit_hash']
    print(f"\n--- {commit_hash} 버전으로 전환 시도 ---")

    try:
        flash(f"Git 커밋 {commit_hash}으로 체크아웃 중...", "info")
        stdout, stderr = run_command(f"git checkout {commit_hash}")
        print(f"Git Checkout Output: {stdout}")
        if stderr: flash(f"Git Checkout 경고: {stderr}", "warning")

        flash("DVC 데이터를 복원 중...", "info")
        stdout, stderr = run_command("dvc checkout")
        print(f"DVC Checkout Output: {stdout}")
        if stderr: flash(f"DVC Checkout 경고: {stderr}", "warning")

        flash("DVC 파이프라인 재실행 중...", "info")
        stdout, stderr = run_command("dvc repro")
        print(f"DVC Repro Output: {stdout}")
        if stderr: flash(f"DVC Repro 경고: {stderr}", "warning")

        flash(f"성공적으로 {commit_hash} 버전으로 전환되었고 파이프라인이 재실행되었습니다.", "success")
        print(f"--- {commit_hash} 버전 전환 완료 ---")

    except RuntimeError as e:
        flash(f"버전 전환 중 오류 발생: {e}", "error")
        print(f"--- {commit_hash} 버전 전환 실패 ---")

    return redirect(url_for('index'))

@app.route('/start_mlflow_ui')
def start_mlflow_ui():
    flash("MLflow UI를 시작하려면 터미널에서 'mlflow ui'를 실행하세요. 잠시 후 http://127.0.0.1:5000으로 접속 가능합니다.", "info")
    return redirect(url_for('index'))

if __name__ == '__main__':
    print("🌍 Flask 애플리케이션이 http://localhost:5001 에서 실행됩니다.")
    print("📊 MLflow UI는 별도 터미널에서 'mlflow ui' 명령어로 실행해야 합니다 (http://127.0.0.1:5000).")
    app.run(debug=True, port=5001)
EOF
echo "✅ app.py created."

## 2. Create templates/index.html
cat << 'EOF' > templates/index.html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MLOps 로컬 버전 관리자</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #e9eff4; color: #333; line-height: 1.6; }
        .container { max-width: 960px; margin: auto; background: #ffffff; padding: 30px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
        h1, h2, h3 { color: #2c3e50; margin-bottom: 15px; }
        hr { border: 0; height: 1px; background: #ddd; margin: 30px 0; }
        pre { background: #f8f8f8; padding: 15px; border-radius: 6px; overflow-x: auto; font-family: 'Cascadia Code', 'Fira Code', monospace; font-size: 0.9em; border: 1px solid #eee; }
        .current-status, .version-history { margin-bottom: 30px; padding-bottom: 20px; }
        .status-item { margin-bottom: 10px; }
        .status-item strong { color: #34495e; }
        .version-item { background: #f0f4f7; border: 1px solid #dce4e8; padding: 12px 15px; margin-bottom: 12px; border-radius: 6px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
        .version-item:hover { background-color: #e5edf2; }
        .version-item div { flex-grow: 1; margin-right: 15px; }
        .version-item code { background-color: #e9ecef; padding: 3px 6px; border-radius: 3px; font-family: monospace; font-size: 0.85em; }
        .version-item button { background-color: #007bff; color: white; padding: 9px 15px; border: none; border-radius: 5px; cursor: pointer; font-size: 0.9em; transition: background-color 0.2s ease; }
        .version-item button:hover { background-color: #0056b3; }
        .mlflow-link { text-align: center; margin-top: 30px; }
        .mlflow-link a { background-color: #28a745; color: white; padding: 12px 20px; border-radius: 5px; text-decoration: none; font-size: 1.1em; transition: background-color 0.2s ease; }
        .mlflow-link a:hover { background-color: #218838; }

        .flash-message {
            padding: 10px 15px;
            margin-bottom: 20px;
            border-radius: 5px;
            font-weight: bold;
            display: flex;
            align-items: center;
        }
        .flash-message.success { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .flash-message.error { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .flash-message.info { background-color: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        .flash-message.warning { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
    </style>
</head>
<body>
    <div class="container">
        <h1>MLOps 로컬 버전 관리자</h1>
        <p>Git으로 코드를, DVC로 데이터를, MLflow로 실험을 관리합니다.</p>

        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="flash-message {{ category }}">{{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}

        <hr>

        <div class="current-status">
            <h2>✨ 현재 작업 공간 상태</h2>
            <div class="status-item">
                <strong>Git HEAD:</strong> <code>{{ status.git_head }}</code>
            </div>
            <div class="status-item">
                <strong>커밋 메시지:</strong> <em>{{ status.git_message }}</em>
            </div>
            <div class="status-item">
                <h3>현재 DVC 추적 데이터 버전:</h3>
                <pre>{{ status.dvc_data_status | tojson(indent=2) }}</pre>
            </div>
            <div class="status-item">
                <h3>현재 params.yaml 내용:</h3>
                <pre>{{ status.params_content }}</pre>
            </div>
        </div>

        <hr>

        <div class="version-history">
            <h2>📚 버전 기록 (Git 커밋)</h2>
            <p>아래 목록에서 원하는 커밋을 선택하여 해당 시점의 코드와 데이터로 전환할 수 있습니다.</p>
            {% for log in git_logs %}
                <div class="version-item">
                    <div>
                        <strong><code>{{ log.hash }}</code></strong>: {{ log.message }}
                    </div>
                    <form action="/checkout" method="post" style="display:inline;">
                        <input type="hidden" name="commit_hash" value="{{ log.hash }}">
                        <button type="submit">이 버전으로 전환</button>
                    </form>
                </div>
            {% endfor %}
        </div>

        <hr>

        <div class="mlflow-link">
            <p>실험 지표, 파라미터, 모델 아티팩트는 MLflow UI에서 상세히 확인하세요.</p>
            <a href="http://127.0.0.1:5000" target="_blank" onclick="alert('MLflow UI는 별도 터미널에서 \'mlflow ui\' 명령어로 먼저 실행되어야 합니다.');">MLflow UI 열기</a>
        </div>
    </div>
</body>
</html>
EOF
echo "✅ templates/index.html created."

## Commit Flask app files to Git
git add app.py templates/index.html
git commit -m "feat: Add Flask web UI for version management"
echo "✅ Git: Flask web UI files committed."

# [5] Run MLOps Applications

## Terminal 1: Start MLflow UI
```
conda activate mlops-versioning-example
mlflow ui
```
(Open your web browser and navigate to http://127.0.0.1:5000)


## Terminal 2: Start Flask Web UI
```
conda activate mlops-versioning-example
python app.py
```
(Open your web browser and navigate to http://localhost:5001)

