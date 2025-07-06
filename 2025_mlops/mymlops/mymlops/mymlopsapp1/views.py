# mymlopsapp1/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse
from django.conf import settings
from django import forms  # Import forms
from django.forms import (
    MultipleChoiceField,
    CheckboxSelectMultiple,
)  # 결과 선택을 위한 폼 필드 임포트

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
)
import joblib

import json  # json 모듈 임포트

# 폼 임포트
from .forms import DatasetConfigForm, FeatureConfigForm, ModelConfigForm
from .models import DatasetConfig, FeatureConfig, ModelConfig

# 데이터셋 루트 경로 (settings.py의 BASE_DIR 기준)
DATA_ROOT = os.path.join(settings.BASE_DIR, "data")
# 모델 저장 경로 (settings.MEDIA_ROOT 사용)
MODEL_DIR = os.path.join(settings.MEDIA_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)  # 모델 디렉토리 생성


# --- 데이터셋 업로드 및 설정 폼 (단일 단계) ---
class DatasetUploadAndConfigForm(forms.Form):
    file = forms.FileField(
        label="CSV 파일 선택", help_text="업로드할 데이터셋 CSV 파일을 선택하세요."
    )
    dataset_name = forms.CharField(
        max_length=100,
        label="데이터셋 이름",
        help_text="데이터셋의 고유한 이름을 입력하세요 (예: WineQualityRed_New).",
    )
    # target_column = forms.CharField(
    #     max_length=100,
    #     label="타겟 컬럼 이름",
    #     help_text="예측하거나 분류할 타겟 변수의 정확한 컬럼 이름을 입력하세요.",
    # )
    description = forms.CharField(
        widget=forms.Textarea(attrs={"rows": 3}),
        required=False,
        label="설명",
        help_text="데이터셋에 대한 간단한 설명을 입력하세요.",
    )


# --- 임시 파일 업로드 폼 (초기 파일 업로드) ---
class InitialUploadFileForm(forms.Form):
    file = forms.FileField(
        label="CSV 파일 선택", help_text="업로드할 데이터셋 CSV 파일을 선택하세요."
    )
    dataset_name = forms.CharField(
        max_length=100,
        label="데이터셋 이름",
        help_text="데이터셋의 고유한 이름을 입력하세요 (예: WineQualityRed_New).",
    )
    description = forms.CharField(
        widget=forms.Textarea(attrs={"rows": 3}),
        required=False,
        label="설명",
        help_text="데이터셋에 대한 간단한 설명을 입력하세요.",
    )


# --- 타겟 컬럼 선택 폼 (컬럼 목록 동적 생성) ---
class SelectTargetColumnForm(forms.Form):
    # 이 필드는 뷰에서 동적으로 choices를 채울 것입니다.
    target_column = forms.ChoiceField(
        label="타겟 컬럼 선택",
        help_text="예측하거나 분류할 타겟 변수 컬럼을 선택하세요.",
    )
    # dataset_name, description은 hidden 필드로 전달받아 DatasetConfig 생성에 사용
    dataset_name = forms.CharField(widget=forms.HiddenInput())
    uploaded_filename = forms.CharField(widget=forms.HiddenInput())
    description = forms.CharField(widget=forms.HiddenInput(), required=False)


# --- File Upload Form (Define this directly in views.py or in forms.py) ---
# For simplicity, we'll define it here. For more complex apps, put in forms.py
class UploadFileForm(forms.Form):
    # This field will be used to upload the actual dataset file
    file = forms.FileField(label="Select a CSV file to upload")
    # This field will be used to create a DatasetConfig record for the uploaded file
    dataset_name = forms.CharField(
        max_length=100, label="Dataset Name (e.g., MyNewData)"
    )
    target_column = forms.CharField(
        max_length=50, label="Target Column Name (e.g., price)"
    )
    description = forms.CharField(
        widget=forms.Textarea(attrs={"rows": 3}), required=False
    )


# --- 홈 페이지 ---
def home(request):
    return render(
        request, "mymlopsapp1/home.html", {"message": "Welcome to myMLOpsApp1!"}
    )


# --- DatasetConfig 생성 뷰 ---
def create_dataset_config_view(request):
    if request.method == "POST":
        form = DatasetConfigForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect(
                "mymlopsapp1:list_dataset_configs"
            )  # 목록 페이지로 리다이렉트
    else:
        form = DatasetConfigForm()
    return render(
        request,
        "mymlopsapp1/create_config.html",
        {"form": form, "config_type": "Dataset"},
    )


def list_dataset_configs_view(request):
    configs = DatasetConfig.objects.all()
    return render(
        request,
        "mymlopsapp1/list_configs.html",
        {
            "configs": configs,
            "config_type": "Dataset",
            "create_url_name": "mymlopsapp1:create_dataset_config",
        },
    )


# --- FeatureConfig 생성 뷰 ---
def create_feature_config_view(request):
    if request.method == "POST":
        form = FeatureConfigForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect("mymlopsapp1:list_feature_configs")
    else:
        form = FeatureConfigForm()
    return render(
        request,
        "mymlopsapp1/create_config.html",
        {"form": form, "config_type": "Feature"},
    )


def list_feature_configs_view(request):
    configs = FeatureConfig.objects.all()
    return render(
        request,
        "mymlopsapp1/list_configs.html",
        {
            "configs": configs,
            "config_type": "Feature",
            "create_url_name": "mymlopsapp1:create_feature_config",
        },
    )


# --- ModelConfig 생성 뷰 ---
def create_model_config_view(request):
    if request.method == "POST":
        form = ModelConfigForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect("mymlopsapp1:list_model_configs")
    else:
        form = ModelConfigForm()
    return render(
        request,
        "mymlopsapp1/create_config.html",
        {"form": form, "config_type": "Model"},
    )


def list_model_configs_view(request):
    configs = ModelConfig.objects.all()
    return render(
        request,
        "mymlopsapp1/list_configs.html",
        {
            "configs": configs,
            "config_type": "Model",
            "create_url_name": "mymlopsapp1:create_model_config",
        },
    )


# --- 모델 학습 뷰 ---
def train_model_view(request):
    dataset_configs = DatasetConfig.objects.all()
    feature_configs = FeatureConfig.objects.all()
    model_configs = ModelConfig.objects.all()

    if request.method == "POST":
        dataset_config_id = request.POST.get("dataset_config")
        feature_config_id = request.POST.get("feature_config")
        model_config_id = request.POST.get("model_config")

        if not dataset_config_id or not feature_config_id or not model_config_id:
            return render(
                request,
                "mymlopsapp1/train_model.html",
                {
                    "error": "데이터셋, 피처, 모델 설정을 모두 선택해주세요.",
                    "dataset_configs": dataset_configs,
                    "feature_configs": feature_configs,
                    "model_configs": model_configs,
                },
            )

        try:
            selected_dataset_config = get_object_or_404(
                DatasetConfig, id=dataset_config_id
            )
            selected_feature_config = get_object_or_404(
                FeatureConfig, id=feature_config_id
            )
            selected_model_config = get_object_or_404(ModelConfig, id=model_config_id)

            # 선택된 모델 설정이 올바른 데이터셋/피처 설정을 참조하는지 검증 (선택 사항)
            if (
                selected_model_config.dataset_config != selected_dataset_config
                or selected_model_config.feature_config != selected_feature_config
            ):
                raise ValueError(
                    "선택된 모델 설정이 선택된 데이터셋 또는 피처 설정과 일치하지 않습니다."
                )

            # --- 데이터 로드 ---
            dataset_full_path = os.path.join(
                DATA_ROOT, selected_dataset_config.file_path
            )

            if not os.path.exists(dataset_full_path):
                raise FileNotFoundError(
                    f"데이터셋 파일이 존재하지 않습니다: {dataset_full_path}. 'data/' 폴더에 파일을 확인하세요."
                )

            df = pd.read_csv(dataset_full_path)
            print(f"Dataset loaded from local file: {dataset_full_path}")

            features = json.loads(selected_feature_config.features)
            target_column = (
                selected_dataset_config.target_column
            )  # DatasetConfig에서 타겟 컬럼 로드

            all_required_cols = features + [target_column]
            missing_cols = [col for col in all_required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"데이터셋에 누락된 컬럼이 있습니다: {', '.join(missing_cols)}. 필요한 컬럼: {', '.join(all_required_cols)}"
                )

            df = df.dropna()

            X = df[features]
            y = df[target_column]

            # --- 모델 타입에 따른 타겟 변수 전처리 및 모델 인스턴스화 ---
            model_type = selected_model_config.model_type
            model_params = selected_model_config.parameters
            evaluation_results = {}

            # 분류 모델 (LogisticRegression, RandomForestClassifier)
            if model_type in ["LogisticRegression", "RandomForestClassifier"]:
                # 와인 품질 데이터셋 등 연속형 타겟을 분류로 쓸 경우 타겟 변수 이진 분류 변환
                # 다른 데이터셋은 이 변환이 필요 없을 수 있음. 일반화 필요 시 조건 추가.
                if (
                    selected_dataset_config.name == "WineQualityRed"
                ):  # 또는 특정 데이터셋 이름으로 조건 설정
                    # 여기서는 'quality' 컬럼을 이진 분류로 변환한다고 가정
                    y = (y >= 7).astype(int)
                    print(
                        f"Target variable transformed for binary classification (quality >= 7): {y.unique()}"
                    )
                else:
                    # 다른 데이터셋에 대한 기본 분류 타겟 처리 로직 (필요시 추가)
                    pass

                if model_type == "LogisticRegression":
                    model = LogisticRegression(**model_params)
                elif model_type == "RandomForestClassifier":
                    model = RandomForestClassifier(**model_params)

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(
                    y_test, y_pred, output_dict=True, zero_division=0
                )
                evaluation_results = {
                    "accuracy": accuracy,
                    "classification_report": report,
                }

            # 회귀 모델 (LinearRegression, RandomForestRegressor)
            elif model_type in ["LinearRegression", "RandomForestRegressor"]:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                if model_type == "LinearRegression":
                    model = LinearRegression(**model_params)
                elif model_type == "RandomForestRegressor":
                    model = RandomForestRegressor(**model_params)

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                evaluation_results = {"mse": mse, "r2_score": r2}
            else:
                raise ValueError(f"지원하지 않는 모델 타입: {model_type}")

            # 모델 저장
            model_filename = os.path.join(
                MODEL_DIR,
                f"{selected_model_config.name}_{selected_dataset_config.name}_{selected_feature_config.name}.pkl",
            )
            joblib.dump(model, model_filename)

            # 결과를 세션에 저장
            request.session["model_results"] = {
                "evaluation_results": evaluation_results,
                "model_name": selected_model_config.name,
                "dataset_name": selected_dataset_config.name,
                "feature_name": selected_feature_config.name,  # 피처 이름도 추가
                "model_path": model_filename,
                "model_type": model_type,
            }

            return redirect("mymlopsapp1:model_results")

        except FileNotFoundError as e:
            error_message = (
                f"파일 오류: {e}. 'data/' 폴더에 필요한 데이터셋 파일을 확인하세요."
            )
            return render(
                request,
                "mymlopsapp1/train_model.html",
                {
                    "error": error_message,
                    "dataset_configs": dataset_configs,
                    "feature_configs": feature_configs,
                    "model_configs": model_configs,
                },
            )
        except ValueError as e:
            error_message = f"데이터/설정 오류: {e}. 데이터셋/피처/모델 설정을 확인하세요. (누락 컬럼, 타입 불일치 등)"
            return render(
                request,
                "mymlopsapp1/train_model.html",
                {
                    "error": error_message,
                    "dataset_configs": dataset_configs,
                    "feature_configs": feature_configs,
                    "model_configs": model_configs,
                },
            )
        except Exception as e:
            error_message = f"모델 학습 중 알 수 없는 오류 발생: {e}"
            return render(
                request,
                "mymlopsapp1/train_model.html",
                {
                    "error": error_message,
                    "dataset_configs": dataset_configs,
                    "feature_configs": feature_configs,
                    "model_configs": model_configs,
                },
            )

    # GET 요청 시, 모든 사용 가능한 설정 정보를 템플릿으로 전달
    context = {
        "dataset_configs": dataset_configs,
        "feature_configs": feature_configs,
        "model_configs": model_configs,
        "error": None,
    }
    return render(request, "mymlopsapp1/train_model.html", context)


def model_results_view(request):
    """
    가장 최근 학습된 모델 결과 표시 뷰
    """
    # 'current_run_id'를 사용하여 특정 실행 결과를 가져옵니다.
    current_run_id = request.session.get("current_run_id")
    results = None
    if current_run_id and "past_runs" in request.session:
        for run in request.session["past_runs"]:
            if run["run_id"] == current_run_id:
                results = run
                break

    if not results:
        return redirect("mymlopsapp1:train_model")

    context = {
        "evaluation_results": results.get("evaluation_results"),
        "model_name": results.get("model_name"),
        "dataset_name": results.get("dataset_name"),
        "feature_name": results.get("feature_name"),
        "model_filename": os.path.basename(results.get("model_path", "N/A")),
        "model_type": results.get("model_type"),
        "run_id": results.get("run_id"),  # 실행 ID 추가
        "timestamp": results.get("timestamp"),  # 타임스탬프 추가
    }
    return render(request, "mymlopsapp1/model_results.html", context)


# --- 새로운 compare_results_view 함수 ---
class CompareResultsForm(forms.Form):
    """
    비교할 모델 학습 결과를 선택하기 위한 폼
    """

    results_to_compare = MultipleChoiceField(
        widget=CheckboxSelectMultiple,
        choices=[],  # 뷰에서 동적으로 채워질 것임
        label="Select results to compare",
    )


def compare_results_view(request):
    """
    여러 모델 학습 결과를 비교하여 표시하는 뷰
    """
    past_runs = request.session.get("past_runs", [])

    # 폼 필드에 선택 옵션 동적으로 채우기
    # 각 실행 결과에 대한 고유한 레이블 생성 (예: 모델이름_데이터셋이름_시간_ID)
    choices = []
    for run in past_runs:
        label = f"[{run['timestamp']}] {run['model_name']} on {run['dataset_name']} ({run['model_type']}) - ID: {run['run_id']}"
        choices.append((run["run_id"], label))  # (value, label)

    form = CompareResultsForm(request.POST or None)
    form.fields["results_to_compare"].choices = choices

    selected_results_data = []

    if request.method == "POST" and form.is_valid():
        selected_run_ids = form.cleaned_data["results_to_compare"]
        for run_id in selected_run_ids:
            for run in past_runs:
                if run["run_id"] == run_id:
                    selected_results_data.append(run)
                    break
        # 결과를 정렬 (예: timestamp 기준)
        selected_results_data.sort(key=lambda x: x["timestamp"])

    context = {
        "form": form,
        "selected_results": selected_results_data,
        "has_results_to_compare": len(past_runs)
        > 1,  # 비교할 결과가 2개 이상일 때만 표시
    }
    return render(request, "mymlopsapp1/compare_results.html", context)


def dataset_upload_view(request):
    if request.method == "POST":
        form = DatasetUploadAndConfigForm(request.POST, request.FILES)
        print("\n--- Form Submitted (POST Request) ---")

        if form.is_valid():
            print("--- Form is VALID ---")
            uploaded_file = form.cleaned_data["file"]
            dataset_name = form.cleaned_data["dataset_name"]
            description = form.cleaned_data["description"]

            file_name = uploaded_file.name
            file_path_in_data_dir = os.path.join(DATA_ROOT, file_name)

            print(f"Attempting to save file to: {file_path_in_data_dir}")

            try:
                os.makedirs(DATA_ROOT, exist_ok=True)

                with open(file_path_in_data_dir, "wb+") as destination:
                    for chunk in uploaded_file.chunks():
                        destination.write(chunk)
                print("--- File saved successfully to disk ---")

                df = pd.read_csv(file_path_in_data_dir)
                actual_columns = df.columns.tolist()
                print(f"--- CSV parsed. Detected columns: {actual_columns} ---")

                if not actual_columns:
                    print("ERROR: CSV has no columns/headers.")
                    form.add_error(
                        "file",
                        "업로드된 CSV 파일에 컬럼(헤더)이 없습니다. 유효한 CSV 파일인지 확인해주세요.",
                    )
                    return render(
                        request, "mymlopsapp1/dataset_upload.html", {"form": form}
                    )

                auto_detected_target_column = actual_columns[-1]
                print(
                    f"--- Auto-detected target column: {auto_detected_target_column} ---"
                )

                # DatasetConfig 생성 또는 업데이트 시 features는 빈 리스트를 JSON 문자열로 변환하여 저장
                dataset_config, created = DatasetConfig.objects.get_or_create(
                    name=dataset_name,
                    defaults={
                        "file_path": file_name,
                        "target_column": auto_detected_target_column,
                        "features": json.dumps([]),
                        "description": description,
                    },
                )
                if not created:
                    dataset_config.file_path = file_name
                    dataset_config.target_column = auto_detected_target_column
                    dataset_config.description = description
                    dataset_config.features = json.dumps([])
                    dataset_config.save()

                if created:
                    print(
                        f"--- DatasetConfig '{dataset_name}' CREATED successfully. ---"
                    )
                else:
                    print(
                        f"--- DatasetConfig '{dataset_name}' UPDATED successfully. ---"
                    )

                from django.contrib import messages

                messages.success(
                    request,
                    f"데이터셋 '{dataset_name}'이(가) 성공적으로 업로드 및 설정되었습니다. (타겟 컬럼: '{auto_detected_target_column}')",
                )
                print("--- Redirecting to list_dataset_configs ---")
                return redirect("mymlopsapp1:list_dataset_configs")

            except pd.errors.EmptyDataError:
                print("EXCEPTION CAUGHT: pandas.errors.EmptyDataError")
                form.add_error("file", "업로드된 CSV 파일이 비어 있습니다.")
            except pd.errors.ParserError:
                print("EXCEPTION CAUGHT: pandas.errors.ParserError")
                form.add_error(
                    "file",
                    "CSV 파일을 파싱하는 데 실패했습니다. 파일 형식을 확인해주세요.",
                )
            except Exception as e:
                print(f"EXCEPTION CAUGHT: General Error - {e}")
                import traceback

                traceback.print_exc()
                form.add_error(None, f"파일 처리 중 알 수 없는 오류 발생: {e}")

            print("--- Rendering form again due to error/exception ---")
            return render(request, "mymlopsapp1/dataset_upload.html", {"form": form})

        else:
            print("--- Form is NOT VALID ---")
            print("Form Errors:", form.errors)
            return render(request, "mymlopsapp1/dataset_upload.html", {"form": form})

    else:  # GET request
        form = DatasetUploadAndConfigForm()
    return render(request, "mymlopsapp1/dataset_upload.html", {"form": form})


# # --- About 페이지 ---
# def about(request):
#     return HttpResponse("This is the about page of myMLOpsApp1.")
