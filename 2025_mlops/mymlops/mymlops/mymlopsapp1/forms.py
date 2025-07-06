from django import forms
import json
from .models import DatasetConfig, FeatureConfig, ModelConfig


class DatasetConfigForm(forms.ModelForm):
    # file_path는 사용자가 직접 입력하도록 합니다.
    # CSV 파일을 서버에 업로드하는 기능은 별도의 FileField와 핸들링 로직이 필요하므로,
    # 여기서는 경로를 텍스트로 입력하는 방식으로 간소화합니다.
    class Meta:
        model = DatasetConfig
        fields = ["name", "file_path", "target_column", "description"]
        widgets = {
            "file_path": forms.TextInput(
                attrs={"placeholder": "예: winequality-red.csv (data/ 폴더 기준)"}
            ),
            "description": forms.Textarea(attrs={"rows": 3}),
        }


class FeatureConfigForm(forms.ModelForm):
    # features 필드는 JSONField이므로, 사용자가 JSON 문자열로 입력하도록 합니다.
    # 실제 앱에서는 JSON 입력 필드를 더 사용자 친화적으로 만들 수 있습니다 (예: textarea, JavaScript 에디터).
    features_json = forms.CharField(
        label="Features (JSON Array)",
        widget=forms.Textarea(
            attrs={
                "rows": 5,
                "placeholder": '예: ["fixed acidity", "volatile acidity"]',
            }
        ),
        help_text="학습에 사용할 피처 이름을 JSON 배열 형식으로 입력하세요.",
    )

    class Meta:
        model = FeatureConfig
        fields = ["name", "description"]  # features 필드는 features_json으로 대체
        widgets = {
            "description": forms.Textarea(attrs={"rows": 3}),
        }

    def clean_features_json(self):
        data = self.cleaned_data["features_json"]
        try:
            # 입력받은 JSON 문자열을 파이썬 리스트로 파싱
            features_list = json.loads(data)
            if not isinstance(features_list, list):
                raise forms.ValidationError("피처는 JSON 배열 형식이어야 합니다.")
            return features_list
        except json.JSONDecodeError:
            raise forms.ValidationError("유효한 JSON 형식이 아닙니다.")

    def save(self, commit=True):
        instance = super().save(commit=False)
        instance.features = self.cleaned_data[
            "features_json"
        ]  # 파싱된 리스트를 모델 필드에 할당
        if commit:
            instance.save()
        return instance


class ModelConfigForm(forms.ModelForm):
    # 모델 타입 드롭다운 목록을 직접 정의 (향후 확장 가능)
    MODEL_TYPE_CHOICES = [
        ("LogisticRegression", "Logistic Regression (Classification)"),
        ("LinearRegression", "Linear Regression (Regression)"),
        ("RandomForestClassifier", "Random Forest Classifier (Classification)"),
        ("RandomForestRegressor", "Random Forest Regressor (Regression)"),
        # 다른 모델 타입 추가 가능
    ]
    model_type = forms.ChoiceField(choices=MODEL_TYPE_CHOICES, label="Model Type")

    # parameters 필드는 JSONField이므로, 사용자가 JSON 문자열로 입력하도록 합니다.
    parameters_json = forms.CharField(
        label="Model Parameters (JSON Object)",
        widget=forms.Textarea(
            attrs={
                "rows": 5,
                "placeholder": '예: {"n_estimators": 100, "max_depth": 5}',
            }
        ),
        help_text="모델의 하이퍼파라미터를 JSON 객체 형식으로 입력하세요.",
    )

    class Meta:
        model = ModelConfig
        fields = [
            "name",
            "dataset_config",
            "feature_config",
            "model_type",
            "description",
        ]
        widgets = {
            "description": forms.Textarea(attrs={"rows": 3}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 폼 로드 시 기존 인스턴스에서 JSON 데이터를 텍스트로 변환하여 폼 필드에 채움 (편집 시)
        if self.instance and self.instance.parameters:
            self.initial["parameters_json"] = json.dumps(
                self.instance.parameters, indent=2
            )

    def clean_parameters_json(self):
        data = self.cleaned_data["parameters_json"]
        try:
            params_dict = json.loads(data)
            if not isinstance(params_dict, dict):
                raise forms.ValidationError("파라미터는 JSON 객체 형식이어야 합니다.")
            return params_dict
        except json.JSONDecodeError:
            raise forms.ValidationError("유효한 JSON 형식이 아닙니다.")

    def save(self, commit=True):
        instance = super().save(commit=False)
        instance.parameters = self.cleaned_data[
            "parameters_json"
        ]  # 파싱된 딕셔너리를 모델 필드에 할당
        if commit:
            instance.save()
        return instance
