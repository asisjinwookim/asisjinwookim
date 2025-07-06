from django import forms
import json
from .models import DatasetConfig, FeatureConfig, ModelConfig

# mymlopsapp1/forms.py


class DatasetConfigForm(forms.ModelForm):
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
    # features 필드는 TextField이므로, 사용자가 JSON 문자열로 입력하고
    # 저장 시 JSON 문자열로 변환하여 저장합니다.
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance and self.instance.features:
            # 기존 인스턴스가 있다면 features 필드의 JSON 문자열을 로드하여 폼 필드에 채움
            self.initial["features_json"] = (
                self.instance.features
            )  # 이미 JSON 문자열이므로 그대로 할당

    def clean_features_json(self):
        data = self.cleaned_data["features_json"]
        try:
            features_list = json.loads(data)
            if not isinstance(features_list, list):
                raise forms.ValidationError("피처는 JSON 배열 형식이어야 합니다.")
            return features_list  # 파이썬 리스트로 반환 (저장 시 다시 덤프할 것임)
        except json.JSONDecodeError:
            raise forms.ValidationError("유효한 JSON 형식이 아닙니다.")

    def save(self, commit=True):
        instance = super().save(commit=False)
        instance.features = json.dumps(
            self.cleaned_data["features_json"]
        )  # <-- 여기 수정: 리스트를 JSON 문자열로 덤프
        if commit:
            instance.save()
        return instance


class ModelConfigForm(forms.ModelForm):
    MODEL_TYPE_CHOICES = [
        ("LogisticRegression", "Logistic Regression (Classification)"),
        ("LinearRegression", "Linear Regression (Regression)"),
        ("RandomForestClassifier", "Random Forest Classifier (Classification)"),
        ("RandomForestRegressor", "Random Forest Regressor (Regression)"),
    ]
    model_type = forms.ChoiceField(choices=MODEL_TYPE_CHOICES, label="Model Type")

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
        if self.instance and self.instance.parameters:
            self.initial["parameters_json"] = (
                self.instance.parameters
            )  # 이미 JSON 문자열이므로 그대로 할당

    def clean_parameters_json(self):
        data = self.cleaned_data["parameters_json"]
        try:
            params_dict = json.loads(data)
            if not isinstance(params_dict, dict):
                raise forms.ValidationError("파라미터는 JSON 객체 형식이어야 합니다.")
            return params_dict  # 파이썬 딕셔너리로 반환
        except json.JSONDecodeError:
            raise forms.ValidationError("유효한 JSON 형식이 아닙니다.")

    def save(self, commit=True):
        instance = super().save(commit=False)
        instance.parameters = json.dumps(
            self.cleaned_data["parameters_json"]
        )  # <-- 여기 수정: 딕셔너리를 JSON 문자열로 덤프
        if commit:
            instance.save()
        return instance
