from django.db import models
from django.utils import timezone

# from django.contrib.postgres.fields import (
#     ArrayField,
# )  # PostgreSQL을 사용할 경우, 다른 DB는 JSONField 고려


class FeatureConfig(models.Model):
    name = models.CharField(
        max_length=100,
        unique=True,
        help_text="피처 설정의 고유 이름 (예: WineFeatures_All)",
    )
    # 어떤 DatasetConfig와 관련된 피처 구성인지 명시적으로 연결
    dataset_config = models.ForeignKey(
        "DatasetConfig",
        on_delete=models.CASCADE,
        related_name="feature_configs",
        help_text="이 피처 조합이 적용될 데이터셋",
    )
    features = models.TextField(
        help_text="학습에 사용할 피처 리스트 (JSON 배열 형식의 문자열)", default="[]"
    )
    description = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} (Dataset: {self.dataset_config.name})"  # 출력 형식 변경

    class Meta:
        verbose_name = "피처 설정"
        verbose_name_plural = "피처 설정"
        # 동일한 데이터셋에 대해 동일한 이름의 피처 구성을 만들 수 없도록 UniqueConstraint 추가
        unique_together = ("name", "dataset_config")


class DatasetConfig(models.Model):
    name = models.CharField(
        max_length=100,
        unique=True,
        help_text="데이터셋 설정의 고유 이름 (예: WineQualityRed)",
    )
    file_path = models.CharField(
        max_length=255, help_text="데이터셋 파일의 상대 경로 (예: winequality-red.csv)"
    )
    target_column = models.CharField(max_length=50, help_text="타겟 변수 컬럼 이름")
    description = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = "데이터셋 설정"
        verbose_name_plural = "데이터셋 설정"


class ModelConfig(models.Model):
    name = models.CharField(
        max_length=100,
        unique=True,
        help_text="모델 설정의 고유 이름 (예: WineQuality_LogisticRegression)",
    )
    dataset_config = models.ForeignKey(
        DatasetConfig, on_delete=models.CASCADE, help_text="이 모델이 사용할 데이터셋"
    )
    feature_config = models.ForeignKey(
        FeatureConfig, on_delete=models.CASCADE, help_text="이 모델이 사용할 피처 조합"
    )
    model_type = models.CharField(
        max_length=50,
        help_text="사용할 모델 타입 (예: LogisticRegression, RandomForest)",
    )
    # JSONField -> TextField로 변경 (필요하다면)
    parameters = models.TextField(
        help_text="모델 하이퍼파라미터 (JSON 객체 형식의 문자열)", default="{}"
    )
    description = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = "모델 설정"
        verbose_name_plural = "모델 설정"


class Dataset(models.Model):
    name = models.CharField(max_length=255, unique=True, verbose_name="데이터셋 이름")
    description = models.TextField(blank=True, verbose_name="설명")
    file = models.FileField(upload_to="data_files/", verbose_name="데이터 파일")
    uploaded_at = models.DateTimeField(auto_now_add=True, verbose_name="업로드 날짜")

    def __str__(self):
        return self.name


class MLModel(models.Model):
    name = models.CharField(max_length=255, verbose_name="모델 이름")
    description = models.TextField(blank=True, verbose_name="설명")
    algorithm = models.CharField(
        max_length=100, verbose_name="알고리즘 (예: RandomForest, LogisticRegression)"
    )
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="생성 날짜")

    def __str__(self):
        return self.name


class TrainingResult(models.Model):
    dataset = models.ForeignKey(
        Dataset,
        on_delete=models.CASCADE,
        related_name="training_results",
        verbose_name="사용된 데이터셋",
    )
    ml_model_config = models.ForeignKey(
        MLModel,
        on_delete=models.CASCADE,
        related_name="training_results",
        verbose_name="모델 설정",
    )

    params = models.JSONField(default=dict, blank=True, verbose_name="학습 파라미터")

    accuracy = models.FloatField(null=True, blank=True, verbose_name="정확도")
    precision = models.FloatField(null=True, blank=True, verbose_name="정밀도")
    recall = models.FloatField(null=True, blank=True, verbose_name="재현율")
    f1_score = models.FloatField(null=True, blank=True, verbose_name="F1 스코어")

    training_duration_seconds = models.FloatField(
        null=True, blank=True, verbose_name="학습 시간 (초)"
    )

    model_file_path = models.CharField(
        max_length=500, blank=True, verbose_name="학습된 모델 파일 경로"
    )

    trained_at = models.DateTimeField(auto_now_add=True, verbose_name="학습 완료 시간")

    version = models.CharField(max_length=50, blank=True, verbose_name="결과 버전")

    def __str__(self):
        return f"Result for {self.ml_model_config.name} on {self.dataset.name} ({self.trained_at.strftime('%Y-%m-%d %H:%M')})"

    class Meta:
        ordering = ["-trained_at"]
