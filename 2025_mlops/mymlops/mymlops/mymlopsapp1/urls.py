from django.urls import path
from . import views

app_name = "mymlopsapp1"  # <-- Make sure this line exists and is uncommented

urlpatterns = [
    # 설정 생성 및 목록 보기 URL 추가
    path(
        "configs/datasets/create/",
        views.create_dataset_config_view,
        name="create_dataset_config",
    ),
    path(
        "configs/datasets/",
        views.list_dataset_configs_view,
        name="list_dataset_configs",
    ),
    path(
        "configs/features/create/",
        views.create_feature_config_view,
        name="create_feature_config",
    ),
    path(
        "configs/features/",
        views.list_feature_configs_view,
        name="list_feature_configs",
    ),
    path(
        "configs/models/create/",
        views.create_model_config_view,
        name="create_model_config",
    ),
    path("configs/models/", views.list_model_configs_view, name="list_model_configs"),
    path("configs/datasets/upload/", views.dataset_upload_view, name="dataset_upload"),
    # 모델 학습 관련 URL
    path("train/", views.train_model_view, name="train_model"),
    # 학습 결과 관련 URL
    path("results/", views.model_results_view, name="model_results"),
    path("compare_results/", views.compare_results_view, name="compare_results"),
    path("", views.home, name="home"),  # 기본 페이지
]
