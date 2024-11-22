import pandas as pd
import mlflow
from airflow import DAG
from airflow.operators.python import PythonOperator
from mlflow.tracking import MlflowClient
from datetime import datetime

# Конфигурация
MLFLOW_TRACKING_URI = 'http://mlflow_server:5000'
METRIC_KEY = 'roc_auc_test'
TIMER_INTERVAL = '@daily'

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(MLFLOW_TRACKING_URI)

def get_all_model_versions():
    """Получить список всех версий моделей из MLflow Registry."""
    all_model_versions = []
    registered_models = client.search_registered_models()
    for model in registered_models:
        for version in model.latest_versions:
            all_model_versions.append({
                "model_name": model.name,
                "version": version.version,
                "run_id": version.run_id,
                "stage": version.current_stage
            })
    return all_model_versions

def find_best_model(metric_key):
    """Поиск лучшей модели (среди всех версий и моделей) по метрике."""
    best_metric = float("-inf")
    best_model = None

    all_model_versions = get_all_model_versions()
    for model_version in all_model_versions:
        if model_version["stage"] not in ["Production"]:  # Исключаем Production-версии
            run = client.get_run(model_version["run_id"])
            metrics = run.data.metrics
            if metric_key in metrics:
                metric_value = metrics[metric_key]
                if metric_value > best_metric:
                    best_metric = metric_value
                    best_model = model_version

    return best_model, best_metric

def promote_best_model():
    """Перевод лучшей модели в Production."""
    best_model, best_metric = find_best_model(METRIC_KEY)

    if best_model:
        model_name = best_model["model_name"]
        version = best_model["version"]

        print(f"Лучшая модель: {model_name}, версия {version}, метрика {METRIC_KEY} = {best_metric}")

        # Перевод в Production
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True  # Архивируем старые Production версии
        )
        print(f"Модель {model_name} версии {version} успешно переведена в Production.")
    else:
        print("Нет доступных моделей для перевода в Production.")
        
        
default_args={
    'owner': 'airflow',
    'start_date': datetime(2024, 11, 19),
    'retries': 1,
}

dag = DAG(
    'MLFlow2',
    default_args=default_args,
    description='DAG for validating and promoting MLflow models',
    schedule_interval=TIMER_INTERVAL,  
)

validate_and_promote_task = PythonOperator(
    task_id='validate_and_promote_models',
    python_callable=promote_best_model,
    dag=dag,
)        

validate_and_promote_task