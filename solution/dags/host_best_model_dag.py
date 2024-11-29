import os
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import mlflow
from mlflow.tracking import MlflowClient
from airflow.utils.dates import days_ago

# Настройка переменных среды
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

# Параметры DAG
default_args = {
    'owner': 'snail',
    'start_date': days_ago(0, 0, 0, 0),
    'retries': 1,
}

dag = DAG(
    'host_best_model_dag',
    default_args=default_args,
)


def validate_and_promote_model():
    mlflow.set_tracking_uri("http://web:5000")
    client = MlflowClient()

    # Получаем все зарегистрированные модели
    registered_models = client.search_registered_models()


    best_accuracy = 0
    best_model_name = None
    best_version = None

    # Проходим по всем моделям
    for model in registered_models:
        model_name = model.name
        versions = client.search_model_versions(f"name='{model_name}'")

        # Для каждой версии модели получаем метрики
        for version in versions:
            run_id = version.run_id
            run = client.get_run(run_id)
            accuracy = run.data.metrics.get("accuracy")

            # Если точность модели лучше, обновляем лучшие параметры
            if accuracy and accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = model_name
                best_version = version

    if best_version:
        # Перевод лучшей модели на стадию "Production"
        client.transition_model_version_stage(
            name=best_model_name,
            version=best_version.version,
            stage="Production",
            archive_existing_versions=True  # Архивируем другие версии
        )
        print(
            f"Модель {best_model_name} версии {best_version.version} переведена на стадию 'Production' с точностью {best_accuracy}"
        )

select_model_task = PythonOperator(
    task_id='select_best_model',
    python_callable=validate_and_promote_model,
    dag=dag,
)
