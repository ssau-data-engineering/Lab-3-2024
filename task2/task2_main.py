import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Установка переменных среды для Minio
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

# Функция для валидации моделей
def validate_and_promote_models():
    mlflow.set_tracking_uri('http://mlflow_server:5000')
    client = MlflowClient()
    best_metric_value = None
    best_model_version = None
    best_model_name = None

    for model_name in client.search_registered_models():
        model_name = model_name.name
        for version in client.search_model_versions(f"name='{model_name}'"):
            version_number = version.version
            run_id = version.run_id

            run = client.get_run(run_id)
            metric_value = run.data.metrics.get('accuracy')

            if best_metric_value is None or metric_value > best_metric_value:
                best_metric_value = metric_value
                best_model_version = version_number
                best_model_name = model_name

    if best_model_name:
        client.transition_model_version_stage(
            name=best_model_name,
            version=best_model_version,
            stage="Production"
        )
        print(f"Promoted model {best_model_name} version {best_model_version} to Production with metric value {best_metric_value}")
    else:
        print("No models found to promote.")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    dag_id = 'lab3_task2',
    default_args=default_args,
    description='Pipeline for validating and hosting the best model',
    schedule_interval=timedelta(days=1),
    catchup=False
)

validate_and_promote_models_task = PythonOperator(
    task_id='validate_and_promote_models',
    python_callable=validate_and_promote_models,
    provide_context=True,
    dag=dag
)

validate_and_promote_models_task