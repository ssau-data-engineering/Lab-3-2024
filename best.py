import os
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

from find_best_model import main as find_best_model


os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

dag = DAG(
    'select_best_model',
    default_args=default_args,
    description='Select best model by metric',
    schedule_interval=None,
)

validate = PythonOperator(
    task_id='best_model',
    python_callable=find_best_model,
    dag=dag
)