from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
import os

os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

default_args = {
    'owner': 'WonMin13',
    'start_date': datetime(2024, 11, 30),
    'retries': 1,
}

dag = DAG(
    'love_bts_validate_model',
    default_args=default_args,
    description='',
    schedule_interval='@daily',
)

validate_model = BashOperator(
    task_id="validate_model",
    bash_command="python /opt/airflow/data/validate.py",
    dag=dag
)

validate_model