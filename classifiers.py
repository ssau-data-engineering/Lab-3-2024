import os
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor

from train import main as train


os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

dag = DAG(
    'train_classifies_from_config',
    default_args=default_args,
    description='DAG for training classifier and getting mlflow logs',
    schedule_interval=None,
)

wait_for_new_file = FileSensor(
    task_id='wait_for_new_file',
    poke_interval=10,
    filepath=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'),
    fs_conn_id='airflow',
    dag=dag
)

train = PythonOperator(
    task_id='train',
    python_callable=train,
    dag=dag
)

wait_for_new_file >> train
