import datetime
from airflow import DAG
from airflow.sensors.filesystem import FileSensor
from airflow.operators.bash_operator import BashOperator
import os

os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

dag = DAG(
    dag_id="LR_3_task_1",
    schedule="0 0 * * *", 
    start_date=datetime.datetime(2024, 11, 9, tzinfo=datetime.timezone.utc), 
    catchup=False,  
    dagrun_timeout=datetime.timedelta(minutes=60),  
    tags=["LR"],
)

wait_for_new_json_file = FileSensor(
    task_id='wait_for_new_json_file',
    poke_interval=10, 
    filepath='./data/LR_3_task1/models.json',  
    fs_conn_id='file-connection-id', 
    dag=dag,
)

train_models = BashOperator(
    task_id = 'train_models',
    bash_command = 'python /opt/***/data/LR_3_task1/train.py',
    dag = dag
)

wait_for_new_json_file >> train_models