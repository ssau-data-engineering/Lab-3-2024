from airflow import DAG
from datetime import datetime
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
from airflow.operators.bash_operator import BashOperator
from airflow.sensors.filesystem import FileSensor
import os

os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"


default_args = {
    'owner': 'IlyaSwallow',
    'retries': 1,
}

dag = DAG(
    'training_model_in_dagestan',
    default_args=default_args,
    start_date = datetime(2023, 12, 3),
    schedule_interval=None,
)

wait_configuration_file_task = FileSensor(
    task_id='wait_file_task',
    poke_interval=5,
    filepath='/opt/airflow/data',
    fs_conn_id='fs_connection_default',
    dag=dag,
)

prepare_data_for_working_task = DockerOperator(
    task_id='prepare_data_for_working_task',
    image='ilyaswallow/our_tensorflow_container:1.0',
    command='python /data/prepare_data.py',
    mounts=[Mount(source='/data', target='/data', type='bind')],
    docker_url="tcp://docker-proxy:2375",
    dag=dag,
)

train_model_task = BashOperator(
    task_id="train_model_task",
    bash_command="python /opt/airflow/data/train.py",
    dag=dag
)

wait_configuration_file_task >> prepare_data_for_working_task >> train_model_task