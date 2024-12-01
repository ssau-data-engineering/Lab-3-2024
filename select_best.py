import os
import yaml
import mlflow

from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer


def select_best():
    x, y = load_breast_cancer(return_X_y=True)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=123)

    with open('data/config.yaml') as file:
        configs = yaml.safe_load(file.read())['configs']

    mlflow.set_tracking_uri('http://mlflow_server:5000')
    acc = {}
    for config in configs:
        acc[config['classifier']] = accuracy_score(
            y_val,
            mlflow.sklearn.load_model(f"models:/breast_cancer_{config['classifier']}/None").predict(x_val)
        )

    best_model_name = max(acc, key=acc.get)
    client, filter_string = mlflow.MlflowClient(), f"name='breast_cancer_{best_model_name}'"
    best_model = client.search_registered_models(filter_string)[0]
    all_versions = client.search_model_versions(filter_string)
    client.transition_model_version_stage(
        name=best_model.name,
        version=all_versions[0].version,
        stage='Production'
    )


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
    python_callable=select_best,
    dag=dag
)