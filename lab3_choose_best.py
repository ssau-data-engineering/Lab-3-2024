from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import os
import pandas as pd
import mlflow
import yaml
import pickle
import importlib
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

current_dir = os.path.dirname(os.path.abspath(__file__))
path_to_val = os.path.join(current_dir, '..', 'data', 'validation')
path_to_data = os.path.join(current_dir, '..', 'data', 'dataset')
path_to_input = os.path.join(current_dir, '..', 'data', 'config')


def choose_best_model():

    mlflow.set_tracking_uri('http://mlflow_server:5000')

    x_val = pd.read_csv(os.path.join(path_to_val, 'x_val.csv'))
    y_val = pd.read_csv(os.path.join(path_to_val, 'y_val.csv'))

    with open(os.path.join(path_to_input, 'config.yaml'), 'r') as file:
        config = yaml.safe_load(file)

    f1_scores = {}

    for _, conf in enumerate(config['configs']):
        model = mlflow.sklearn.load_model(f"models:/{conf['classificator']}/None")
        y_pred = model.predict(x_val)
        f1_scores[conf['classificator']] = f1_score(y_val, y_pred)

    max_f1 = max(f1_scores.values())
    for m in f1_scores.keys():
        if f1_scores[m] == max_f1:
            best_model_name = m

    client = mlflow.MlflowClient()

    filter_string = f"name='{best_model_name}'"
    best_model = client.search_registered_models(filter_string)[0]
    all_versions = client.search_model_versions(filter_string=filter_string)
    client.transition_model_version_stage(
            name=best_model.name,
            version=all_versions[0].version,
            stage='Production'
        )



default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 11, 7),
    'retries': 1,
}

dag = DAG(
    'validation_mlflow',
    default_args=default_args,
    description='DAG for validation',
    schedule_interval=None,
)


validate = PythonOperator(
        task_id='choose_best_model',
        python_callable=choose_best_model,
        dag=dag,
    )


validate