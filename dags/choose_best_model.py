import os
import datetime
import pandas as pd
import mlflow
import importlib
import yaml
from sklearn.metrics import f1_score, accuracy_score
from airflow import DAG
from airflow.operators.python import PythonOperator


os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, '..', 'data', 'config.yaml')
VALIDATION_PATH = os.path.join(BASE_DIR, '..', 'data')


def load_validation_data():
    x_val = pd.read_csv(os.path.join(VALIDATION_PATH, 'x_val.csv'))
    y_val = pd.read_csv(os.path.join(VALIDATION_PATH, 'y_val.csv'))
    return x_val, y_val

def load_config():
    with open(CONFIG_PATH, 'r') as file:
        return yaml.safe_load(file)

def evaluate_models(X_val, y_val, config):
    scores = {}
    for _, model_config in enumerate(config['configs']):
        model = mlflow.sklearn.load_model(f"models:/{model_config['classificator']}/None")
        y_pred = model.predict(X_val)
        scores[model_config['classificator']] = f1_score(y_val, y_pred)
    return scores

def select_best_model(scores):
    best_model_name = max(scores, key=scores.get)
    return best_model_name

def transition_model_to_production(best_model_name):
    client = mlflow.MlflowClient()
    model_filter = f"name='{best_model_name}'"
    best_model = client.search_registered_models(model_filter)[0]
    all_versions = client.search_model_versions(filter_string=model_filter)
    client.transition_model_version_stage(
        name=best_model.name,
        version=all_versions[0].version,
        stage='Production'
    )

def main():
    mlflow.set_tracking_uri('http://mlflow_server:5000')
    X_val, y_val = load_validation_data()
    config = load_config()
    scores = evaluate_models(X_val, y_val, config)
    best_model_name = select_best_model(scores)
    transition_model_to_production(best_model_name)


dag = DAG(
    'choose_best_model',
    default_args={'owner': 'airflow'},
    dagrun_timeout=datetime.timedelta(minutes=60),
    start_date=datetime.datetime(2024, 1, 1),
    catchup=False,
    description='DAG selecting the best model',
    schedule_interval=None
)

model_selection_task = PythonOperator(
    task_id='model_selection_task',
    python_callable=main,
    dag=dag
)

model_selection_task