from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import os
import pandas as pd
import mlflow
import yaml
from sklearn.metrics import f1_score

# Настройка переменных окружения для MinIO и MLflow
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

# Пути к данным
current_dir = os.path.dirname(os.path.abspath(__file__))
path_to_val = os.path.join(current_dir, '..', 'data', 'validation')
path_to_input = os.path.join(current_dir, '..', 'data', 'config')

# Функция для выбора лучшей модели и перевода её в stage 'Production'
def choose_best_model():
    # Настройка MLflow
    mlflow.set_tracking_uri('http://mlflow_server:5000')
    client = mlflow.MlflowClient()

    # Загрузка валидационных данных
    x_val = pd.read_csv(os.path.join(path_to_val, 'x_val.csv'))
    y_val = pd.read_csv(os.path.join(path_to_val, 'y_val.csv'))

    # Загрузка конфигурационного файла
    with open(os.path.join(path_to_input, 'config.yaml'), 'r') as file:
        config = yaml.safe_load(file)

    # Словарь для хранения значений F1-score для каждой модели
    f1_scores = {}

    # Оценка каждой модели из конфигурации
    for _, conf in enumerate(config['configs']):
        model_uri = f"models:/{conf['classificator']}/None"
        model = mlflow.sklearn.load_model(model_uri)
        y_pred = model.predict(x_val)
        f1_scores[conf['classificator']] = f1_score(y_val, y_pred)

    # Выбор модели с наилучшим F1-score
    best_model_name = max(f1_scores, key=f1_scores.get)

    # Перевод выбранной модели на stage 'Production'
    filter_string = f"name='{best_model_name}'"
    best_model = client.search_registered_models(filter_string=filter_string)[0]
    all_versions = client.search_model_versions(filter_string=filter_string)
    client.transition_model_version_stage(
        name=best_model.name,
        version=all_versions[0].version,
        stage='Production'
    )

# Определение DAG для Airflow
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 11, 26),
    'retries': 1,
}

dag = DAG(
    'model_validation_and_hosting',
    default_args=default_args,
    description='DAG for validating and hosting the best model',
    schedule_interval='@daily',
)

validate_and_host = PythonOperator(
    task_id='choose_best_model',
    python_callable=choose_best_model,
    dag=dag,
)

validate_and_host
