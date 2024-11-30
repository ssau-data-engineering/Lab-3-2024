import os
from datetime import datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.sensors.filesystem import FileSensor
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import sklearn
import pandas as pd
import mlflow
import mlflow.sklearn
import yaml
import json

os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

CONFIG_FOLDER = './data/configs'

def load_config(file_path):
    with open(file_path, 'r') as file:
        if file_path.endswith('.yaml') or file_path.endswith('.yml'):
            return yaml.safe_load(file)
        elif file_path.endswith('.json'):
            return json.load(file)
        else:
            raise ValueError("Unsupported file format")

def train_model(**kwargs):
    config_file = kwargs['params']['config_file']
    config = load_config(config_file)
    
    data = pd.read_csv("./data/dataset/train.csv")

    X = data.drop(columns=['price_range'])
    y = data['price_range']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    mlflow.set_tracking_uri('http://mlflow_server:5000')
    experiment_id = mlflow.create_experiment("12345")
    mlflow.set_experiment("12345")

    for model_config in config['models']:
        model_class = eval(model_config['classificator'])
        model = model_class(**model_config['kwargs'])
        model.fit(X_train, y_train)
        
        # Логгирование параметров модели в MLflow
        with mlflow.start_run(run_name=model_config['classificator'], experiment_id=experiment_id):
            mlflow.log_params(model_config['kwargs'])
            
            # Логгирование процесса обучения
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}

            mlflow.log_metrics(metrics)
            
            # Сохранение модели в MLflow
            model_info = mlflow.sklearn.log_model(model, artifact_path=model_config['classificator'])
            
            # Сохранение модели в Minio
            model_uri = model_info.model_uri

            registered_model = mlflow.register_model(model_uri=model_uri, name=model_config['classificator'])

        mlflow.end_run()

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

dag = DAG(
    dag_id = 'lab3_task1',
    default_args=default_args,
    description='Pipeline for training classifiers',
    schedule_interval=None,
    catchup=False
)

wait_for_config_file = FileSensor(
    task_id='wait_for_config_file',
    filepath=CONFIG_FOLDER,
    fs_conn_id='lab3_task1',
    poke_interval=10,
    timeout=600,
    dag=dag
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    provide_context=True,
    params={'config_file': os.getenv('CONFIG_FILE_PATH', f'{CONFIG_FOLDER}/123.yaml')},
    dag=dag
)

wait_for_config_file >> train_model_task