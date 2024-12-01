import os
from datetime import datetime, timezone
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.sensors.filesystem import FileSensor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
import mlflow
import mlflow.sklearn
import json

os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

CONFIG_FOLDER = './data'

def load_config(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def train_model(**kwargs):
    config_file = kwargs['params']['config_file']
    config = load_config(config_file)
    
    data = fetch_openml('mnist_784', version=1, as_frame=False)
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    mlflow.set_tracking_uri('http://mlflow_server:5000')

    experiment_name = "MNIST_Experiment"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)

    for model_config in config['configs']:
        module_name = model_config['module']
        classificator_name = model_config['classificator']
        model_class = getattr(__import__(module_name, fromlist=[classificator_name]), classificator_name)
        model = model_class(**model_config['args'])
        model.fit(X_train, y_train)
        
        with mlflow.start_run(run_name=f"{module_name}.{classificator_name}", experiment_id=experiment_id):
            mlflow.log_params(model_config['args'])
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}

            mlflow.log_metrics(metrics)
            
            model_info = mlflow.sklearn.log_model(model, artifact_path=f"{module_name}.{classificator_name}")
            
            model_uri = model_info.model_uri

            registered_model = mlflow.register_model(model_uri=model_uri, name=f"{module_name}.{classificator_name}")

        mlflow.end_run()

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 11, 30, tzinfo=timezone.utc), 
    'retries': 1,
}

dag = DAG(
    dag_id = 'task1',
    default_args=default_args,
    description='Pipeline for training classifiers',
    schedule_interval=None,
    catchup=False
)

wait_for_config_file = FileSensor(
    task_id='wait_for_config_file',
    filepath=CONFIG_FOLDER,
    fs_conn_id='task1',
    poke_interval=10,
    timeout=600,
    dag=dag
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    provide_context=True,
    params={'config_file': os.getenv('CONFIG_FILE_PATH', f'{CONFIG_FOLDER}/config.json')},
    dag=dag
)

wait_for_config_file >> train_model_task