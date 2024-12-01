import os
import datetime
import pandas as pd
import mlflow
import yaml
import uuid
import importlib
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE_PATH = os.path.join(BASE_DIR, '..', 'data', 'config.yaml')
DATA_FILE_PATH = os.path.join(BASE_DIR, '..', 'data', 'brca.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, '..', 'data')

def read_config():
    with open(CONFIG_FILE_PATH, 'r') as f:
        return yaml.safe_load(f)

def preprocess_and_split_data():
    data = pd.read_csv(DATA_FILE_PATH).dropna()
    
    features = data.drop(columns=['Unnamed: 0', 'y'])
    target = data['y'].map({'B': 0, 'M': 1})

    X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=7)

    X_val.to_csv(os.path.join(OUTPUT_PATH, 'X_val.csv'), index=False)
    y_val.to_csv(os.path.join(OUTPUT_PATH, 'y_val.csv'), index=False)

    return X_train, X_test, y_train, y_test

def train_and_log_models():
    X_train, X_test, y_train, y_test = preprocess_and_split_data()
    config = read_config()

    mlflow.set_tracking_uri('http://mlflow_server:5000')
    experiment_id = str(uuid.uuid4())
    mlflow.create_experiment(experiment_id)
    mlflow.set_experiment(experiment_id)

    for conf in config['configs']:
        with mlflow.start_run(run_name=conf['classificator']):
            module = importlib.import_module('.'.join(conf['classificator'].split('.')[:-1]))
            model_class = getattr(module, conf['classificator'].split('.')[-1])
            model_instance = model_class(**conf['kwargs'])
            model_instance.fit(X_train, y_train)

            y_pred = model_instance.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred)
            }
            mlflow.log_metrics(metrics)

            model_info = mlflow.sklearn.log_model(model_instance, artifact_path=conf['classificator'])
            register_model = mlflow.register_model(model_info.model_uri, name=conf['classificator'])

dag = DAG(
    'train_five_models_from_config',
    default_args={'owner': 'airflow'},
    dagrun_timeout=datetime.timedelta(minutes=45),
    start_date=datetime.datetime(2024, 1, 1),
    catchup=False,
    description='DAG training 5 models from config file',
    schedule_interval=None
)

wait_for_config_file = FileSensor(
    task_id='wait_for_config_file',
    poke_interval=10,
    filepath=CONFIG_FILE_PATH,
    fs_conn_id='second_lab_con',
    dag=dag
)

train_models_task = PythonOperator(
    task_id='train_models_task',
    python_callable=train_and_log_models,
    dag=dag
)

wait_for_config_file >> train_models_task
