import json
import random
import uuid
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.datasets import load_iris

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from airflow.sensors.filesystem import FileSensor

from utils.api_keys_hub import FS_CONN_ID, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, MLFLOW_S3_ENDPOINT_URL

from datetime import datetime
import importlib
import os
from pathlib import Path

os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL


def load_config_from_json():
    work_dir = Path("./data/input_data")
    configs = []

    for file in work_dir.rglob("*.json"):
        with file.open('+r') as f:
            config = json.load(f)
            configs.append(config)

    print(type(configs))
    print(configs)

    Variable.set('configs', configs, serialize_json=True)


def train_model_mlflow():
    print(json.loads(Variable.get('configs')))
    configs = json.loads(Variable.get('configs'))

    print(type(configs))
    print(configs)

    exp_name = str(uuid.uuid4())[:8]
    mlflow.set_tracking_uri('http://mlflow_server:5000')
    exp = mlflow.create_experiment(f"sklearn_classification_{exp_name}")
    mlflow.set_experiment(exp)

    for idx, cfg in enumerate(configs):
        classifier_name = cfg["classifier_name"]
        params = cfg["kwargs"]

        module_name, class_name = classifier_name.rsplit(".", 1)

        module = importlib.import_module(module_name)
        classifier_class = getattr(module, class_name)

        classifier = classifier_class(**params)

        data = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

        with mlflow.start_run(experiment_id=exp):
            classifier.fit(X_train, y_train)
            
            y_pred = classifier.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            
            mlflow.log_param("classifier", classifier_name)
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("precision", precision)
            mlflow.sklearn.log_model(classifier, classifier_name.lower())
            
            print(f"Model: {classifier_name}, Accuracy: {accuracy}")

            model_info = mlflow.sklearn.log_model(sk_model=classifier, artifact_path=classifier_name)
            mlflow.register_model(model_uri=model_info.model_uri, name=classifier_name)

        print("Training complete. Logged to MLflow.")


dag = DAG(
    dag_id='mlflow_model_training',
    schedule_interval=None,
    start_date=datetime.now(),
    catchup=False,
)

wait_for_new_file = FileSensor(
    task_id='wait_for_new_file',
    poke_interval=10,
    filepath='./data/input_data',
    fs_conn_id=FS_CONN_ID,
    dag=dag,
)

load_config = PythonOperator(
    task_id='load_config',
    python_callable=load_config_from_json,
    dag=dag,
)

train_model = PythonOperator(
    task_id='train_model',
    python_callable=train_model_mlflow,
    dag=dag,
)

wait_for_new_file >> load_config >> train_model