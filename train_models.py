import os
import uuid
import yaml
import mlflow

from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def train():
    x, y = load_breast_cancer(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

    classifiers = {
        'sklearn.linear_model.LogisticRegression': LogisticRegression,
        'sklearn.neighbors.KNeighborsClassifier': KNeighborsClassifier,
        'sklearn.naive_bayes.GaussianNB': GaussianNB,
        'sklearn.tree.DecisionTreeClassifier': DecisionTreeClassifier,
        'sklearn.discriminant_analysis.LinearDiscriminantAnalysis': LinearDiscriminantAnalysis,
        'sklearn.ensemble.RandomForestClassifier': RandomForestClassifier
    }

    mlflow.set_tracking_uri('http://mlflow_server:5000')
    mlflow.set_experiment(mlflow.create_experiment(str(uuid.uuid4())))

    with open('data/config.yaml') as file:
        configs = yaml.safe_load(file.read())["configs"]

    for config in configs:
        mlflow.start_run(run_name=f"breast_cancer_{config['classifier']}")

        classifier = classifiers[config['classifier']](**config['params'])

        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        mlflow.log_metrics({
            "f1_score": f1_score(y_test, y_pred, average='micro'),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='micro'),
            "recall": recall_score(y_test, y_pred, average='micro')
        })

        model_info = mlflow.sklearn.log_model(
            sk_model=classifier,
            artifact_path=f"breast_cancer_{config['classifier']}"
        )
        mlflow.register_model(
            model_uri=model_info.model_uri,
            name=f"breast_cancer_{config['classifier']}"
        )
        mlflow.end_run()


os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

dag = DAG(
    'train_classifiers',
    default_args=default_args,
    description='DAG for training classifier and getting mlflow logs',
    schedule_interval=None,
)

wait_for_new_file = FileSensor(
    task_id='wait_for_new_file',
    poke_interval=10,
    filepath='data',
    fs_conn_id='airflow',
    dag=dag
)

train = PythonOperator(
    task_id='train',
    python_callable=train,
    dag=dag
)

wait_for_new_file >> train
