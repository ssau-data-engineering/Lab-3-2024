import datetime
from math import inf
from airflow import DAG
from airflow.operators.python import PythonOperator
import os
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score

os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"


def get_best_model():
    file_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'LR_3_task1')
    path_input_x_val = os.path.join(file_directory, "x_valid.npy")
    path_input_y_val = os.path.join(file_directory, "y_valid.npy")
    x_valid = np.load(path_input_x_val)
    y_valid = np.load(path_input_y_val)
     
    url = 'http://mlflow_server:5000'
    mlflow.set_tracking_uri(url)
    experiment_name = "LR_3"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

    client = MlflowClient()
    runs = client.search_runs(experiment_ids=[experiment_id])

    best_accuracy = -inf
    best_model_uri = None 
    for run in runs:
        model_uri = f"runs:/{run.info.run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        y_valid_pred = model.predict(x_valid)
        accuracy_val = accuracy_score(y_valid, y_valid_pred)
        print(f"Model {model} - accuracy_val = {accuracy_val}")
        if accuracy_val > best_accuracy:
            best_accuracy = accuracy_val
            best_model_uri = model_uri
    
    for registered_model in client.search_registered_models():
        for version in registered_model.latest_versions:
            if f"runs:/{version.run_id}/model" == best_model_uri:
                 model_name = registered_model.name  
                 version_to_transition = version.version

                 client.transition_model_version_stage(
                    name=model_name,
                    version=version_to_transition,
                    stage='Production'
                )
                 print(f"Best model {model_name} - accuracy_val = {best_accuracy} uri = {best_model_uri}")
                 break


dag = DAG(
    dag_id="LR_3_task_2",
    schedule="0 0 * * *", 
    start_date=datetime.datetime(2024, 11, 9, tzinfo=datetime.timezone.utc), 
    catchup=False,  
    dagrun_timeout=datetime.timedelta(minutes=60),  
    tags=["LR"],
)


get_best_model = PythonOperator(
        task_id='get_best_model',  
        python_callable=get_best_model,  
        dag = dag
)

get_best_model