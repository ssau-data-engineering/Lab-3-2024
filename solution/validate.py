import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient 

tracking_url = 'http://mlflow_server:5000'
data_path = '/opt/airflow/data'

def get_data(data_path:str, labels:str):
    array_data = np.asarray(pd.read_csv(f'{data_path}/x_{labels}.csv'), dtype=np.float32)
    array_target = pd.read_csv(f"{data_path}/y_{labels}.csv")
    return array_data, array_target

with open(f'{data_path}/mlflow_experiment_id.txt', 'r') as f:
   id_current_experiment = f.read()

mlflow.set_tracking_uri(tracking_url)
mlflow.set_experiment(id_current_experiment)

x_val, y_val = get_data(data_path, 'val')
list_models_for_validate = {}

with mlflow.start_run(run_name = "Production model") as start_run:
    models_file = pd.read_csv(f"{data_path}/models.csv", header=None)
    for model_Info in models_file.iterrows():
        name = model_Info[1][1]
        uri = model_Info[1][2]
        list_models_for_validate[name + " " + uri] = mlflow.sklearn.load_model(uri)

    current_results = {}
    for name, model in list_models_for_validate.items():
        prediction = model.predict(x_val)
        current_results[name] = accuracy_score(y_val, prediction)

    best_model_in_list_validate_model = max(current_results, key=current_results.get)

client= MlflowClient()
version = client.search_model_versions(f"name='{best_model_in_list_validate_model.split(' ')[0]}' and run_id='{best_model_in_list_validate_model.split(' ')[1].split('/')[1]}'")[0].version
client.transition_model_version_stage(name=best_model_in_list_validate_model.split(' ')[0], version=version, stage="Production")
