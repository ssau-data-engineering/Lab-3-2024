import os, datetime, pandas as pd, mlflow, yaml
from airflow import DAG
from airflow.operators.python import PythonOperator
from sklearn.metrics import f1_score


os.environ.update({
    "AWS_ACCESS_KEY_ID": "minio",
    "AWS_SECRET_ACCESS_KEY": "minio123",
    "MLFLOW_S3_ENDPOINT_URL": "http://minio:9000"
})
current_dir = os.path.dirname(os.path.abspath(__file__))
path_to_val = os.path.join(current_dir, '..', 'data', 'val')
path_to_data = os.path.join(current_dir, '..', 'data', 'dataset')
path_to_input = os.path.join(current_dir, '..', 'data', 'config')
config_file = 'config.yaml'


def find_best_model():
    x_val, y_val = pd.read_csv(f"{path_to_val}/x.csv"), pd.read_csv(f"{path_to_val}/y.csv")
    with open(f"{path_to_input}/{config_file}", 'r') as file: config = yaml.safe_load(file)

    mlflow.set_tracking_uri('http://mlflow_server:5000')
    f1_set = {conf['classifier']:
        f1_score(
            y_val,
            mlflow.sklearn.load_model(f"models:/milk_{conf['classifier']}/None").predict(x_val),
            average='micro')
        for conf in config['configs']}

    best_model_name = max(f1_set, key=f1_set.get)
    client, filter_string = mlflow.MlflowClient(), f"name='milk_{best_model_name}'"
    best_model, all_versions = client.search_registered_models(filter_string)[0], client.search_model_versions(filter_string)
    client.transition_model_version_stage(
        name=best_model.name,
        version=all_versions[0].version,
        stage='Production'
    )

dag = DAG(
    dag_id='lab3_2',
    dagrun_timeout=datetime.timedelta(minutes=40),
    start_date=datetime.datetime(2024, 1, 1),
    catchup=False,
    description='DAG for training models',
    schedule_interval=None
)

validate = PythonOperator(
    task_id='best_model',
    python_callable=find_best_model,
    dag=dag
)

validate
