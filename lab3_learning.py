from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import yaml
import mlflow
import uuid
import pickle
import importlib
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

current_dir = os.path.dirname(os.path.abspath(__file__))
path_to_input = os.path.join(current_dir, '..', 'data', 'config')
path_to_val = os.path.join(current_dir, '..', 'data', 'validation')
path_to_data = os.path.join(current_dir, '..', 'data', 'dataset')

def train_model_from_file():

    df = pd.read_csv(os.path.join(path_to_data, 'parkinsons.data'))
    
    X = df.loc[:,df.columns!='status'].values[:,1:]
    y = df.loc[:,'status'].values

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=7)

    pd.DataFrame(x_val, columns=df.columns.to_list().remove('status')).to_csv(f'{path_to_val}/x_val.csv', index=False)
    pd.DataFrame(y_val, columns=['status']).to_csv(f'{path_to_val}/y_val.csv', index=False)

    with open(os.path.join(path_to_input, 'config.yaml'), 'r') as file:
        config = yaml.safe_load(file)

    mlflow.set_tracking_uri('http://mlflow_server:5000')
    experiment_id = uuid.uuid4()
    experiment = mlflow.create_experiment(str(experiment_id))
    mlflow.set_experiment(experiment)
    
    for _, conf in enumerate(config['configs']):
        mlflow.start_run(run_name=conf['classificator'], experiment_id=experiment)

        module = importlib.import_module('.'.join(conf['classificator'].split('.')[:-1]))
        classificator_class = getattr(module, conf['classificator'].split('.')[-1])
        model = classificator_class(**conf['kwargs'])

        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        mlflow.log_metrics({
            "f1_score": f1_score(y_test, y_pred),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred)
        })

        model_info = mlflow.sklearn.log_model(sk_model=model, artifact_path=conf['classificator'])
        model_uri = model_info.model_uri
        registered_model = mlflow.register_model(model_uri=model_uri, name=conf['classificator'])

        mlflow.end_run()


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 11, 7),
    'retries': 1,
}

dag = DAG(
    'training_from_config',
    default_args=default_args,
    description='DAG for training models',
    schedule_interval=None,
)

wait_for_new_file = FileSensor(
    task_id='wait_for_new_file',
    poke_interval=10,
    filepath=path_to_input,
    fs_conn_id='lab_connect',
    dag=dag,
)

train_model = PythonOperator(
        task_id='read_files',
        python_callable=train_model_from_file,
        dag=dag,
    )


wait_for_new_file >> train_model