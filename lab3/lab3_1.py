import os, datetime, pandas as pd, yaml, mlflow, uuid, importlib
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


os.environ.update({
    "AWS_ACCESS_KEY_ID": "minio",
    "AWS_SECRET_ACCESS_KEY": "minio123",
    "MLFLOW_S3_ENDPOINT_URL": "http://minio:9000"
})
current_dir = os.path.dirname(os.path.abspath(__file__))
path_to_input = os.path.join(current_dir, '..', 'data', 'config')
path_to_val = os.path.join(current_dir, '..', 'data', 'val')
path_to_data = os.path.join(current_dir, '..', 'data', 'dataset')
input_csv, config_file = 'milknew.csv', 'config.yaml'


def train():
    df, col_name = pd.read_csv(f"{path_to_data}/{input_csv}"), 'Grade'
    x, y =  df.loc[:,df.columns!=col_name].values[:,1:], df.loc[:,col_name].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
    pd.DataFrame(x_val,
                 columns=df.columns.to_list().remove(col_name)).to_csv(f'{path_to_val}/x.csv', index=False)
    pd.DataFrame(y_val,
                 columns=[col_name]).to_csv(f'{path_to_val}/y.csv', index=False)
    with open(f"{path_to_input}/{config_file}", 'r') as file: config = yaml.safe_load(file)

    mlflow.set_tracking_uri('http://mlflow_server:5000')
    mlflow.set_experiment(mlflow.create_experiment(str(uuid.uuid4())))

    for conf in config['configs']:
        mlflow.start_run(run_name=conf['classifier'])
        model = getattr(importlib.import_module('.'.join(conf['classifier'].split('.')[:-1])),
                        conf['classifier'].split('.')[-1])(**conf['kwargs'])
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        mlflow.log_metrics({
            "f1_score": f1_score(y_test, y_pred, average='micro'),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='micro'),
            "recall": recall_score(y_test, y_pred, average='micro')
        })
        model_info = mlflow.sklearn.log_model(sk_model=model,
                                              artifact_path='_'.join(['milk', conf['classifier']]))
        mlflow.register_model(model_uri=model_info.model_uri,
                              name='_'.join(['milk', conf['classifier']]))
        mlflow.end_run()

dag = DAG(
    dag_id='lab3_1',
    dagrun_timeout=datetime.timedelta(minutes=40),
    start_date=datetime.datetime(2024, 1, 1),
    catchup=False,
    description='DAG for training models',
    schedule_interval=None
)

wait_for_new_file = FileSensor(
    task_id='wait_for_new_file',
    poke_interval=10,
    filepath=path_to_input,
    fs_conn_id='lab',
    dag=dag
)
train = PythonOperator(
    task_id='train',
    python_callable=train,
    dag=dag
)

wait_for_new_file >> train
