import os
import yaml
import mlflow
import glob
import pandas as pd
from airflow import DAG
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from airflow.sensors.filesystem import FileSensor
from airflow.operators.python import PythonOperator
from sklearn.model_selection import train_test_split
from mlflow.exceptions import MlflowException
from sklearn.metrics import f1_score 

BASE_PATH = '/opt/airflow/data/lab_3'
DATA_PATH = os.path.join(BASE_PATH, 'data', 'bank.csv')
YAML_DIR = os.path.join(BASE_PATH, 'config')

os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

# Обработка данных
def prepare_data(file_path, target_column='y', test_size=0.3, random_state=42):
    df = pd.read_csv(file_path, sep=';')
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )

    return X_train, X_test, y_train, y_test

def get_yaml_files():
    yaml_pattern = os.path.join(YAML_DIR, '*.yaml')
    return glob.glob(yaml_pattern)

def process_single_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

def get_model_name_from_yaml(yaml_path):
    base_name = os.path.basename(yaml_path)
    model_name = base_name.replace('experiment_', '').replace('.yaml', '')
    return model_name

# Обработка экспериментов
def setup_experiment(experiment_name):
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        elif experiment.lifecycle_stage == 'deleted':
            mlflow.tracking.MlflowClient().restore_experiment(experiment.experiment_id)
            experiment_id = experiment.experiment_id
        else:
            experiment_id = experiment.experiment_id

        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string="",
        )
        for run in runs:
            client.delete_run(run.info.run_id)
        
        mlflow.set_experiment(experiment_name)
        return experiment_id, experiment_name

# Обучение
def train_model_function(**context):
    yaml_files = get_yaml_files()
    
    mlflow.set_tracking_uri('http://mlflow_server:5000')
    client = mlflow.tracking.MlflowClient()
    
    for yaml_path in yaml_files:
        model_name = get_model_name_from_yaml(yaml_path)
        base_experiment_name = f"{model_name}"
        
        experiment_id, actual_experiment_name = setup_experiment(base_experiment_name)
        
        config = process_single_yaml(yaml_path)
        
        model_name = config.get("classificator")
        model_kwargs = {param_name: param_value for param in config.get("kwargs", []) 
                       for param_name, param_value in param.items()}
        
        components = model_name.split(".")
        module_name = ".".join(components[:-1])
        class_name = components[-1]
        model_module = __import__(module_name, fromlist=[class_name])
        model_class = getattr(model_module, class_name)

        X_train, X_test, y_train, y_test = prepare_data(
            file_path=DATA_PATH,
            target_column='y',
            test_size=0.2,
            random_state=42
        )

        run_name = f"train_{base_experiment_name}"

        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
            mlflow.log_params(model_kwargs)
            mlflow.log_param("target_column", 'y')
            mlflow.log_param("test_size", 0.3)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("config_file", os.path.basename(yaml_path))

            model = model_class(**model_kwargs)
            model.fit(X_train, y_train)

            mlflow.sklearn.log_model(model, "model")

            # Изменение расчета метрик с accuracy на f1-score
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_f1 = f1_score(y_train, y_train_pred)
            test_f1 = f1_score(y_test, y_test_pred)

            mlflow.log_metric("train_f1", train_f1)
            mlflow.log_metric("test_f1", test_f1)

            print(f'Model {module_name} trained with F1-score {train_f1:.4f} on train and {test_f1:.4f} on test')

            model_uri = f"runs:/{run.info.run_id}/model"
            registered_name = f"model_{get_model_name_from_yaml(yaml_path)}"
            
            try:
                model_details = client.get_registered_model(registered_name)
                mlflow.register_model(model_uri, registered_name)
            except MlflowException:
                mlflow.register_model(model_uri, registered_name)

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 11, 8),
    'retries': 1,
}

dag = DAG(
    'data_engineering_lab_3',
    default_args=default_args,
    description='DAG for data engineering lab 3: training classification models',
    schedule_interval=None,
)

yaml_file_sensor = FileSensor(
    task_id='yaml_file_sensor',
    poke_interval=10,
    filepath=os.path.join(YAML_DIR, '*.yaml'),
    fs_conn_id='vas_lab',
    dag=dag,
)

train_model = PythonOperator(
    task_id='train_model',
    python_callable=train_model_function,
    provide_context=True,
    dag=dag,
)

yaml_file_sensor >> train_model