import os
import glob
import yaml
import pandas as pd
from airflow import DAG
from datetime import datetime
from airflow.sensors.filesystem import FileSensor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import mlflow
from datetime import datetime
from airflow.operators.python import PythonOperator
from mlflow.exceptions import MlflowException

os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

def get_models_and_data_from_yaml(yaml_path):
    yaml_files = glob.glob(os.path.join(yaml_path, '*.yaml'))
    models_and_data = []

    for file_path in yaml_files:
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        model_name = os.path.basename(file_path).replace('experiment_', '').replace('.yaml', '')
        models_and_data.append((model_name, data))
    
    return models_and_data

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

def train_model_function(**kwargs):
    
    TRAIN_PATH = '/opt/airflow/data/Marina_lab_3/data/train_5.csv'
    TEST_PATH = '/opt/airflow/data/Marina_lab_3/data/test_5.csv'
    yaml_path = '/opt/airflow/data/Marina_lab_3/config'
    
    train_data = pd.read_csv(TRAIN_PATH)
    #test_data = pd.read_csv(TEST_PATH)
    
    # Получаем данные моделей и их конфигурации из YAML-файлов
    models_and_data = get_models_and_data_from_yaml(yaml_path)

    mlflow.set_tracking_uri('http://mlflow_server:5000')
    client = mlflow.tracking.MlflowClient()

    for model_name, config in models_and_data:
        # Настройка эксперимента
        experiment_id, experiment_name = setup_experiment(model_name)

        # Считывание параметров модели
        model_class_path = config.get("classificator")
        model_kwargs = {param_name: param_value for param in config.get("kwargs", []) 
                        for param_name, param_value in param.items()}

        # Импорт класса модели
        components = model_class_path.split(".")
        module_name = ".".join(components[:-1])
        class_name = components[-1]
        model_module = __import__(module_name, fromlist=[class_name])
        model_class = getattr(model_module, class_name)

        X_train, X_test, y_train, y_test = train_test_split(
            train_data.drop(['target', 'smpl'], axis=1), 
            train_data['target'], 
            test_size=0.3, stratify=train_data['target'], random_state=42
        )

        run_name = f"train_{model_name}"

        # Запуск эксперимента в MLflow
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
            mlflow.log_params(model_kwargs)
            mlflow.log_param("target_column", 'y')
            mlflow.log_param("test_size", 0.3)
            mlflow.log_param("random_state", 42)

            model = model_class(**model_kwargs)
            model.fit(X_train, y_train)

            # Сохранение модели в MLflow
            mlflow.sklearn.log_model(model, "model")

            # Вычисление и логирование метрик
            y_val_pred = model.predict_proba(X_train)[:, 1]
            y_test_pred = model.predict_proba(X_test)[:, 1]
            train_roc_auc = roc_auc_score(y_train, y_val_pred)
            test_roc_auc = roc_auc_score(y_test, y_test_pred)

            mlflow.log_metric("roc_auc_train", train_roc_auc)
            mlflow.log_metric("roc_auc_test", test_roc_auc)

            print(f"Model {model_name} trained with ROC-AUC {train_roc_auc:.4f} (train sample) and {test_roc_auc:.4f} (test sample)")
        
            # Регистрация модели
            model_uri = f"runs:/{run.info.run_id}/model"
            registered_name = f"model_{model_name}"
            
            try:
                mlflow.register_model(model_uri, registered_name)
            except MlflowException as e:
                print(f"Failed to register model {registered_name}: {e}")


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

dag = DAG(
    'MLFlow',
    default_args=default_args,
    description='DAG for training classificated models with the help MLFLOW',
    schedule_interval=None,
)

wait_for_new_file = FileSensor(
    task_id='wait_for_new_file',
    poke_interval=10,  # Interval to check for new files (in seconds)
    filepath='/opt/airflow/data/Marina_lab_3/data',  # Target folder to monitor
    fs_conn_id='manch_lab', # Check FAQ for info
    dag=dag,
)

train_model = PythonOperator(
    task_id='train_model',
    python_callable=train_model_function,
    dag=dag,
)

wait_for_new_file >> train_model