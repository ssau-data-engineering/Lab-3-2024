import os
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import yaml
import json
import glob
import mlflow
import mlflow.sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import importlib
from airflow.utils.dates import days_ago

# Настройка переменных среды для MLflow и Minio
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

# Параметры DAG
default_args = {
    'owner': 'snail',
    'start_date': days_ago(0, 0, 0, 0),
    'retries': 1,
}

dag = DAG(
    'train_classifier_dag',
    default_args=default_args,
    schedule_interval=None,
)

# Пути к директориям
CONFIG_DIR = '/opt/airflow/config'  # Папка с конфигурационными файлами



def monitor_configurations():
    # Поиск новых конфигурационных файлов
    config_files = glob.glob(f"{CONFIG_DIR}/*.yaml") + glob.glob(f"{CONFIG_DIR}/*.json")
    print(f"Found config files: {config_files}")
    return config_files


def train_model(**context):
    config_files = context['task_instance'].xcom_pull(task_ids='monitor_configurations')

    # Проверка на наличие конфигурационных файлов
    if not config_files:
        print("No configuration files found. Skipping model training.")
        return  # Если нет файлов, выход из функции

    for config_file in config_files:
        # Чтение конфигурационного файла
        if config_file.endswith('.yaml'):
            with open(config_file) as f:
                config = yaml.safe_load(f)
        elif config_file.endswith('.json'):
            with open(config_file) as f:
                config = json.load(f)
        else:
            raise ValueError("Unsupported file format")

        # Извлечение параметров классификатора
        classifier_path = config['classificator']
        kwargs = config.get('kwargs', {})

        # Динамический импорт классификатора
        module_name, class_name = classifier_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        Classifier = getattr(module, class_name)
        kwargs = config.get('kwargs', {})

        if isinstance(kwargs, list):
            # Объединяем все словари из списка в один
            kwargs = {key: value for d in kwargs for key, value in d.items()}

        classifier = Classifier(**kwargs)



        # Загрузка данных (например, iris dataset)
        iris = datasets.load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.2, random_state=42
        )

        experiment_name = "Classifier Training - " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Логирование в MLflow
        mlflow.set_tracking_uri("http://web:5000")
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            # Логирование параметров модели
            mlflow.log_params(kwargs)

            # Обучение модели
            classifier.fit(X_train, y_train)

            # Логирование модели
            mlflow.sklearn.log_model(classifier, "model")

            # Тестирование модели
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Логирование метрики
            mlflow.log_metric("accuracy", accuracy)

            # Регистрация модели
            model_name = f"{class_name}Model"
            result = mlflow.register_model(
                "runs:/{run_id}/model".format(run_id=mlflow.active_run().info.run_id),
                model_name
            )




monitor_task = PythonOperator(
    task_id='monitor_configurations',
    python_callable=monitor_configurations,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    provide_context=True,
    dag=dag,
)

monitor_task >> train_task
