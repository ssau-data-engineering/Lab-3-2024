from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import yaml
import mlflow
import uuid
import importlib
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
os.environ["GIT_PYTHON_REFRESH"] = "quiet" #чтобы убрать ошибку с гитом

#Чтобы Airflow смог сохранить модель - необходимо в файле описывающем DAG установить переменные среды
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

#Патчи к конфигурациям и данным
current_dir = os.path.dirname(os.path.abspath(__file__))
path_to_in = os.path.join(current_dir, '..', 'data', 'config')
path_to_val = os.path.join(current_dir, '..', 'data', 'validation')
path_to_data = os.path.join(current_dir, '..', 'data', 'dataset')

def train_models():
    data = pd.read_csv(os.path.join(path_to_data, 'Housing.csv'))
    #Кодируем категориальные значения
    label_encoder = LabelEncoder()
    data['mainroad'] = label_encoder.fit_transform(data['mainroad'])
    data['guestroom'] = label_encoder.fit_transform(data['guestroom'])
    data['basement'] = label_encoder.fit_transform(data['basement'])
    data['hotwaterheating'] = label_encoder.fit_transform(data['hotwaterheating'])
    data['airconditioning'] = label_encoder.fit_transform(data['airconditioning'])
    data['prefarea'] = label_encoder.fit_transform(data['prefarea'])
    data['furnishingstatus'] = label_encoder.fit_transform(data['furnishingstatus'])
    data['stories'] = label_encoder.fit_transform(data['stories'])

    #Разделяем данные на X и y
    X = data.drop(columns=['bedrooms'])
    y = data['bedrooms']

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=7)
    #Сохраняем валидационные данные
    pd.DataFrame(x_val, columns=data.columns.to_list().remove('bedrooms')).to_csv(f'{path_to_val}/x_val.csv', index=False)
    pd.DataFrame(y_val, columns=['bedrooms']).to_csv(f'{path_to_val}/y_val.csv', index=False)

    #Читаем конфигурационный файл
    with open(os.path.join(path_to_in, 'config.yaml'), 'r') as file:
        config = yaml.safe_load(file)

    #Настраиваем MLFLOW
    mlflow.set_tracking_uri('http://mlflow_server:5000')
    experiment_id = uuid.uuid4()
    experiment = mlflow.create_experiment(str(experiment_id))
    mlflow.set_experiment(experiment)

    #Обучаем модели по конфигурации
    for _, conf in enumerate(config['configs']):
        mlflow.start_run(run_name=conf['classificator'], experiment_id=experiment)
        #Динамический импорт
        module = importlib.import_module('.'.join(conf['classificator'].split('.')[:-1]))
        classificator_class = getattr(module, conf['classificator'].split('.')[-1])
        model = classificator_class(**conf['kwargs'])
        #Обучение
        model.fit(x_train, y_train)
        #Предсказание
        y_pred = model.predict(x_test)
        #мЕТРИКИ
        mlflow.log_metrics({
            "f1_score": f1_score(y_test, y_pred, average='micro'),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='micro'),
            "recall": recall_score(y_test, y_pred, average='micro')
        })
        #Сохраняем модели в MLflow Model Registry
        model_info = mlflow.sklearn.log_model(sk_model=model, artifact_path=conf['classificator'])
        model_uri = model_info.model_uri
        registered_model = mlflow.register_model(model_uri=model_uri, name=conf['classificator'])

        mlflow.end_run()

#DAG для Airflow
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 11, 30),
    'retries': 1,
}

dag = DAG(
    'data_engineering_lab_3_1',
    default_args=default_args,
    description='DAG for data engineering lab 3',
    schedule_interval=None,
)

wait_for_new_file = FileSensor(
    task_id='wait_for_new_file',
    poke_interval=10,  # Interval to check for new files (in seconds)
    filepath=path_to_in,  # Target folder to monitor
    fs_conn_id='Conn_lab2', # Check FAQ for info
    dag=dag,
)

train_model = PythonOperator(
    task_id='train_model',
    python_callable=train_models,
    dag=dag,
)


wait_for_new_file >> train_model