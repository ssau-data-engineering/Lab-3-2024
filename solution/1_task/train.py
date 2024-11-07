#!/usr/bin/env python3
import json
import importlib
import random as rnd
import pandas as pd
import numpy as np
import uuid
import mlflow
import mlflow.sklearn
import predobr as pred
from mlflow.models import infer_signature
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


# URL для подключения к MLflow Tracking Server
tracking_url = 'http://mlflow_server:5000'

# Получаем путь к конфигурационному файлу
config_file_path = sys.argv[1]

# Загрузка конфигурации из JSON файла
with open(config_file_path, 'r') as config_file:
        config_data = json.load(config_file)

# Извлекаем информацию о наборе данных из JSON
dataset_name = config_data['dataset_info']['name']
data_path = config_data['dataset_info']['data_path']
drop_columns = config_data['dataset_info']['drop_columns']
categorical_columns = config_data['dataset_info']['categorical_columns']

# Загрузка данных
x_data, y_data = pred.get_data(data_path, dataset_name)

# Применяем предобработку к данным
# Применяем предобработку к данным
x_data, y_data, encoders = pred.preprocess_data(pd.DataFrame(x_data), pd.DataFrame(y_data), drop_columns, categorical_columns)

# Разделение данных на тренировочные и тестовые выборки
x_train, x_test, y_train, y_test = pred.split_data(x_data, y_data)
x_val, x_temp, y_val, y_temp = pred.split_data(x_data, y_data, test_size=0.5, random_state = 17)

# Сохранение валидационных данных в CSV это для задания 2
x_val.to_csv(f'{data_path}/x_val.csv', index=False)
y_val.to_csv(f'{data_path}/y_val.csv', index=False)

# Установка URI для подключения к MLflow Tracking Server
mlflow.set_tracking_uri(tracking_url)

# Генерация уникального ID эксперимента
id_current_experiment = str(uuid.uuid4())
exp_id = mlflow.create_experiment(id_current_experiment)
mlflow.set_experiment(exp_id)

# Сохранение ID эксперимента в файл так же надо для 2 задания
with open(f'{data_path}/experiment_id.txt', 'w') as f:
    f.write(id_current_experiment)

# Проходим по каждой конфигурации в JSON
for i, config in enumerate(config_data['configs']):
    mlflow.start_run(run_name=config['classificator'], experiment_id=exp_id)

    # Загрузка модуля и классификатора
    module = importlib.import_module(config['module'])
    classificator_class = getattr(module, config['classificator'])
    model = classificator_class(**config['args'])

    # Проверка на наличие метода partial_fit
    if hasattr(model, 'partial_fit'):
        max_iter = config['args'].get('max_iter', 50)
        for iter in range(max_iter):  
            model.partial_fit(x_train, y_train, classes=np.unique(y_train))
            y_pred = model.predict(x_test)
            mlflow.log_metrics({
                f"f1_score": f1_score(y_test, y_pred, average='weighted'),
                f"accuracy": accuracy_score(y_test, y_pred),
                f"precision": precision_score(y_test, y_pred, average='weighted'),
                f"recall": recall_score(y_test, y_pred, average='weighted')
            })
    else:
        model.fit(x_train, y_train)  # Стандартное обучение

    # Тестирование модели на тестовых данных
    y_pred_test = model.predict(x_test)
    # Логирование метрик на тестовых данных
    mlflow.log_metrics({
        "f1_score_test": f1_score(y_test, y_pred_test, average='weighted'),
        "accuracy_test": accuracy_score(y_test, y_pred_test),
        "precision_test": precision_score(y_test, y_pred_test, average='weighted'),
        "recall_test": recall_score(y_test, y_pred_test, average='weighted')
    })
    
    # Сохранение информации о модели в CSV
    model_info = mlflow.sklearn.log_model(sk_model=model, artifact_path=f'{config["module"]}/{config["classificator"]}')
    
    # Регистрация модели в Model Registry
    model_uri = model_info.model_uri
    registered_model = mlflow.register_model(model_uri=model_uri, name=config['classificator'])

    
    df = pd.DataFrame({"name": config['classificator'], "uri": model_info.model_uri}, index=[i])
    df.to_csv(f'{data_path}/models.csv', mode='a', header=False)  # Добавляем заголовки только при первой записи

    mlflow.end_run()