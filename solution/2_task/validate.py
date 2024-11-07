#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

tracking_url = 'http://mlflow_server:5000'
data_path = '/opt/airflow/data/lr3'

# Чтение ID текущего эксперимента
with open(f'{data_path}/experiment_id.txt', 'r') as f:
    id_current_experiment = f.read()

mlflow.set_tracking_uri(tracking_url)
mlflow.set_experiment(id_current_experiment)

# Загрузка валидационных данных
x_val = pd.read_csv(f'{data_path}/x_val.csv')
y_val = pd.read_csv(f'{data_path}/y_val.csv')

list_models_for_validate = {}

with mlflow.start_run(run_name="Production model") as start_run:
    models_file = pd.read_csv(f"{data_path}/models.csv", header=0)
    for model_Info in models_file.iterrows():
        name = model_Info[1][1]
        uri = model_Info[1][2]
        print(f"Loading model: {name} from URI: {uri}")  # Логируем информацию о моделях
        list_models_for_validate[name + " " + uri] = mlflow.sklearn.load_model(uri)

    current_results = {}
    for name, model in list_models_for_validate.items():
        prediction = model.predict(x_val)
        current_results[name] = accuracy_score(y_val, prediction)

    # Находим лучшую модель
    best_model_in_list_validate_model = max(current_results, key=current_results.get)
    model_name, model_uri = best_model_in_list_validate_model.split(" ")

    # Сохранение лучшей модели в CSV
    best_model_data = pd.DataFrame({
        "name": [model_name],
        "uri": [model_uri],
        "accuracy": [current_results[best_model_in_list_validate_model]]
    })

    # Проверяем, существует ли уже файл с лучшими моделями
    try:
        best_models_csv = pd.read_csv(f"{data_path}/best_models.csv")
        # Добавляем новую строку с лучшей моделью
        best_model_data.to_csv(f"{data_path}/best_models.csv", mode='a', header=False, index=False)
    except FileNotFoundError:
        # Если файл не существует, создаем его с заголовками
        best_model_data.to_csv(f"{data_path}/best_models.csv", index=False)

    print(f"Best model saved: {model_name} with URI: {model_uri}")
