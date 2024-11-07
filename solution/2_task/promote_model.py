#!/usr/bin/env python3
import pandas as pd
import mlflow
from mlflow import MlflowClient
import subprocess
import requests
import time

tracking_url = 'http://mlflow_server:5000'
data_path = '/opt/airflow/data/lr3'

mlflow.set_tracking_uri(tracking_url)
client = MlflowClient()

models = client.search_registered_models()
for model in models:
    print(model.name)

# Чтение последней строки из файла CSV с лучшими моделями
best_model_df = pd.read_csv(f"{data_path}/best_models.csv")
best_model_info = best_model_df.iloc[-1]  # Берем последнюю строку

model_name = best_model_info['name']
model_uri = best_model_info['uri']

# Извлечение run_id из model_uri
run_id = model_uri.split('/')[1]

# Получение версии модели
print(f"Поиск модели с именем: {model_name} и run_id: {run_id}")
results = client.search_model_versions(f"name='{model_name}' and run_id='{run_id}'")
if not results:
    print(f"No model versions found for name: {model_name} and run_id: {run_id}")
else:
    version = results[0].version
    client.transition_model_version_stage(name=model_name, version=version, stage="Production")
    print(f"Model {model_name} version {version} transitioned to Production.")
    model_version = f"models:/{model_name}/{version}"
    # Команда для запуска хостинга модели через mlflow serve
    host_command = f"mlflow models serve -m {model_version} --host 0.0.0.0 --port 5001"
    print(f"Запуск хостинга модели командой: {host_command}")
    
    
    
    # Запуск команды для хостинга модели
    return_code = subprocess.call(host_command, shell=True)
    print(f"Return code: {return_code}")
    
    time.sleep(5)
    
    try:
        # Отправка POST-запроса для проверки
        response = requests.post(f'http://localhost/invocations',
                                 json={
                                        "columns": ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"],
                                        "data": [
                                            [1, 0, 58.0, 0, 0, 26.55, 2],
                                            [2, 0, 23.0, 0, 0, 13.7917, 0],
                                            [3, 1, 23.0, 0, 0, 7.8542, 2]
                                        ]
                                        })
        print(f"Response from model: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
        