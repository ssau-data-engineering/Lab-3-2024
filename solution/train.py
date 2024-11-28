# train.py

import json
import importlib
import uuid  # Импортируем uuid
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import numpy as np

data_path = '/opt/airflow/data'

# Загрузка данных из CSV файла
df = pd.read_csv(f'{data_path}/iris.csv')
x = df.drop(columns=['variety']) 
y = df['variety']  

# Разделение данных
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=163)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, shuffle=True, random_state=163)

tracking_url = 'http://mlflow_server:5000'

# Чтение конфигурации из conf.json
with open(f'{data_path}/conf.json', 'r') as config_file:
    config_data = json.load(config_file)

def generate_experiment_id(name_file: str):
    result_str = str(uuid.uuid4())  # uuid4 генерирует случайный UUID
    with open(f'{data_path}/{name_file}', 'w') as f:
        f.write(result_str)
    return result_str

def logirovanie(current_configs, y_test_dataset, current_prediction):
    mlflow.log_params(current_configs)
    mlflow.log_metrics({
        "f1": f1_score(y_test_dataset, current_prediction, average='weighted'),
        'acc': accuracy_score(y_test_dataset, current_prediction),
        'precision': precision_score(y_test_dataset, current_prediction, average='weighted'),
        'recall': recall_score(y_test_dataset, current_prediction, average='weighted')
    })

mlflow.set_tracking_uri(tracking_url)
id_current_experiment = generate_experiment_id('mlflow_experiment_id.txt')
exp_id = mlflow.create_experiment(id_current_experiment)
mlflow.set_experiment(exp_id)

# Обучение моделей по конфигурации
for i, config in enumerate(config_data['configs']):
    mlflow.start_run(run_name=config['classificator'], experiment_id=exp_id)

    module = importlib.import_module(config['module'])
    classificator = getattr(module, config['classificator'])
    model = classificator(**config['args'])

    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    signature = infer_signature(x_test, prediction)

    logirovanie(config['args'], y_test, prediction)

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=config['module'],
        signature=signature,
        registered_model_name=config['classificator']
    )

    df_model = pd.DataFrame({"name": config['classificator'], "uri": model_info.model_uri}, index=[i])
    df_model.to_csv(f'{data_path}/models.csv', mode='a', header=False)
    mlflow.end_run()
