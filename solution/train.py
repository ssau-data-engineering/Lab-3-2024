import json
import importlib
import random as rnd
import pandas as pd
import helps as hlp
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

data_path = '/opt/airflow/data'

with open(f'{data_path}/conf.json', 'r') as config_file:
   config_data = json.load(config_file)

def generate_experiment_id(name_file:str):
    sources_string = '1234567890AaBbCcDdEeFfGgHhIiJjKkLlMm1234567890NnOoPpQqRrSsTtUuVvWwXxYyZz1234567890'
    list_str = []
    for i in range(19):
        list_str.append(sources_string[rnd.randint(0, len(sources_string)-1)])
    result_str = ''.join(list_str)
    f = open(f'{data_path}/{name_file}', 'w')
    f.write(result_str)
    f.close()
    return result_str

def logirovanie(cuerrnt_configs, y_test_dataset, current_prediction):
    mlflow.log_params(cuerrnt_configs)
    mlflow.log_metrics({"f1": f1_score(y_test_dataset, current_prediction, average='weighted')})
    mlflow.log_metrics({'acc': accuracy_score(y_test_dataset, current_prediction)})
    mlflow.log_metrics({'precision':precision_score(y_test_dataset, current_prediction, average='weighted')})
    mlflow.log_metrics({'recall': recall_score(y_test_dataset, current_prediction, average='weighted')})

x_train , y_train = hlp.get_data(data_path, 'train')
x_test, y_test = hlp.get_data(data_path, 'test')

mlflow.set_tracking_uri(hlp.tracking_url)
id_current_experiment = generate_experiment_id('mlflow_experiment_id.txt')
exp_id = mlflow.create_experiment(id_current_experiment)
mlflow.set_experiment(exp_id)

for i, config in enumerate(config_data['configs']):
    mlflow.start_run(run_name = config['classificator'], experiment_id = exp_id)

    module = importlib.import_module(config['module'])
    classificator = getattr(module, config['classificator'])
    model = classificator(**config['args'])

    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    signature = infer_signature(x_test, prediction)

    logirovanie(config['args'],y_test, prediction)

    modelInfo = mlflow.sklearn.log_model(
        sk_model=model, 
        artifact_path=config['module'], 
        signature=signature, 
        registered_model_name=config['classificator'])

    df = pd.DataFrame({"name":config['classificator'], "uri":modelInfo.model_uri}, index=[i])
    df.to_csv(f'{data_path}/models.csv', mode='a', header=False)
    mlflow.end_run()