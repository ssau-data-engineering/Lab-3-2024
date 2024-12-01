import yaml
import mlflow
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

def get_config(path: str = "config.yaml"):
    with open(path) as file:
        config = yaml.safe_load(file.read())
    return config

def main():
    x, y = load_iris(return_X_y=True)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=123)

    mlflow.set_tracking_uri('http://mlflow_server:5000')
    f1 = {}
    for conf in get_config("data/config.yaml")['configs']:
        f1[conf['classificator']] = f1_score(
            y_val,
            mlflow.sklearn.load_model(f"models:/iris_{conf['classificator']}/None").predict(x_val),
            average='micro'
        )

    best_model_name = max(f1, key=f1.get)
    client, filter_string = mlflow.MlflowClient(), f"name='iris_{best_model_name}'"
    best_model = client.search_registered_models(filter_string)[0]
    all_versions = client.search_model_versions(filter_string)
    client.transition_model_version_stage(
        name=best_model.name,
        version=all_versions[0].version,
        stage='Production'
    )