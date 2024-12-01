import uuid

import yaml
import mlflow
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def get_config(path: str = "config.yaml"):
    with open(path) as file:
        config = yaml.safe_load(file.read())
    return config


def main():
    x, y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

    classifiers = {
        'sklearn.linear_model.LogisticRegression': LogisticRegression,
        'sklearn.neighbors.KNeighborsClassifier': KNeighborsClassifier,
        'sklearn.svm.SVC': SVC,
        'sklearn.naive_bayes.GaussianNB': GaussianNB,
        'sklearn.tree.DecisionTreeClassifier': DecisionTreeClassifier
    }

    mlflow.set_tracking_uri('http://mlflow_server:5000')
    mlflow.set_experiment(mlflow.create_experiment(str(uuid.uuid4())))

    for config in get_config("data/config.yaml")["configs"]:
        mlflow.start_run(run_name=config['classificator'])

        classificator = classifiers[config['classificator']](**config['params'])

        classificator.fit(x_train, y_train)
        y_pred = classificator.predict(x_test)
        mlflow.log_metrics({
            "f1_score": f1_score(y_test, y_pred, average='micro'),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='micro'),
            "recall": recall_score(y_test, y_pred, average='micro')
        })

        model_info = mlflow.sklearn.log_model(
            sk_model=classificator,
            artifact_path=f"iris_{config['classificator']}"
        )
        mlflow.register_model(
            model_uri=model_info.model_uri,
            name=f"iris_{config['classificator']}"
        )
        mlflow.end_run()