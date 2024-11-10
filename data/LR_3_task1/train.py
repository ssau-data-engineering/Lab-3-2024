import numpy as np
import json
import os
import importlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime

def get_data_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def create_model(model_info_json):
    module = importlib.import_module(model_info_json["module"])
    model_classificator = getattr(module, model_info_json["model_classificator"])
    return model_classificator(**model_info_json["args"])

def load_and_preprocess_data(df_json):
    path_input_df = os.path.join(file_directory, df_json["name_file"])

    df = pd.read_csv(path_input_df)
    df.drop(df_json["drop_columns"], axis=1, inplace=True, errors='ignore')

    X = df.drop(columns=df_json["target_columns"])
    y = df[df_json["target_columns"]]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=None)  
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=None)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_valid_scaled = scaler.transform(X_valid)

    path_output_x_valid = os.path.join(file_directory, df_json["name_valid_X_file"])
    path_otuput_y_valid = os.path.join(file_directory, df_json["name_valid_y_file"])

    np.save(path_output_x_valid,X_valid_scaled)
    np.save(path_otuput_y_valid,y_valid)
    
    return X_train_scaled, y_train, X_test_scaled, y_test


if __name__ == '__main__':
    
    file_directory = os.path.dirname(os.path.abspath(__file__))
    path_input_json = os.path.join(file_directory, "models.json")

    data_json = get_data_json(path_input_json)
    models_json = data_json["models"]
    df_json = data_json["dataset"]
    
    X_train, y_train, X_test, y_test = load_and_preprocess_data(df_json)
    
    url = 'http://mlflow_server:5000'
    mlflow.set_tracking_uri(url)
    experiment_name = "LR_3"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()

    for model_json in models_json:

        trained_model  = create_model(model_json)
        model_name = model_json["model_classificator"]
        
        with mlflow.start_run(run_name=model_name) as run:

            for param_name, param_value in model_json["args"].items():
                mlflow.log_param(param_name, param_value)

            

            start_time = datetime.now()
            trained_model.fit(X_train, y_train)
            end_time = datetime.now()
            elapsed_time = end_time - start_time

            y_train_pred = trained_model.predict(X_train)
            y_test_pred = trained_model.predict(X_test)

            accuracy_train = accuracy_score(y_train, y_train_pred)
            precision_train = precision_score(y_train, y_train_pred, average='binary')
            recall_train = recall_score(y_train, y_train_pred, average='binary')
            f1_train = f1_score(y_train, y_train_pred, average='binary')

            accuracy_test = accuracy_score(y_test, y_test_pred)
            precision_test = precision_score(y_test, y_test_pred, average='binary')
            recall_test = recall_score(y_test, y_test_pred, average='binary')
            f1_test = f1_score(y_test, y_test_pred, average='binary')

            print(f"{trained_model} model Train_data\nAccuracy: {accuracy_train:.4f}\nPrecision: {precision_train:.4f}\nRecall: {recall_train:.4f}\nF1: {f1_train:.4f}")
            print(f"{trained_model} model\nAccuracy: {accuracy_test:.4f}\nPrecision: {precision_test:.4f}\nRecall: {recall_test:.4f}\nF1: {f1_test:.4f}")

            mlflow.log_param("training_time_seconds", elapsed_time.total_seconds())
            mlflow.sklearn.log_model(trained_model, "model")

            mlflow.log_metrics({
                 "accuracy_train": accuracy_train,
                 "precision_train": precision_train,
                 "recall_train": recall_train,
                 "f1_train": f1_train
            })

            mlflow.log_metrics({
                 "accuracy_test": accuracy_test,
                 "precision_test": precision_test,
                 "recall_test": recall_test,
                 "f1_test": f1_test
            })

            
            client.get_registered_model(model_name)
            model_version = mlflow.register_model(
                    f"runs:/{run.info.run_id}/model",
                     model_name
                )



