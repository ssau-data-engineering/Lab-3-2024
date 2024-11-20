import pandas as pd
import numpy as np

tracking_url = 'http://mlflow_server:5000'

def get_data(data_path:str, labels:str):
    array_data = np.asarray(pd.read_csv(f'{data_path}/x_{labels}.csv'), dtype=np.float32)
    array_target = pd.read_csv(f"{data_path}/y_{labels}.csv")
    return array_data, array_target

def save_data(dataset, data_path:str, types:str, labels:str):
    pd.DataFrame(dataset).to_csv(f'{data_path}/{types}_{labels}.csv', index=False)