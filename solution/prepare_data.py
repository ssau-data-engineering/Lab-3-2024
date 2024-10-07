from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

data_path = '/data/'

wines = load_wine()

x_train, x_test, y_train, y_test = train_test_split(wines.data, wines.target, test_size=0.15, shuffle=True, random_state=163)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, shuffle=True, random_state=163)

pd.DataFrame(x_train).to_csv(f'{data_path}/x_train.csv', index=False)
pd.DataFrame(y_train).to_csv(f'{data_path}/y_train.csv', index=False)

pd.DataFrame(x_val).to_csv(f'{data_path}/x_val.csv', index=False)
pd.DataFrame(y_val).to_csv(f'{data_path}/y_val.csv', index=False)

pd.DataFrame(x_test).to_csv(f'{data_path}/x_test.csv', index=False)
pd.DataFrame(y_test).to_csv(f'{data_path}/y_test.csv', index=False)