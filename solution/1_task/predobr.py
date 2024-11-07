#!/usr/bin/env python3
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

def get_data(data_path: str, labels: str):
    """Загрузка данных из CSV файлов."""
    df_features = pd.read_csv(f'{data_path}/{labels}_features.csv')
    df_target = pd.read_csv(f"{data_path}/targets_{labels}.csv")
    return df_features, df_target


def preprocess_data(df_features, df_labels, drop_columns, categorical_columns):
    # Удаляем ненужные столбцы
    df_features.drop(columns=drop_columns, inplace=True)

    # Обрабатываем категориальные столбцы, преобразовывая их в числовые
    for column in categorical_columns:
        df_features[column] = pd.Categorical(df_features[column]).codes
    
    # Заполняем пропуски в числовых столбцах медианой
    df_features.fillna(df_features.median(numeric_only=True), inplace=True)
    
    return df_features, df_labels



def split_data(x_data, y_data, test_size=0.2, random_state = 42):
    """Разделение данных на тренировочные и тестовые выборки."""
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size, random_state = random_state)
    return x_train, x_test, y_train, y_test


def log_metrics(config, y_test, y_pred):
    """Логирование метрик модели в MLflow."""
    mlflow.log_params(config)
    mlflow.log_metrics({
        "f1_score": f1_score(y_test, y_pred, average='weighted'),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted')
    })
