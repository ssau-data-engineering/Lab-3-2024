# Выполнил: Гудков Сергей, группа 6233

### Цель работы

Разработать два пайплайна с использованием Apache Airflow и MLflow для работы с датасетом по предсказанию сердечных приступов "Heart Attack Analysis & Prediction Dataset". Первый пайплайн осуществляет обучение нескольких моделей классификации, а второй - выбор лучшей модели и её хостинг в production среде.

---

## Пайплайн 1: Обучение моделей классификации

### Описание

Первый пайплайн был создан для обучения нескольких моделей классификации с использованием датасета по анализу сердечных приступов. Он включает следующие шаги:

1. **Загрузка и предобработка данных**: Датасет был загружен и нормализован с использованием `StandardScaler`. Данные были разделены на обучающую, тестовую и валидационную выборки.

2. **Обучение моделей**: Использовались пять различных классификаторов:
   - **LogisticRegression**
   - **DecisionTreeClassifier**
   - **KNeighborsClassifier**
   - **RandomForestClassifier**
   - **SVC**

3. **Конфигурация моделей**: Параметры для каждой модели были заданы в конфигурационном файле `config.yaml`.

4. **Логирование метрик**: Каждая модель была обучена, и её метрики, такие как F1-score, accuracy, precision, recall, были залогированы в MLflow для последующего анализа и выбора лучшей модели.

### Итоги

Все модели были успешно обучены, а их метрики сохранены в MLflow. Валидационные данные были также сохранены для дальнейшего использования во втором пайплайне.
![image](https://github.com/user-attachments/assets/c7ead420-81dd-4418-9b37-e7def101414e)

Метрики после изменения конфигурационного файла: 
![image](https://github.com/user-attachments/assets/acc8d2b6-ac41-4013-8401-ad7b175c0c8c)


### Программный код пайплайна

```python
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import yaml
import mlflow
import uuid
import importlib
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# Настройка переменных окружения для MLflow и MinIO
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

# Пути к данным и конфигурациям
current_dir = os.path.dirname(os.path.abspath(__file__))
path_to_input = os.path.join(current_dir, '..', 'data', 'config')
path_to_val = os.path.join(current_dir, '..', 'data', 'validation')
path_to_data = os.path.join(current_dir, '..', 'data', 'heart')

# Функция для обучения моделей

def train_model_from_file():
    # Шаг 1: Загрузка данных
    df = pd.read_csv(os.path.join(path_to_data, 'heart.csv'))
    
    # Шаг 2: Разделение данных на признаки и целевую переменную
    X = df.drop(columns=['output'])
    y = df['output']

    # Шаг 3: Нормализация признаков
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Шаг 4: Разделение данных на обучающую, тестовую и валидационную выборки
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

    # Сохранение валидационных данных
    pd.DataFrame(x_val, columns=df.columns.drop('output')).to_csv(f'{path_to_val}/x_val.csv', index=False)
    pd.DataFrame(y_val, columns=['output']).to_csv(f'{path_to_val}/y_val.csv', index=False)

    # Шаг 5: Чтение конфигурационного файла
    with open(os.path.join(path_to_input, 'config.yaml'), 'r') as file:
        config = yaml.safe_load(file)

    # Шаг 6: Настройка MLflow
    mlflow.set_tracking_uri('http://mlflow_server:5000')
    experiment_id = uuid.uuid4()
    experiment = mlflow.create_experiment(str(experiment_id))
    mlflow.set_experiment(experiment)
    
    # Шаг 7: Обучение моделей согласно конфигурации
    for _, conf in enumerate(config['configs']):
        mlflow.start_run(run_name=conf['classificator'], experiment_id=experiment)

        # Динамический импорт модели
        module = importlib.import_module('.'.join(conf['classificator'].split('.')[:-1]))
        classificator_class = getattr(module, conf['classificator'].split('.')[-1])
        model = classificator_class(**conf['kwargs'])

        # Обучение модели
        model.fit(x_train, y_train)

        # Предсказание на тестовых данных
        y_pred = model.predict(x_test)

        # Логирование метрик в MLflow
        mlflow.log_metrics({
            "f1_score": f1_score(y_test, y_pred),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred)
        })

        # Сохранение модели в MLflow Model Registry
        model_info = mlflow.sklearn.log_model(sk_model=model, artifact_path=conf['classificator'])
        model_uri = model_info.model_uri
        mlflow.register_model(model_uri=model_uri, name=conf['classificator'])

        mlflow.end_run()

# Определение DAG для Airflow
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 11, 26),
    'retries': 1,
}

dag = DAG(
    'heart_attack_training',
    default_args=default_args,
    description='DAG for training heart attack prediction models',
    schedule_interval=None,
)

wait_for_new_file = FileSensor(
    task_id='wait_for_new_file',
    poke_interval=10,
    filepath=path_to_input,
    fs_conn_id='lab_connect',
    dag=dag,
)

train_model = PythonOperator(
    task_id='train_models',
    python_callable=train_model_from_file,
    dag=dag,
)

wait_for_new_file >> train_model

```

---

## Пайплайн 2: Выбор лучшей модели и её хостинг

### Описание

Второй пайплайн был создан для выбора лучшей из обученных моделей на основе метрики F1-score и перевода этой модели на этап "Production". Пайплайн выполняет следующие шаги:

1. **Проверка новых моделей**: Пайплайн запускается ежедневно и проверяет все новые модели, залогированные в MLflow.

2. **Валидация моделей**: Используются сохраненные валидационные данные для оценки каждой модели. Пайплайн загружает все версии моделей из Model Registry и рассчитывает F1-score на валидационной выборке.

3. **Выбор лучшей модели**: Модель с наилучшим F1-score выбирается и переводится на этап "Production".

4. **Хостинг модели**: Лучшая модель переводится на stage "Production", что позволяет использовать её для предсказаний в реальной среде.

![image](https://github.com/user-attachments/assets/c5333cf0-214e-4e85-a535-73930097c618)

После публикации модели в продакшен был изменён конфигурационный файл, и пайплайны запущены снова, в виду чего изменились метрики и на продакшен попала другая модель

![image](https://github.com/user-attachments/assets/1424c0d7-6447-49c7-9d33-9f6bfc127d04)

![image](https://github.com/user-attachments/assets/e721c504-6f5d-45af-879b-55a5894d76df)


### Программный код пайплайна

```python
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import os
import pandas as pd
import mlflow
import yaml
from sklearn.metrics import f1_score

# Настройка переменных окружения для MinIO и MLflow
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

# Пути к данным
current_dir = os.path.dirname(os.path.abspath(__file__))
path_to_val = os.path.join(current_dir, '..', 'data', 'validation')
path_to_input = os.path.join(current_dir, '..', 'data', 'config')

# Функция для выбора лучшей модели и перевода её в stage 'Production'
def choose_best_model():
    # Настройка MLflow
    mlflow.set_tracking_uri('http://mlflow_server:5000')
    client = mlflow.MlflowClient()

    # Загрузка валидационных данных
    x_val = pd.read_csv(os.path.join(path_to_val, 'x_val.csv'))
    y_val = pd.read_csv(os.path.join(path_to_val, 'y_val.csv'))

    # Загрузка конфигурационного файла
    with open(os.path.join(path_to_input, 'config.yaml'), 'r') as file:
        config = yaml.safe_load(file)

    # Словарь для хранения значений F1-score для каждой модели
    f1_scores = {}

    # Оценка каждой модели из конфигурации
    for _, conf in enumerate(config['configs']):
        model_uri = f"models:/{conf['classificator']}/None"
        model = mlflow.sklearn.load_model(model_uri)
        y_pred = model.predict(x_val)
        f1_scores[conf['classificator']] = f1_score(y_val, y_pred)

    # Выбор модели с наилучшим F1-score
    best_model_name = max(f1_scores, key=f1_scores.get)

    # Перевод выбранной модели на stage 'Production'
    filter_string = f"name='{best_model_name}'"
    best_model = client.search_registered_models(filter_string=filter_string)[0]
    all_versions = client.search_model_versions(filter_string=filter_string)
    client.transition_model_version_stage(
        name=best_model.name,
        version=all_versions[0].version,
        stage='Production'
    )

# Определение DAG для Airflow
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 11, 7),
    'retries': 1,
}

dag = DAG(
    'model_validation_and_hosting',
    default_args=default_args,
    description='DAG for validating and hosting the best model',
    schedule_interval='@daily',
)

validate_and_host = PythonOperator(
    task_id='choose_best_model',
    python_callable=choose_best_model,
    dag=dag,
)

validate_and_host
```
