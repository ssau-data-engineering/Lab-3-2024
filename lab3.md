# Пайплайн для обучения классификаторов
Для обучения были выбраны следующие классификаторы:
- Логистическая регрессия
- Наивный байесовский классификатор
- Метод k ближайших соседей
- Метод опорных векторов
- Дерево решений

Были использованы параметры по умолчанию в sklearn, см. в файле config.yaml

В качестве данных - датасет ирисы из sklearn

![Airflow1](./images/airflow1.png)

![MlFlow](./images/mlflow1.png)
![MlFlow](./images/mlflow2.png)
![MlFlow](./images/mlflow3.png)
![MlFlow](./images/mlflow4.png)

# Пайплайн для хостинга лучшей модейли

![Airflow](./images/airflow2.png)
![!Prod](./images/production.png)

Для работоспособности mlflow необходимо было добавить зависимость boto3