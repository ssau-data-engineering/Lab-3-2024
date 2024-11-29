# Engineering

## Ионов Артем группа 6231-010402D

Весь необходимый минимум для выполнения лабораторных работ был выполнен перед 1 лабораторной работой.

![image](https://github.com/user-attachments/assets/41eaed57-c69b-4a4a-aa2f-3ba4910b417f)

### Лабораторная работа №3 "Airflow и MLflow - логгирование экспериментов и версионирование моделей"

## Часть 1 "Prepare and train"

Для начала нужно было подобрать модели и датасет для обучения. Датасет я взял тот же [iris](iris.csv) из предыдущей лабораторной работы. А модели выбрал дефолтные:

- RandomForestClassifier
- GradientBoostingClassifier
- KNeighborsClassifier
- SGDClassifier
- SVC

Вот как выглядит [даг файл](Dag1lr3.py) для первой части. У нас предобрабатывается датасет, делится на обучащие, тестовые и тд выборки. 
На этих данных обучается и все у нас отправляется в mlflow...

![image](https://github.com/user-attachments/assets/bfaa4dd0-75f0-4586-8e32-ded067e7483e)

Даг у меня прогнался без проблем.

![image](https://github.com/user-attachments/assets/3f673b08-5fcc-4aa2-bc9c-a00f27339e44)

И в mlflow можно наблюдать все проведенные эксперименты с нашими модельками:

![image](https://github.com/user-attachments/assets/1b79fdcf-bf5d-48e6-b27e-85a6d5a88bcf)

## Часть 2 "Production"

Здесь нужно было перевести наилучшую модель в Production. Вот как выглядит [даг файл](Dag2lr3.py). 

![image](https://github.com/user-attachments/assets/a7c8de03-d85c-4e58-813e-d58282946fbd)

- Модель с наибольшей точностью выбирается с помощью функции max, которая ищет ключ (модель) с максимальным значением точности в словаре current_results
- С помощью метода search_model_versions ищется версия модели с наилучшей точностью по её имени и идентификатору run_id.
- Модель переводится в стадию "Production" с помощью метода transition_model_version_stage, где указывается имя модели и её версия.

Даг прогнался почти без проблем:

![image](https://github.com/user-attachments/assets/be500b67-7ef8-460b-a76b-51e5a5d032d3)

И в mlflow в Models можно наблюдать, что наилучшая модель перешла в Production

![image](https://github.com/user-attachments/assets/bba5b86c-6e69-4a9f-a131-9b86295d2256)

## Hosting

Захостить модель не вышло, хотя задача казалась не такой и сложной.
В некоторых моментах думал что получилось, но запрос отправить я не смог.

Я думаю что Mlflow нужно развернуть отдельно и тогда наверное все получится.

```
mlflow models build-docker --model-uri models:/SGDClassifier/production

```


