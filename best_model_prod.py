from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import mlflow

import joblib
import os


def validate_and_select_best_model():
    mlflow.set_tracking_uri('http://mlflow_server:5000')
    client = mlflow.tracking.MlflowClient()

    print("\ndebug\n")

    best_model = None
    best_metric = -1.0
    best_v = None

    for model in client.search_registered_models():
        versions = client.search_model_versions(f"name='{model.name}'")
        for version in versions:
            metric_value = client.get_run(version.run_id).data.metrics.get('accuracy')

            if metric_value > best_metric:
                best_metric = metric_value
                best_model = model
                best_v = version.version

    if best_model:
        client.transition_model_version_stage(
            name=best_model.name,
            version=best_v,
            stage="Production"
        )

    print("\n\n\n")
    print(best_model.name)
    print("\n\n\n")


dag = DAG(
    'best_model_hosting_pipeline',
    default_args={
        'owner': 'airflow',
    },
    description='Pipeline for selecting and hosting the best model from registry',
    schedule_interval=timedelta(days=1),
    start_date=datetime.now(),
    catchup=False,
)

validate_task = PythonOperator(
    task_id='validate_and_select_best_model',
    python_callable=validate_and_select_best_model,
    dag=dag,
)

validate_task
