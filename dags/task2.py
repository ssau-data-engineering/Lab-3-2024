import os
from datetime import datetime, timedelta, timezone
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import mlflow
from mlflow.tracking import MlflowClient

os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

def validate_and_promote_models():
    mlflow.set_tracking_uri('http://mlflow_server:5000')
    client = MlflowClient()
    best_metric_value = None
    best_model_version = None
    best_model_name = None
    
    experiment_name = "MNIST_Experiment"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment {experiment_name} not found.")
        return
    experiment_id = experiment.experiment_id    
    runs = client.search_runs(experiment_ids=[experiment_id])
    
    for run in runs:
        run_id = run.info.run_id
        for mv in client.search_model_versions(f"run_id='{run_id}'"):
            model_name = mv.name
            version_number = mv.version
            metric_value = run.data.metrics.get('accuracy')

            if best_metric_value is None or metric_value > best_metric_value:
                best_metric_value = metric_value
                best_model_version = version_number
                best_model_name = model_name

    if best_model_name:
        client.transition_model_version_stage(
            name=best_model_name,
            version=best_model_version,
            stage="Production"
        )
        print(f"Promoted model {best_model_name} version {best_model_version} to Production with metric value {best_metric_value}")
    else:
        print("No models found to promote.")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 11, 30, tzinfo=timezone.utc), 
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    dag_id = 'task2',
    default_args=default_args,
    description='Pipeline for validating and hosting the best model',
    schedule_interval=timedelta(days=1),
    catchup=False
)

validate_and_promote_models_task = PythonOperator(
    task_id='validate_and_promote_models',
    python_callable=validate_and_promote_models,
    provide_context=True,
    dag=dag
)

validate_and_promote_models_task