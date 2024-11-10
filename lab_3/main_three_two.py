import mlflow
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from mlflow.tracking import MlflowClient
from typing import Optional, Dict, List, Tuple

# Конфигурация
MLFLOW_TRACKING_URI = 'http://mlflow_server:5000'
TARGET_METRIC = 'test_f1'
SCHEDULE_INTERVAL = '@daily'
MINIMUM_ACCEPTABLE_ACCURACY = 0.4

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(MLFLOW_TRACKING_URI)

# Список всех моделей
def get_registered_models() -> List[str]:
    registered_models = client.search_registered_models()
    return [model.name for model in registered_models]

# Метрики моделей
def get_model_metrics(version) -> Dict:
    run = client.get_run(version.run_id)
    return {
        'version': version.version,
        'test_f1': run.data.metrics.get(TARGET_METRIC, -1),
        'current_stage': version.current_stage,
        'run_id': version.run_id
    }

# Поиск лучшей модели
def find_best_model() -> Optional[Tuple[str, Dict]]:
    model_names = get_registered_models()
    best_metric = float('-inf')
    best_model_info = None
    best_model_name = None
    
    for model_name in model_names:
        versions = client.get_registered_model(model_name).latest_versions
        
        for version in versions:
            metrics = get_model_metrics(version)
            
            if (version.current_stage in ['None', 'Staging'] and 
                metrics['test_f1'] >= MINIMUM_ACCEPTABLE_ACCURACY and 
                metrics['test_f1'] > best_metric):
                best_metric = metrics['test_f1']
                best_model_info = {
                    'version': version,
                    'metric': metrics['test_f1']
                }
                best_model_name = model_name
    
    return (best_model_name, best_model_info) if best_model_name else None

# Продвижение в Production
def promote_to_production(model_name: str, version) -> bool:
    for name in get_registered_models():
        versions = client.get_registered_model(name).latest_versions
        for ver in versions:
            if ver.current_stage == 'Production':
                client.transition_model_version_stage(
                    name=name,
                    version=ver.version,
                    stage='Archived'
                )
    
    # Перевод в Staging, если необходимо
    if version.current_stage == 'None':
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage='Staging'
        )
    
    # Перевод в Production
    client.transition_model_version_stage(
        name=model_name,
        version=version.version,
        stage='Production'
    )
    return True

def validate_and_promote_best_model():
    best_model = find_best_model()
    if best_model:
        model_name, model_info = best_model
        promote_to_production(model_name, model_info['version'])

dag = DAG(
    'data_engineering_lab_3_part_2',
    default_args={
        'owner': 'airflow',
        'start_date': datetime(2024, 11, 8),
        'retries': 1,
    },
    description='DAG validating and promoting best model',
    schedule_interval=SCHEDULE_INTERVAL,
)

validate_and_promote_task = PythonOperator(
    task_id='validate_and_promote_best_model',
    python_callable=validate_and_promote_best_model,
    dag=dag,
)

validate_and_promote_task