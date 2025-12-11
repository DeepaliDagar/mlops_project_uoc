import boto3
import mlflow
import os
from urllib.parse import urlparse
from botocore.exceptions import ClientError

MLFLOW_TRACKING_ARN = os.environ.get("MLFLOW_TRACKING_ARN")
REGION_NAME = os.environ.get("AWS_REGION")
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "NYC-Taxi-Fare-Autopilot-Runs")
MODEL_REGISTRY_NAME = os.environ.get("MLFLOW_MODEL_REGISTRY_NAME", "NYC-Taxi-Fare-Model-MLflow")

if not MLFLOW_TRACKING_ARN or not REGION_NAME:
    raise EnvironmentError(
        "The environment variables MLFLOW_TRACKING_ARN and AWS_REGION must be set."
    )

def get_sagemaker_client(region_name):
    """Gets the SageMaker client using the provided region."""
    return boto3.client("sagemaker", region_name=region_name)

def get_algorithm_name(inference_containers):
    try:
        if not inference_containers: return "Unknown"
        model_image_uri = inference_containers[-1]['Image']
        return model_image_uri.split('/')[-1].split(':')[0].replace('sagemaker-', '')
    except:
        return "Unknown"

def clean_metric_name(name):
    return name.replace(':', '_')

def run_tracking(automl_job_name):
    """
    Main logic to log a specific Autopilot job to MLflow.
    """
    region_name = REGION_NAME 

    print(f"\n--- Starting MLflow Tracking for Job: {automl_job_name} ---")
    
    sm_client = get_sagemaker_client(region_name)
    s3_client = boto3.client("s3", region_name=region_name)

    print(f"Connecting to MLflow: {MLFLOW_TRACKING_ARN}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_ARN)
    mlflow.set_experiment(EXPERIMENT_NAME)

    try:
        automl_desc = sm_client.describe_auto_ml_job(AutoMLJobName=automl_job_name)
    except ClientError as e:
        print(f"Error finding AutoML job: {e}")
        return

    best_candidate = automl_desc.get('BestCandidate')
    if not best_candidate:
        print("No 'BestCandidate' found.")
        return

    best_candidate_name = best_candidate['CandidateName']
    training_job_desc = sm_client.describe_training_job(TrainingJobName=best_candidate_name)

    algo_name = get_algorithm_name(best_candidate.get('InferenceContainers', []))
    hyperparameters = training_job_desc.get('HyperParameters', {})
    metric_list = training_job_desc.get('FinalMetricDataList', [])
    model_data_url = training_job_desc.get('ModelArtifacts', {}).get('S3ModelArtifacts')

    with mlflow.start_run(run_name=f"Autopilot-Best-{algo_name}") as run:
        mlflow.set_tag("sagemaker_automl_job", automl_job_name)
        mlflow.set_tag("sagemaker_training_job", best_candidate_name)
        mlflow.set_tag("source", "SageMaker Autopilot")

        if hyperparameters: 
            mlflow.log_params(hyperparameters)
        if metric_list:
            for m in metric_list:
                mlflow.log_metric(clean_metric_name(m['MetricName']), m['Value'])

        if model_data_url:
            try:
                parsed_url = urlparse(model_data_url)
                bucket_name, key = parsed_url.netloc, parsed_url.path.lstrip('/')
                local_filename = "model.tar.gz"
                
                s3_client.download_file(bucket_name, key, local_filename)
                mlflow.log_artifact(local_filename, artifact_path="model")
                
                model_uri = f"runs:/{run.info.run_id}/model"
                mlflow.register_model(model_uri, MODEL_REGISTRY_NAME)
                
                if os.path.exists(local_filename): os.remove(local_filename)
                print(f"Run logged and model registered in MLflow: {run.info.run_id}")
            except Exception as e:
                print(f"Error handling artifact: {e}")
