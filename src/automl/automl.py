import os
import boto3
import sagemaker
from sagemaker.automl.automl import AutoML
import time
from sagemaker.exceptions import UnexpectedStatusException

# Environment Variables
BUCKET = os.environ.get("S3_BUCKET_NAME") 
INPUT_DATA_KEY = os.environ.get("S3_INPUT_KEY", "processed/train.csv")
AUTOML_OUTPUT_PREFIX = os.environ.get("AUTOML_OUTPUT_PREFIX", "automl/autopilot-output") 
TARGET_COLUMN = os.environ.get("TARGET_COLUMN_NAME", "fare_amount")
REGION = os.environ.get("AWS_REGION") 
EXECUTION_ROLE_ARN = os.environ.get("SAGEMAKER_EXECUTION_ROLE_ARN") 

if not BUCKET or not REGION:
    raise EnvironmentError("The environment variables S3_BUCKET_NAME and AWS_REGION must be set.")

def get_sagemaker_session_and_role():
    """Gets the SageMaker session and execution role."""
    boto_session = boto3.Session(region_name=REGION)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    role = EXECUTION_ROLE_ARN or sagemaker.get_execution_role(sagemaker_session)
    return sagemaker_session, role

def start_autopilot_job(sagemaker_session, role):
    """Starts a SageMaker Autopilot training job with RMSE objective."""
    
    job_name = f'NYC-Fare-{int(time.time())}'
    input_data_uri = f's3://{BUCKET}/{INPUT_DATA_KEY}'
    output_location = f's3://{BUCKET}/{AUTOML_OUTPUT_PREFIX}/{job_name}'
    
    autopilot_estimator = AutoML(
        role=role,
        target_attribute_name=TARGET_COLUMN,
        sagemaker_session=sagemaker_session,
        output_path=output_location,
        max_candidates=10,
        
        problem_type='Regression',
        job_objective={'MetricName': 'RMSE'}, 
    )

    print(f"Starting SageMaker Autopilot Job: {job_name}")
    print(f"Objective: Minimizing RMSE")
    print(f"Output: {output_location}")

    autopilot_estimator.fit(
        inputs=input_data_uri,
        job_name=job_name,
        wait=False
    )
    
    return autopilot_estimator, job_name

def monitor_job_and_display_results(autopilot_estimator, job_name):
    """
    Polls the job status and prints new models as they finish training.
    """
    print("\nMonitoring Autopilot Job (Live Feed)...")
    
    sm_client = autopilot_estimator.sagemaker_session.boto_session.client('sagemaker')
    
    seen_candidates = set()
    job_status = "InProgress"
    
    while job_status in ["InProgress", "Stopping"]:
        try:
            desc = sm_client.describe_auto_ml_job(AutoMLJobName=job_name)
            job_status = desc['AutoMLJobStatus']
            secondary_status = desc['AutoMLJobSecondaryStatus']
            
            candidates = sm_client.list_candidates_for_auto_ml_job(
                AutoMLJobName=job_name,
                SortBy='FinalObjectiveMetricValue',
                SortOrder='Ascending', 
                MaxResults=100
            )
            
            for cand in candidates['Candidates']:
                c_name = cand['CandidateName']
                if c_name not in seen_candidates:
                    score = cand['FinalAutoMLJobObjectiveMetric']['Value']
                    image_uri = cand['InferenceContainers'][0]['Image']
                    algo_short = image_uri.split('/')[-1].split(':')[0].replace('sagemaker-', '')
                    
                    print(f"Model Finished: {algo_short:<15} | RMSE: {score:.4f}")
                    seen_candidates.add(c_name)
            
            print(f"\r   Status: {job_status} - {secondary_status} ...", end="")
            
            if job_status in ["Completed", "Failed", "Stopped"]:
                print()
                break
                
            time.sleep(30) 

        except Exception as e:
            print(f"\nMonitoring interrupted: {e}")
            break

    if job_status == "Completed":
        print("\nAutopilot Job Completed Successfully!")
        best_candidate = desc['BestCandidate']
        best_name = best_candidate['CandidateName']
        best_metric = best_candidate['FinalAutoMLJobObjectiveMetric']['Value']
        
        print(f"BEST CANDIDATE: {best_name}")
        print(f"Final RMSE: {best_metric:.4f}")
        return best_candidate
    else:
        print(f"\nJob failed or stopped with status: {job_status}")
        return None


