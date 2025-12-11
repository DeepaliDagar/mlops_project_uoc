import os
import time
import boto3
from sagemaker.model import ModelPackage
from sagemaker import get_execution_role, Session
from sagemaker.predictor import Predictor
from sagemaker.serializers import IdentitySerializer
from sagemaker.deserializers import CSVDeserializer

MODEL_PACKAGE_ARN = os.environ.get("SAGEMAKER_MODEL_PACKAGE_ARN")
INSTANCE_TYPE = os.environ.get("ENDPOINT_INSTANCE_TYPE", "ml.m5.large")
REGION = os.environ.get("AWS_REGION")
ENDPOINT_NAME_PREFIX = os.environ.get("ENDPOINT_NAME_PREFIX", "NYC-Fare-Prod")

if not MODEL_PACKAGE_ARN or not REGION:
    raise EnvironmentError(
        "The environment variables SAGEMAKER_MODEL_PACKAGE_ARN and AWS_REGION must be set."
    )

def deploy_model(model_package_arn, instance_type, region, endpoint_name=None):
    """
    Deploys the specified SageMaker Model Package to a real-time endpoint.
    """
    boto_session = boto3.Session(region_name=region)
    sagemaker_session = Session(boto_session=boto_session)
    role = get_execution_role(sagemaker_session)
    print(f"SageMaker Execution Role: {role}")
    
    model_package = ModelPackage(
        model_package_arn=model_package_arn,
        role=role,
        predictor_cls=Predictor,
        sagemaker_session=sagemaker_session
    )


    if not endpoint_name:
        endpoint_name = f"{ENDPOINT_NAME_PREFIX}-{int(time.time())}"
    print(f"Deploying model to endpoint: {endpoint_name}...")
    
    
    predictor = model_package.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name
    )

    print(f"Model deployed successfully!")
    print(f"Endpoint name: {predictor.endpoint_name}")
    
    return predictor

def test_prediction(predictor):
    """
    Sends a test payload to the newly deployed endpoint.
    """
    print("-" * 50)
    print("Testing prediction on the new endpoint...")
    
    predictor.serializer = IdentitySerializer(content_type='text/csv')
    predictor.deserializer = CSVDeserializer()
    csv_data = "40:00.0,2010-03-14 15:40:00 UTC,-73.979872,40.749027,-73.976553,40.757498,5,15,3,6,1.0"

    try:
        response = predictor.predict(csv_data)
        print(f"Test Prediction Result (Fare Amount): {response}")
    except Exception as e:
        print(f"Error during test prediction: {e}")

if __name__ == "__main__":
    
    predictor = deploy_model(
        model_package_arn=MODEL_PACKAGE_ARN,
        instance_type=INSTANCE_TYPE,
        region=REGION,
        endpoint_name=f"{ENDPOINT_NAME_PREFIX}-{int(time.time())}" 
    )

    if predictor:
        test_prediction(predictor)