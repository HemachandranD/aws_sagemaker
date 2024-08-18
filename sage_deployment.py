# %%
import warnings
from time import gmtime, strftime

import boto3
import sagemaker

# %%
# Initialize SageMaker session and client
sagemaker_session = sagemaker.Session()
sm_client = boto3.client("sagemaker")
role = sagemaker.get_execution_role()  # Replace with your SageMaker execution role
warnings.filterwarnings("ignore")

# %%
sagemaker_bucket_name = "hemz-sagemaker-bucket"

inference_image_uri = (
    "654654222480.dkr.ecr.ap-south-1.amazonaws.com/prophet_infernece:latest"
)

model_name = "Prophet-serverless-model" + "-" + strftime("%Y%m%d%H%M%S", gmtime())

endpoint_config_name = "prophet-endpoint-config"

endpoint_name = "prophet-endpoint"

# %%
# List training jobs
response = sm_client.list_training_jobs(SortBy="CreationTime", SortOrder="Descending")

# Print training jobs to find the relevant one
latest_training_job_name = (
    response["TrainingJobSummaries"][0]["TrainingJobName"]
    if response["TrainingJobSummaries"][0]["TrainingJobStatus"] == "Completed"
    else print("The latest training Job is not successful")
)

# %%
# Create a model for endpoint
sm_client.create_model(
    ModelName=model_name,
    ExecutionRoleArn=role,
    Containers=[
        {
            "Image": inference_image_uri,
            "Mode": "SingleModel",
            "ModelDataUrl": f"s3://{sagemaker_bucket_name}/output-data/model-artifacts/{latest_training_job_name}/output/model.tar.gz",
        }
    ],
)

# %%
# Create endpoint configuration
endpoint_config = sm_client.create_endpoint_config(
    EndpointConfigName="prophet-endpoint-config",
    ProductionVariants=[
        {
            "ModelName": model_name,
            "VariantName": "AllTraffic",
            "ServerlessConfig": {
                "MemorySizeInMB": 2048,
                "MaxConcurrency": 1,
                "ProvisionedConcurrency": 1,
            },
        }
    ],
)

# %%
# Create the endpoint
sm_client.create_endpoint(
    EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
)

# %%
