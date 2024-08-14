import sagemaker
import boto3
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CreateModelStep, EndpointConfigStep, EndpointStep
from sagemaker.processing import ScriptProcessor
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.model_step import RegisterModel
from sagemaker.workflow.parameters import ParameterString

# Initialize SageMaker session and client
sagemaker_session = sagemaker.Session()
sm_client = boto3.client('sagemaker')
role = 'YOUR_SAGEMAKER_EXECUTION_ROLE'  # Replace with your SageMaker execution role

# Define input parameters
input_data_uri = ParameterString(
    name="InputDataUri",
    default_value="s3://your-bucket/input-data/"
)

output_data_uri = ParameterString(
    name="OutputDataUri",
    default_value="s3://your-bucket/output-data/"
)

model_approval_status = ParameterString(
    name="ModelApprovalStatus",
    default_value="Approved"
)

# Define preprocessing step
script_processor = ScriptProcessor(
    image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/forecasting-deep-forecasting:latest",  # Use a custom image with Prophet installed
    command=["python3"],
    role=role,
    instance_count=1,
    instance_type="ml.m5.large"
)

preprocessing_step = ProcessingStep(
    name="PreprocessData",
    processor=script_processor,
    inputs=[sagemaker.processing.ProcessingInput(source=input_data_uri, destination="/opt/ml/processing/input")],
    outputs=[sagemaker.processing.ProcessingOutput(output_name="processed_data", destination=output_data_uri, source="/opt/ml/processing/output")],
    code="preprocessing.py"  # Replace with your preprocessing script
)

# Define the custom estimator for Prophet
prophet_estimator = sagemaker.estimator.Estimator(
    image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/forecasting-deep-forecasting:latest",  # Use a custom image with Prophet installed
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    output_path="s3://your-bucket/model-artifacts/"
)

# Define training step
training_step = TrainingStep(
    name="TrainProphetModel",
    estimator=prophet_estimator,
    inputs={
        "train": TrainingInput(s3_data=output_data_uri, content_type="text/csv")
    }
)

# Define evaluation step (optional)
evaluation_processor = ScriptProcessor(
    image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/forecasting-deep-forecasting:latest",  # Use a custom image with Prophet installed
    command=["python3"],
    role=role,
    instance_count=1,
    instance_type="ml.m5.large"
)

evaluation_step = ProcessingStep(
    name="EvaluateModel",
    processor=evaluation_processor,
    inputs=[sagemaker.processing.ProcessingInput(source=training_step.properties.ModelArtifacts.S3ModelArtifacts, destination="/opt/ml/processing/model")],
    outputs=[sagemaker.processing.ProcessingOutput(output_name="evaluation", destination="s3://your-bucket/evaluation", source="/opt/ml/processing/evaluation")],
    code="evaluate.py"  # Replace with your evaluation script
)

# Define model registration step
model = prophet_estimator.create_model()
register_model_step = RegisterModel(
    name="RegisterProphetModel",
    model=model,
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.m5.large"],
    transform_instances=["ml.m5.large"],
    model_package_group_name="YourModelPackageGroup",  # Replace with your model package group name
    approval_status=model_approval_status
)

# Define model deployment step
create_model_step = CreateModelStep(
    name="CreateProphetModel",
    model=model,
    inputs=sagemaker.inputs.CreateModelInput(instance_type="ml.m5.large")
)

endpoint_config_step = EndpointConfigStep(
    name="CreateEndpointConfig",
    endpoint_config_name="prophet-endpoint-config",
    model_name=create_model_step.properties.ModelName,
    initial_instance_count=1,
    instance_type="ml.m5.large"
)

endpoint_step = EndpointStep(
    name="DeployModelToEndpoint",
    endpoint_name="prophet-endpoint",
    endpoint_config_name=endpoint_config_step.properties.EndpointConfigName
)

# Training pipeline
training_pipeline = Pipeline(
    name="TrainingPipeline",
    steps=[preprocessing_step, training_step, evaluation_step]
)

# Deployment pipeline
deployment_pipeline = Pipeline(
    name="DeploymentPipeline",
    steps=[register_model_step, create_model_step, endpoint_config_step, endpoint_step]
)

# Execute pipelines
training_pipeline.upsert(role_arn=role)
deployment_pipeline.upsert(role_arn=role)

# Start pipelines
training_pipeline.start()
deployment_pipeline.start()
