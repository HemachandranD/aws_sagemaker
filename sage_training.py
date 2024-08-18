# %%
import warnings

import boto3
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.processing import ScriptProcessor
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

# %%
# Initialize SageMaker session and client
sagemaker_session = sagemaker.Session()
sm_client = boto3.client("sagemaker")
pipeline_session = PipelineSession()
role = sagemaker.get_execution_role()  # Replace with your SageMaker execution role
warnings.filterwarnings("ignore")

# %%
# Define input parameters
input_data_uri = ParameterString(
    name="InputDataUri", default_value="s3://hemz-sagemaker-bucket/input-data/"
)

output_data_uri = ParameterString(
    name="OutputDataUri", default_value="s3://hemz-sagemaker-bucket/output-data/"
)

model_approval_status = ParameterString(
    name="ModelApprovalStatus", default_value="Approved"
)

training_image_uri = (
    "654654222480.dkr.ecr.ap-south-1.amazonaws.com/prophet_training:latest"
)

model_package_group_name = "ProphetModelGroup"

# %%
# Define preprocessing step
script_processor = ScriptProcessor(
    image_uri=training_image_uri,  # Use a custom image with Prophet installed
    command=["python3"],
    role=role,
    sagemaker_session=pipeline_session,
    instance_count=1,
    instance_type="ml.t3.medium",
)

# %%
preprocessing_step = ProcessingStep(
    name="PreprocessData",
    processor=script_processor,
    inputs=[
        sagemaker.processing.ProcessingInput(
            source=input_data_uri, destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        sagemaker.processing.ProcessingOutput(
            output_name="train",
            destination=f"{output_data_uri.default_value}train",
            source="/opt/ml/processing/output/train",
        )
    ],
    code="code/preprocess.py",  # Replace with your preprocessing script
)

# %%
# Define the custom estimator for Prophet
prophet_estimator = sagemaker.estimator.Estimator(
    image_uri=training_image_uri,  # Use a custom image with Prophet installed
    role=role,
    sagemaker_session=pipeline_session,
    instance_count=1,
    instance_type="ml.m5.large",
    entry_point="code/train.py",
    script_mode=True,
    output_path="s3://hemz-sagemaker-bucket/output-data/model-artifacts/",
)

# %%
# Define training step
training_step = TrainingStep(
    name="TrainModel",
    estimator=prophet_estimator,
    inputs={
        "train": TrainingInput(
            s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs[
                "train"
            ].S3Output.S3Uri,
            content_type="text/csv",
        )
    },
)

# %%
model = Model(
    image_uri=training_image_uri,
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    sagemaker_session=pipeline_session,
    role=role,
)

# %%
register_model_step = model.register(
    content_types=["text/csv"],
    response_types=["text/csv"],
    model_package_group_name=model_package_group_name,
    approval_status=model_approval_status,
)

# %%
# Define model registration step
register_model_step = ModelStep(name="RegisterModel", step_args=register_model_step)

# %%
# Training pipeline
training_pipeline = Pipeline(
    name="TrainingPipeline",
    parameters=[input_data_uri, output_data_uri, model_approval_status],
    steps=[preprocessing_step, training_step, register_model_step],
)

# %%
# Execute pipelines
training_pipeline.upsert(role_arn=role)

# %%
# Start pipelines
execution = training_pipeline.start()

# %%
execution.wait()

# %%
