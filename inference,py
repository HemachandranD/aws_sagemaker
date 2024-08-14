import boto3
import pandas as pd
import json

# Initialize the SageMaker runtime client
sagemaker_runtime = boto3.client('sagemaker-runtime')

# Define the endpoint name (replace with your actual endpoint name)
endpoint_name = "prophet-endpoint"

# Prepare the input data
# Example data format (must match the format the model expects)
data = {
    "ds": ["2024-08-01", "2024-08-02", "2024-08-03", "2024-08-04"]
}

# Convert the data to a CSV string (this format should match what the model was trained on)
data_df = pd.DataFrame(data)
csv_data = data_df.to_csv(index=False, header=False)

# Send the data to the endpoint
response = sagemaker_runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="text/csv",  # Adjust if your model expects a different content type
    Body=csv_data
)

# Parse the response
result = response['Body'].read().decode('utf-8')
predictions = json.loads(result)

# Display the predictions
print(predictions)
