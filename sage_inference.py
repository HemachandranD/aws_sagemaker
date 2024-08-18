# %%
import json

import boto3
import pandas as pd

client = boto3.client("sagemaker-runtime")

df = pd.read_csv("synthetic_data.csv")
# Convert DataFrame to a list of dictionaries
realtime_data = df.iloc[:2, :].to_dict(orient="records")

# Convert to JSON
json_data = json.dumps(realtime_data)

response = client.invoke_endpoint(
    EndpointName="prophet-endpoint",
    ContentType="application/json",
    Body=json.dumps(json_data),
)

response = response["Body"].read().decode()
# Clean the JSON string by removing extra escape characters
cleaned_response = response.replace('\\"', '"').strip('"\n')

# %%
# Convert to DataFrame
prediction_df = pd.DataFrame(json.loads(cleaned_response))
prediction_df

# %%
