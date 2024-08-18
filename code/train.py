import os
import pickle

import pandas as pd
from prophet import Prophet


def train(df):
    # Initialize and fit Prophet model
    model = Prophet()
    model.fit(df)
    return model


if __name__ == "__main__":
    train_data_path = "/opt/ml/input/data/train"  # Adjust the filename if necessary
    df = pd.read_csv(f"{train_data_path}/train.csv")
    model = train(df)
    # Save the model to the /opt/ml/model/ directory
    model_path = os.path.join("/opt/ml/model", "prophet_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
