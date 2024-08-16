from prophet import Prophet
import pandas as pd


def train():
    # Initialize and fit Prophet model
    model = Prophet()
    model.fit(df)


if __name__ == "__main__":
    # Load preprocessed data
    base_dir = "/opt/ml/processing"
    
    df = pd.read_csv(f"{base_dir}/output/train.csv")
    train()