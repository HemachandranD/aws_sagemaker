from prophet import Prophet
import pandas as pd


def train():
    # Initialize and fit Prophet model
    model = Prophet()
    model.fit(df)


if __name__ == "__main__":
    train_data_path = "/opt/ml/input/data/train"  # Adjust the filename if necessary
    df = pd.read_csv(f"{train_data_path}/train.csv")
    train()