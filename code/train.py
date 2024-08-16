from prophet import Prophet
import pandas as pd



def train():
    # Initialize and fit Prophet model
    model = Prophet()
    model.fit(df)


if __name__ == "__main__":
    # The input data directory is typically stored in /opt/ml/input/data/<channel_name>
    train_input_path = os.environ['SM_CHANNEL_TRAIN']
    
    df = pd.read_csv(f"{train_input_path}/train.csv")
    train()