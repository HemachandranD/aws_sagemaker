import pandas as pd


def preprocess(df):
    processed_df = df.dropna(axis=0)
    return processed_df


if __name__ == "__main__":
    base_dir = "/opt/ml/processing"

    df = pd.read_csv(f"{base_dir}/input/synthetic_data.csv", header=None)
    processed_df = preprocess(df)
    processed_df.to_csv(f"{base_dir}/output/train/train.csv", index=False, header=False)