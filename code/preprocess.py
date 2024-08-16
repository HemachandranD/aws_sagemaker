import pandas as pd




if __name__ == "__main__":
    base_dir = "/opt/ml/processing"

    df = pd.read_csv(
    f"{base_dir}/input/raw_data.csv",
    header=None,
    )