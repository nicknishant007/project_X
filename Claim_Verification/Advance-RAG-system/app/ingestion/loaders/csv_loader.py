import pandas as pd


def load_csv(file_path: str) -> str:

    dataframe = pd.read_csv(file_path)

    return dataframe.to_string()