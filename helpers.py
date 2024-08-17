import os
import pandas as pd


def get_csv_dataset(file_path: str, dataset_path: str) -> pd.DataFrame:
    current_dir = os.path.dirname(file_path)
    dataset_path = os.path.join(current_dir, dataset_path)

    return pd.read_csv(dataset_path, on_bad_lines="warn")
