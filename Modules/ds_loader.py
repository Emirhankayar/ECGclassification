import pandas as pd
import numpy as np
import pathlib


class DatasetLoader:
    def __init__(self, xlsx_path, data_dir):
        self.xlsx_path = pathlib.Path(xlsx_path)
        self.data_dir = pathlib.Path(data_dir)
        self.df = pd.read_excel(self.xlsx_path, usecols=["FileName", "Rhythm"])

    def load_data(self):
        X = []
        y = []

        for _, row in self.df.iterrows():
            file_name = row["FileName"]
            label = row["Rhythm"]

            file_path = self.data_dir / f"{file_name}.csv"
            if file_path.exists():
                data = pd.read_csv(file_path, header=None).values

                X.append(data)
                y.append(label)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)

        return X, y


# Example usage:
# dataset_loader = DatasetLoader(xlsx_path='path_to_label_file.xlsx', data_dir='path_to_data_directory')
# X, y = dataset_loader.load_data()
