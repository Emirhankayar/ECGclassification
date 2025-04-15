import pathlib
import numpy as np
import pandas as pd
import Modules.constants as constants


def load_data(data_dir, expected_shape=(5000 // constants.WINDOW_SIZE, 12)):
    X = []
    y = []

    data_dir = pathlib.Path(data_dir)

    for label_dir in data_dir.iterdir():
        if label_dir.is_dir():
            label = int(label_dir.name)

            for csv_file in label_dir.glob("*.csv"):
                try:
                    data = pd.read_csv(csv_file, header=None).astype(np.float32).values

                    if data.shape != expected_shape:
                        print(
                            f"[ !! ]Skipping {csv_file.name}: Unexpected shape {data.shape}"
                        )
                        continue

                    if np.isnan(data).any():
                        print(f"[ !! ]Skipping {csv_file.name}: Contains NaNs")
                        continue

                    X.append(data)
                    y.append(label)

                except Exception as e:
                    print(f"[ XX ] Failed to load {csv_file}: {e}")

    X = np.stack(X, axis=0, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"[ OK ] Loaded {X.shape[0]} samples with shape {X.shape[1:]}")

    return X, y
