import os

n_threads = str(os.cpu_count())
os.environ["OMP_NUM_THREADS"] = n_threads
os.environ["MKL_NUM_THREADS"] = n_threads
os.environ["OPENBLAS_NUM_THREADS"] = n_threads
os.environ["NUMEXPR_NUM_THREADS"] = n_threads
import pathlib
import imblearn
import Modules.constants as constants
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import Counter

# DATA_PATH = constants.DATASET
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "Data" / "Dataset"


def load_data(data_dir):
    X = []
    y = []
    data_dir = pathlib.Path(data_dir)
    expected_shape = (constants.FINAL_SIZE, 12)
    for label_dir in data_dir.iterdir():
        if label_dir.is_dir():
            label = int(label_dir.name)
            for csv_file in label_dir.glob("*.csv"):
                try:
                    data = (
                        pd.read_csv(csv_file, header=None, engine="c", low_memory=False)
                        .astype(np.float32)
                        .values
                    )

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


def load_tf_data():
    tr = DATA_PATH / "train"
    vl = DATA_PATH / "val"
    tst = DATA_PATH / "test"
    X_train, y_train = load_data(tr)
    X_test, y_test = load_data(tst)
    X_val, y_val = load_data(vl)
    print("Unique classes in y:", np.unique(y_train))
    print("Datatype:", (X_train.dtype), (y_train.dtype))
    print(f"Min and Max of X_train: {np.min(X_train)}, {np.max(X_train)}")
    print(f"Min and Max of X_val: {np.min(X_val)}, {np.max(X_val)}")
    print(f"Min and Max of X_test: {np.min(X_test)}, {np.max(X_test)}")
    print(f"NaNs in X: {np.isnan(X_train).sum()}")
    print(f"Infs in X: {np.isinf(X_train).sum()}")
    print(f"Class distribution before SMOTE: {Counter(y_train)}")
    print(f"\n\n Applying oversampling via SMOTE")
    X_train_flat = X_train.reshape((X_train.shape[0], -1))
    smote = imblearn.over_sampling.SMOTE(random_state=42)
    X_resampled, y_train = smote.fit_resample(X_train_flat, y_train)
    X_train = X_resampled.reshape((-1, *X_train.shape[1:]))

    print(f"Class distribution after SMOTE: {Counter(y_train)}")

    return X_train, y_train, X_val, y_val, X_test, y_test
