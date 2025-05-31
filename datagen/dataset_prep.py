import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def preprocess_dataset(directory="Dataset", output_directory="DataProcessed"):
    dataset_dir = Path(directory)
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        (output_dir / split).mkdir(exist_ok=True)

    file_labels = []

    for class_dir in sorted(dataset_dir.iterdir()):
        if class_dir.is_dir():
            class_label = class_dir.stem
            (output_dir / "train" / class_label).mkdir(parents=True, exist_ok=True)
            (output_dir / "val" / class_label).mkdir(parents=True, exist_ok=True)
            (output_dir / "test" / class_label).mkdir(parents=True, exist_ok=True)

            for patient_file in class_dir.glob("*.h5"):
                file_labels.append((patient_file, class_label))

    files, labels = zip(*file_labels)

    X_train, X_temp, y_train, y_temp = train_test_split(
        files, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    def process_and_save(files, labels, split, scaler=None):
        for file, label in tqdm(
            zip(files, labels), desc=f"Processing {split} files", unit="file"
        ):
            with h5py.File(file, "r") as f:
                if "sequences" in f:
                    sequences = f["sequences"][:]

                    if sequences.shape != (5000, 12):
                        print(
                            f"Skipping {file.stem}, incorrect shape: {sequences.shape}"
                        )
                        continue

                    if split == "train":
                        scaler = MinMaxScaler()
                        sequences = scaler.fit_transform(sequences)
                    else:
                        sequences = scaler.transform(sequences)

                    save_path = output_dir / split / label / file.name
                    with h5py.File(save_path, "w") as f_out:
                        f_out.create_dataset("sequences", data=sequences)
                else:
                    print(f"Missing 'sequences' dataset in {file.stem}")
        return scaler

    scaler = None
    scaler = process_and_save(X_train, y_train, "train", scaler)
    scaler = process_and_save(X_val, y_val, "val", scaler)
    scaler = process_and_save(X_test, y_test, "test", scaler)


if __name__ == "__main__":
    preprocess_dataset("Dataset", "DataProcessed")
