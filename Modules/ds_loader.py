import h5py
import numpy as np

LABELS = {"y_0": 0, "y_1": 1, "y_2": 2, "y_3": 3}


def load_patient_data(file_path):
    with h5py.File(file_path, "r") as f:
        data = f["sequences"][:]
    return data


def load_all_data(directory, sample_percentage=1.0):
    all_data = []
    all_labels = []

    class_dirs = [directory / cls for cls in LABELS]

    for class_dir in class_dirs:
        label = LABELS[class_dir.name]
        patient_files = [file for file in class_dir.glob("*.h5")]

        num_samples = int(len(patient_files) * sample_percentage)
        patient_files = np.random.choice(patient_files, num_samples, replace=False)

        for file_path in patient_files:
            patient_data = load_patient_data(file_path)
            all_data.append(patient_data)
            all_labels.append(label)

    X = np.array(all_data)
    y = np.array(all_labels, dtype=np.int32)

    return X, y
