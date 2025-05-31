import h5py
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def check_dataset(directory="Dataset"):
    dataset_dir = Path(directory)

    if not dataset_dir.exists():
        print("Dataset directory not found!")
        return

    seen_files = defaultdict(list)

    for class_dir in sorted(dataset_dir.iterdir()):
        if class_dir.is_dir():
            class_label = class_dir.stem
            print(f"\nChecking class: {class_label}")

            patient_files = list(class_dir.glob("*.h5"))
            if not patient_files:
                print(f"No patient files found in {class_label}")
                continue

            for patient_file in tqdm(
                patient_files, desc=f"Processing {class_label}", unit="file"
            ):
                filename = patient_file.stem
                seen_files[filename].append(class_label)

            first_file = patient_files[0]
            with h5py.File(first_file, "r") as f:
                if "sequences" in f:
                    sequences = f["sequences"][:]
                    print(f"Patient: {first_file.stem} | Shape: {sequences.shape}")

                    if sequences.shape != (5000, 12):
                        print("Data shape is incorrect! Expected (5000, 12)")
                else:
                    print(f"Missing 'sequences' dataset in {first_file.stem}")

    duplicates = {k: v for k, v in seen_files.items() if len(v) > 1}
    if duplicates:
        print("\nDuplicate Files Found!")
        for filename, classes in duplicates.items():
            print(f" - {filename} appears in: {', '.join(classes)}")
    else:
        print("\nNo duplicate filenames found across directories.")


if __name__ == "__main__":
    check_dataset("Dataset")
