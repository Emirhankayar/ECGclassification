import time
import h5py
import shutil
import zipfile
import sklearn
import constants
import concurrent
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path


class DatasetProcessor:
    def __init__(self):
        self.label_encoder = sklearn.preprocessing.LabelEncoder()
        self.rhythm_mapping = constants.RHYTHM_MAPPING
        self.patient_dict = {}
        self._initialize()

    def _initialize(self):
        print("\n[ ii ] Initializing data preprocessing module...")
        print(
            f"\n\n[ ?? ] (#_samples,{5000//constants.WINDOW_SIZE},12) is the final data shape."
        )
        encoded_classes = list(self.rhythm_mapping.values())
        self.label_encoder.fit(encoded_classes)

    def _extract(self, input_dir, output_dir):

        try:
            with zipfile.ZipFile(input_dir, "r") as zip_ref:
                files = zip_ref.namelist()
                tqdm.write(f"\n[ >> ] Preparing to unzip {len(files)} files...")

                pbar = tqdm(
                    total=len(files), desc="[    ] Unzipping content... ", unit="file"
                )

                def extract_file(file):
                    zip_ref.extract(file, output_dir)
                    pbar.update(1)

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    executor.map(extract_file, files)

            tqdm.write(f"\n[ XX ] Extracted all files to {output_dir}")
            time.sleep(5)

        except FileNotFoundError:
            tqdm.write(f"\n[ !! ] Error: The file '{input_dir}' was not found.")
        except zipfile.BadZipFile:
            tqdm.write(
                f"\n[ !! ] Error: The file '{input_dir}' is not a valid zip file."
            )
        except Exception as e:
            tqdm.write(f"\n[ !! ] An error occurred while extracting the files: {e}")

    def read_xlsx(self, input_dir):
        try:
            print(f"\n[ ii ] Reading file : {input_dir}")
            df = pd.read_excel(
                input_dir, engine="openpyxl", usecols=["FileName", "Rhythm"]
            )

            df["Mapped_Rhythm"] = df["Rhythm"].map(self.rhythm_mapping)
            df = df.dropna(subset=["Mapped_Rhythm"])
            df["Target"] = self.label_encoder.transform(df["Mapped_Rhythm"])
            df["Target"] = df["Target"].astype("int32")

            self.patient_dict = df.set_index("FileName")["Target"].to_dict()

            print(f"\n[ XX ] Total patient entries loaded: {len(self.patient_dict)}")
        except FileNotFoundError:
            print(f"\n[ !! ] Error: The file '{input_dir}' was not found.")
        except Exception as e:
            print(f"\n[ !! ] An error occurred: {e}")

    def read_csv(self, input_dir):
        try:
            print(f"\n[ ii ] Reading all files at {input_dir}")
            files = list(Path(input_dir).glob("*.csv"))

            def process_file(file):
                filename = file.stem
                if filename in self.patient_dict:
                    target_class = self.patient_dict[filename]
                    df = pd.read_csv(file, header=None)
                    df = df.to_numpy(dtype=np.float32)
                    sequences = self.bin_array(df, constants.WINDOW_SIZE)
                    if (
                        sequences.shape[1] == 12
                        and sequences.shape[0] == 5000 // constants.WINDOW_SIZE
                    ):
                        if not np.isnan(sequences).any():
                            file_path = (
                                constants.DATA_TEMP
                                / str(target_class)
                                / f"{filename}.h5"
                            )
                            self.save_hdf5(file_path, sequences)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.map(process_file, files)

            print(f"\n[ XX ] All data is processed and saved.")
        except FileNotFoundError:
            print(f"\n[ !! ] Error: The file '{input_dir}' was not found.")
        except Exception as e:
            print(f"\n[ !! ] An error occurred: {e}")

    def bin_array(self, data, window_size):
        sequence_length, num_channels = data.shape
        new_sequence_length = sequence_length // window_size
        reshaped = data[: new_sequence_length * window_size].reshape(
            new_sequence_length, window_size, num_channels
        )
        return np.mean(reshaped, axis=1)

    def save_hdf5(self, file_path, sequences):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(file_path, "w") as f:
            f.create_dataset("X", data=sequences)

        print(f"[ OK ] Saved {file_path.name}")

    def preprocess_files(self, file, scaler, output_dir):
        try:
            with h5py.File(file, "r+") as f:  # 'r+' allows overwriting
                if "X" in f:
                    X = f["X"][:]
                    # y = f["y"][()]
                    # y = y.astype("int32")

                    if (
                        X.shape[1] == 12
                        and X.shape[0] == 5000 // constants.WINDOW_SIZE
                        and X.dtype == "float32"
                        and not np.isnan(X).any()
                    ):

                        reshaped = X.reshape(-1, X.shape[-1])
                        scaled = scaler.transform(reshaped)
                        X = scaled.reshape(X.shape)
                        # Check for flatlines
                        X_list = []
                        threshold = 0.2
                        for s in X:
                            if (
                                not np.all(s == s[0])
                                or np.max(s) - np.min(s) < threshold
                            ):
                                X_list.append(s)
                        if len(X_list) == 0:
                            print(
                                f"[ !! ] All sequences flat in {file.name}, skipping..."
                            )
                            return

                        X = np.stack(X_list)
                        # Overwrite the dataset
                        del f["X"]
                        #                        del f["y"]

                        f.create_dataset("X", data=X.astype("float32"))
                        #                        f.create_dataset("y", data=y.astype("int32"))

                        print(f"[ OK ] Preprocessed {file.name}")
                    else:
                        print(f"\n[ !! ] Shape or dtype mismatch in {file.name}")
                else:
                    print(f"\n[ !! ] Expected datasets not found in {file.name}")
        except Exception as e:
            print(f"\n[ !! ] Failed to read/write {file.name}: {e}")

    def preprocess_data(self, input_dir, output_dir):
        files = list(input_dir.rglob("*.h5"))
        file_class_pairs = [(file, file.parent.name) for file in files]

        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        for split in ["train", "val", "test"]:
            (output_dir / split).mkdir(exist_ok=True)

        file_paths, class_labels = zip(*file_class_pairs)
        train_files, temp_files, train_labels, temp_labels = (
            sklearn.model_selection.train_test_split(
                file_paths,
                class_labels,
                test_size=0.30,
                stratify=class_labels,
                random_state=42,
            )
        )
        val_files, test_files, val_labels, test_labels = (
            sklearn.model_selection.train_test_split(
                temp_files,
                temp_labels,
                test_size=0.50,
                stratify=temp_labels,
                random_state=42,
            )
        )
        split_map = {
            "train": list(zip(train_files, train_labels)),
            "val": list(zip(val_files, val_labels)),
            "test": list(zip(test_files, test_labels)),
        }

        for split, items in split_map.items():
            print(f"\n[ >> ] Copying {len(items)} files to '{split}' directory...")
            for file, label in items:
                class_dir = output_dir / split / label
                class_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(file, class_dir / file.name)

                try:
                    shutil.copy(file, class_dir / file.name)
                except Exception as e:
                    print(
                        f"\n[ !! ] Failed to copy {file.name} to {class_dir}/{file.name}: {e}"
                    )

        print("\n[ XX ] Data split and copy complete!")

        # Fit scaler on training data
        scaler = sklearn.preprocessing.MinMaxScaler()
        print(f"\n[ >> ] Fitting scaler on training data...")

        for file in (output_dir / "train").rglob("*.h5"):
            with h5py.File(file, "r") as f:
                X = f["X"][:]
                if not np.isnan(X).any():
                    reshaped = X.reshape(-1, X.shape[-1])
                    scaler.partial_fit(reshaped)

        print("[ OK ] Scaler fitted.")

        with concurrent.futures.ProcessPoolExecutor() as executor:
            for split in ["train", "val", "test"]:
                print(f"\n[ -- ] Preprocessing files in '{split}'...")
                split_dir = output_dir / split
                split_files = list(split_dir.glob("**/*.h5"))

                future_to_file = {
                    executor.submit(
                        self.preprocess_files, file, scaler, output_dir
                    ): file
                    for file in split_files
                }

                for future in concurrent.futures.as_completed(future_to_file):
                    file = future_to_file[future]
                    try:
                        future.result()
                    except Exception as exc:
                        print(
                            f"[ !! ] An error occurred while processing {file.name}: {exc}"
                        )

    def get_dir_size(self, path):
        total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / (
            1024 * 3
        )
        print(total)

    def rm_dirs(self, directories: list[Path]):
        for directory in directories:
            if directory.exists():
                if directory.is_dir():
                    print(f"\n[ ii ] Deleting directory: {directory}")
                    shutil.rmtree(directory)
                elif directory.is_file():
                    print(f"\n[ ii ] Deleting file: {directory}")
                    directory.unlink()
                else:
                    print(
                        f"\n[ !! ] Path {directory} is neither a directory nor a file."
                    )
            else:
                print(f"\n[ !! ] Path {directory} not found.")


if __name__ == "__main__":
    start_time = time.time()
    pp = DatasetProcessor()
    pp._extract(constants.ZIP_PATH, constants.PROJECT_DIR)
    pp._extract(constants.ZIP_CONTENT, constants.ZIP_CONTENT_OUTPUT)
    pp.read_xlsx(constants.XLSX_PATH)
    pp.read_csv(constants.CSV_PATH)
    pp.preprocess_data(constants.DATA_TEMP, constants.DATASET)
    rm_dir_list = [constants.DATA_TEMP, constants.CSV_PATH, constants.ZIP_CONTENT]
    pp.rm_dirs(rm_dir_list)
    pp.get_dir_size(constants.ZIP_CONTENT_OUTPUT)
    elapsed_time = time.time() - start_time
    print(f"\n Elapsed_time {elapsed_time}")
