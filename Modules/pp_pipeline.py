import os
import time
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
        self.patient_dict = {}
        self._initialize()

    def _initialize(self):
        print("\n[ ii ] Initializing data preprocessing module...")
        print(
            f"\n\n[ ?? ] (#_samples,{5000//constants.WINDOW_SIZE},12) is the final data shape."
        )

    def _extract(self, input_dir, output_dir):

        try:
            with zipfile.ZipFile(input_dir, "r") as zip_ref:
                files = zip_ref.namelist()
                print(f"\n[ >> ] Preparing to unzip {len(files)} files...")

                pbar = tqdm(
                    total=len(files), desc="[    ] Unzipping content... ", unit="file"
                )

                def extract_file(file):
                    zip_ref.extract(file, output_dir)
                    pbar.update(1)

                with concurrent.futures.ThreadPoolExecutor(os.cpu_count()) as executor:
                    executor.map(extract_file, files)

        except FileNotFoundError:
            print(f"\n[ !! ] Error: The file '{input_dir}' was not found.")
        except zipfile.BadZipFile:
            print(f"\n[ !! ] Error: The file '{input_dir}' is not a valid zip file.")
        except Exception as e:
            print(f"\n[ !! ] An error occurred while extracting the files: {e}")

    def read_xlsx(self, input_dir):
        """
        Step 1: Read Excel and map values to Rhythm.
        Step 2: Match filenames with CSV files in directory.
        Step 3: Create /0, /1, /2, /3 directories.
        Step 4: Move files to respective directories based on Rhythm.
        """
        try:
            if not input_dir.exists():
                print(f"\n[ !! ] Error: The file '{input_dir}' was not found.")
                return

            df = pd.read_excel(input_dir, usecols=["FileName", "Rhythm"])
            df["Rhythm"] = df["Rhythm"].replace(
                {
                    "AF": 0,
                    "AFIB": 0,
                    "SVT": 1,
                    "AT": 1,
                    "SAAWR": 1,
                    "ST": 1,
                    "AVNRT": 1,
                    "AVRT": 1,
                    "SB": 2,
                    "SA": 3,
                    "SR": 3,
                }
            )

            print(df["Rhythm"].unique())

            for i in range(4):
                (constants.DATASET / str(i)).mkdir(parents=True, exist_ok=True)

            patient_files = list(constants.CSV_PATH.glob("*.csv"))
            patient_map = {f.stem: f for f in patient_files}

            def move_file(row):
                file_name = row["FileName"]
                rhythm = row["Rhythm"]
                if file_name in patient_map:
                    file_path = patient_map[file_name]
                    target_dir = constants.DATASET / str(rhythm)
                    destination = target_dir / file_path.name
                    file_path.rename(destination)
                    print(f"Moved {file_path.name} to {target_dir}")
                else:
                    print(f"File {file_name} not found.")

            with concurrent.futures.ThreadPoolExecutor(os.cpu_count()) as executor:
                executor.map(move_file, [row for _, row in df.iterrows()])

            moved_files = {
                f.stem
                for i in range(4)
                for f in (constants.DATASET / str(i)).glob("*.csv")
            }

            expected_files = set(df["FileName"].astype(str))
            missing_after_move = sorted(expected_files - moved_files)

            if missing_after_move:
                print(
                    f"\n[ !! ] {len(missing_after_move)} file(s) NOT found after move:"
                )
                for file in missing_after_move:
                    print(f" - {file}.csv")
            else:
                print(
                    "\n[ OK ] All expected files successfully moved to dataset subdirectories."
                )
        except Exception as e:
            print(f"Some error occurred: {e}")

    def split_dataset(self, input_dir):
        try:
            class_dirs = [input_dir / str(i) for i in range(4)]

            for split in ["train", "val", "test"]:
                for i in range(4):
                    target_dir = input_dir / split / str(i)
                    target_dir.mkdir(parents=True, exist_ok=True)

            for class_dir in class_dirs:
                label = class_dir.name
                files = list(class_dir.glob("*.csv"))

                if not files:
                    print(f"\n[ !! ] No files found in {class_dir}")
                    continue

                train_files, temp_files = sklearn.model_selection.train_test_split(
                    files, test_size=0.3, random_state=42, shuffle=True
                )
                val_files, test_files = sklearn.model_selection.train_test_split(
                    temp_files, test_size=0.5, random_state=42, shuffle=True
                )

                splits = {
                    "train": train_files,
                    "val": val_files,
                    "test": test_files,
                }

                for split, file_list in splits.items():
                    for file_path in file_list:
                        dest = input_dir / split / label / file_path.name
                        file_path.rename(dest)
                        # print(f"Moved {file_path.name} to {dest}")

                print(
                    f"\n[ OK ] Split done for class {label}: "
                    f"{len(train_files)} train, {len(val_files)} val, {len(test_files)} test"
                )

                for directory in Path(constants.DATASET).iterdir():
                    if directory.exists() and directory.is_dir():
                        if not any(directory.iterdir()):
                            directory.rmdir()

        except Exception as e:
            print(f"\n[ !! ] Error in proc_dataset: {e}")

    def bin_array(self, data, window_size):
        sequence_length, num_channels = data.shape
        new_sequence_length = sequence_length // window_size
        reshaped = data[: new_sequence_length * window_size].reshape(
            new_sequence_length, window_size, num_channels
        )
        return np.mean(reshaped, axis=1)

    def proc_dataset(self, input_dir):
        """
        - read files
        - make sure float 32 sequences
        - bin sequences // constants.WINDOW_SIZE
        - shape check if not shape then eliminate
        - make sure no flats max - min th, directly = 0 check
        - apply minmax scaling
        - while continue also rm the skipped files
        - save all the preprocessed files back to where they belong.

        """
        try:
            files = list(Path(input_dir).rglob("*.csv"))
            scaler = sklearn.preprocessing.MinMaxScaler()
            for file_path in files:
                data = pd.read_csv(file_path, header=None).astype(np.float32).values

                if data.size == 0:
                    print(f"[ !! ] Skipping {file_path.name}: Empty file")
                    file_path.unlink(missing_ok=True)
                    continue

                # data = self.bin_array(data, constants.WINDOW_SIZE)  # BINNING
                if constants.WINDOW_SIZE > 1:
                    data = self.bin_array(data, constants.WINDOW_SIZE)

                if data.size == 0 or np.isnan(data).any():
                    print(
                        f"[ !! ] Skipping {file_path.name}: Invalid or NaN after binning"
                    )
                    file_path.unlink(missing_ok=True)
                    continue

                if (
                    data.shape[1] != 12
                    or data.shape[0] != 5000 // constants.WINDOW_SIZE
                ):
                    print(
                        f"[ !! ] Skipping and removing {file_path.name} due to invalid shape. {data.shape}"
                    )
                    file_path.unlink(missing_ok=True)
                    continue

                if np.max(data) - np.min(data) < 0.2 or np.all(data == data[0]):
                    print(
                        f"[ !! ] Skipping and removing {file_path.name} due to flat signal."
                    )
                    file_path.unlink(missing_ok=True)
                    continue

                root_dir = file_path.parent.parent.name
                if root_dir == "train":
                    data = scaler.fit_transform(data)
                else:
                    data = scaler.transform(data)

                if np.isnan(data).any():
                    print(f"[ !! ] Skipping {file_path.name}: NaNs after scaling")
                    file_path.unlink(missing_ok=True)
                    continue

                pd.DataFrame(data.astype(np.float32)).to_csv(
                    file_path, header=False, index=False
                )
                # print(f"\n[ OK ] Saved to {file_path}")
            print(f"\n[ OK ] Preprocessing is done. {file_path}")

        except Exception as e:
            print(f"An error occured while preprocessing dataset:{e}")

    def dummy_excel(self, input_dir):
        try:
            if not input_dir.exists():
                print(f"[ !! ] File not found: {input_dir}")
                return

            backup_path = input_dir.with_name(
                f"{input_dir.stem}Testing{input_dir.suffix}"
            )

            shutil.copy2(input_dir, backup_path)
            print(f"\n[ OK ] Backup created at: {backup_path}")

            df = pd.read_excel(input_dir)

            if "Rhythm" in df.columns:
                df["Rhythm"] = np.nan
                df.to_excel(backup_path, index=False)
                print(f"[ OK ] 'Rhythm' column cleared in {backup_path}")
            else:
                print("[ !! ] Column 'Rhythm' not found in the Excel file.")

        except Exception as e:
            print(f"[ !! ] Error occurred: {e}")

    def get_dir_size(self, path):
        total_bytes = sum(
            f.stat().st_size for f in Path(path).rglob("*") if f.is_file()
        )
        total_gb = total_bytes / (1024**3)
        print(f"Directory size: {total_gb:.2f} GB")

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
    pp.split_dataset(constants.DATASET)
    pp.proc_dataset(constants.DATASET)
    pp.dummy_excel(constants.XLSX_PATH)
    rm_dir_list = [constants.CSV_PATH, constants.ZIP_CONTENT]
    pp.rm_dirs(rm_dir_list)
    pp.get_dir_size(constants.ZIP_CONTENT_OUTPUT)
    elapsed_time = time.time() - start_time
    print(f"\n Elapsed_time {elapsed_time}")
