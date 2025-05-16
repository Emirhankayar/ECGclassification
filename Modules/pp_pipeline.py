import concurrent.futures
import os
import pathlib
import shutil
import time
import zipfile
import sklearn
import numpy as np
import pandas as pd

import constants

N_THREADS = str(os.cpu_count())
os.environ["OMP_NUM_THREADS"] = N_THREADS
os.environ["MKL_NUM_THREADS"] = N_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = N_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = N_THREADS


class DatasetProcessor:
    """
    patient_dict = dictionary of each patient
    _initialize = fancy way to start
    """

    def __init__(self):
        self.patient_dict = {}
        self._initialize()

    def _initialize(self):
        print("\n[ ii ] Initializing data preprocessing module...")
        print(f"\n\n[ ?? ] (#,{constants.FINAL_SIZE},12) is the final data shape.")

    def extract(self, input_dir, output_dir):
        with zipfile.ZipFile(input_dir, "r") as zf:
            members = [m for m in zf.infolist() if not m.is_dir()]

            print(f"[ >> ] Preparing to unzip {len(members)} files with threads")

            def extract_member(member):
                target_path = pathlib.Path(output_dir) / member.filename
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(target_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                return member.file_size

            total = 0
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=os.cpu_count()
            ) as executor:
                futures = [executor.submit(extract_member, m) for m in members]
                for future in concurrent.futures.as_completed(futures):
                    total += future.result()

        return total

    def get_csvs(self, input_dir):
        """
        get all the csvs,
        input_dir = path/to/csvs/
        """

        csv_files = list(pathlib.Path(input_dir).rglob("*.csv"))
        result = []

        def read_csv_file(file_path):
            data = (
                pd.read_csv(file_path, header=None, engine="c")
                .astype(np.float32)
                .values
            )
            print(f"Adding file to the list: {file_path}")
            return (file_path, data)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=os.cpu_count()
        ) as executor:
            futures = [
                executor.submit(read_csv_file, file_path) for file_path in csv_files
            ]
            for future in concurrent.futures.as_completed(futures):
                result.append(future.result())
        return result

    def bin_array(self, data, final_size):
        """
        final_size = define it in Modules/constants.py eg. final_size = 500
        Helps to obtain = 500,12 final shape
        data = content of each csv
        """

        sequence_length, num_channels = data.shape
        if sequence_length < final_size:
            pad_length = final_size - sequence_length
            padding = np.zeros((pad_length, num_channels), dtype=np.float32)
            return np.vstack([data, padding])
        if sequence_length == final_size:
            return data
        window_size = sequence_length // final_size
        new_sequence_length = final_size
        reshaped = data[: new_sequence_length * window_size].reshape(
            new_sequence_length, window_size, num_channels
        )
        return np.mean(reshaped, axis=1)

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
            f.stat().st_size for f in pathlib.Path(path).rglob("*") if f.is_file()
        )
        total_gb = total_bytes / (1024**3)
        print(f"Directory size: {total_gb:.2f} GB")

    def rm_dirs(self, directories: list[pathlib.Path]):
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

    def proc_dir(self, input_dir, final_size):
        # path/filename.csv, data
        for file_path, data in self.get_csvs(input_dir):
            nan_check = np.isnan(data).any()
            flat_check = np.max(data) - np.min(data) < 0.2 or np.all(data == data[0])
            if nan_check or flat_check:
                print(f"[ !! ] Skipping {file_path.name}")
                file_path.unlink(missing_ok=True)
                continue
            pd.DataFrame(self.bin_array(data, final_size)).to_csv(
                file_path, header=False, index=False
            )
        print("Mapping rhythms as integers")

        """
            Add extra column to label_map.xlsx
            Column = dir

            Final product

            FileName/ Rhythm/ Dir

            to load => read label.xlsx append paths as CSV_PATH/Dir/FileName.csv
        """

        map_file = pathlib.Path(constants.XLSX_PATH).with_name("Label_Map.xlsx")
        shutil.copy(constants.XLSX_PATH, map_file)
        # map_file = pathlib.Path(constants.ZIP_CONTENT_OUTPUT) / "Label_Map.xlsx"
        df = pd.read_excel(map_file, usecols=["FileName", "Rhythm"])
        df["Rhythm"] = df["Rhythm"].replace(constants.RHY_DICT)
        df = df[["FileName", "Rhythm"]]

        df.to_excel(map_file, index=False)
        self.dummy_excel(constants.XLSX_PATH)
        rm_dir_list = [constants.ZIP_CONTENT]
        self.rm_dirs(rm_dir_list)
        self.get_dir_size(constants.CSV_PATH)


if __name__ == "__main__":
    start = time.time()
    pp = DatasetProcessor()
    pp.extract(constants.ZIP_PATH, constants.PROJECT_DIR)
    pp.extract(constants.ZIP_CONTENT, constants.ZIP_CONTENT_OUTPUT)
    pp.proc_dir(constants.CSV_PATH, constants.FINAL_SIZE)
    end = time.time() - start
    print(f"Execution Time : {end:.4f} seconds")
