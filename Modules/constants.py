"""
WINDOW_SIZE = TOTAL 5000,
   division by integer, to obtain desired result
   in below case num_samples,250,12
   5000 // 20 = 250 is the number needed if 250 rows wanted
"""

WINDOW_SIZE = 10


from pathlib import Path

PROJECT_DIR = Path("./")
ZIP_PATH = PROJECT_DIR / "Data.zip"
XLSX_PATH = PROJECT_DIR / "Data/Diagnostics.xlsx"
ZIP_CONTENT = PROJECT_DIR / "Data/ECGDataDenoised.zip"
CSV_PATH = PROJECT_DIR / "Data/ECGDataDenoised/"
ZIP_CONTENT_OUTPUT = PROJECT_DIR / "Data/"
DATASET = PROJECT_DIR / "Data" / "Dataset"


"""
 DO NOT MODIFY !
"""
DATA_TEMP = PROJECT_DIR / "Data" / "Data_tmp"
