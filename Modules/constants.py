"""
FINAL_SIZE =5000 // DYNAMIC WINDOW SIZE
(FINAL_SIZE, 12)
"""

FINAL_SIZE = 1000

from pathlib import Path

PROJECT_DIR = Path("./")
ZIP_PATH = PROJECT_DIR / "Data.zip"
XLSX_PATH = PROJECT_DIR / "Data/Diagnostics.xlsx"
ZIP_CONTENT = PROJECT_DIR / "Data/ECGDataDenoised.zip"
CSV_PATH = PROJECT_DIR / "Data/ECGDataDenoised"
ZIP_CONTENT_OUTPUT = PROJECT_DIR / "Data/"
DATASET = PROJECT_DIR / "Data/Dataset"

"""
 MODEL TO LOAD FOR GUI
"""
LOAD_MODEL = PROJECT_DIR / "Results/Model_Weights/RES_W500_L4/RES_W500_L4.keras"
MODELS_DIR = PROJECT_DIR / "src/Models"
"""
 DO NOT MODIFY !
"""
DATA_TEMP = PROJECT_DIR / "Data" / "tmp"
RHYTHM_MAPPING = {
    "AFIB": "AFIB",
    "AF": "AFIB",
    "SVT": "GSVT",
    "AT": "GSVT",
    "SAAWR": "GSVT",
    "ST": "GSVT",
    "AVNRT": "GSVT",
    "AVRT": "GSVT",
    "SB": "SB",
    "SR": "SR",
    "SA": "SR",
}

RHY_DICT = {
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
