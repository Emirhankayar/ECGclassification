"""
WINDOW_SIZE = TOTAL 5000,
   division by integer, to obtain desired result
   in below case num_samples,250,12
   5000 // 20 = 250 is the number needed if 250 rows wanted
"""

WINDOW_SIZE = 5


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
