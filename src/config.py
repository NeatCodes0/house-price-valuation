from pathlib import Path

# Base Directory Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DUCKDB_PATH = DATA_DIR / "housing.duckdb"
MODELS_DIR = BASE_DIR / "models"
SQL_DIR = BASE_DIR / "sql"

# Raw Data Paths
RAW_DATA_PATH = RAW_DATA_DIR / "train.csv"
DATA_DESCRIPTION_PATH = RAW_DATA_DIR / "data_description.txt"

# Core Modeling Constants
DE_COCK_OUTLIER_THRESHOLD = 4000
TARGET_COL = "SalePrice"
RANDOM_STATE = 42