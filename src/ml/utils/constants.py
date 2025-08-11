import os

PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_BASE_DIR = os.path.join(PROJECT_ROOT_DIR, ".data")
RAW_DATA_DIR = os.path.join(DATA_BASE_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_BASE_DIR, "processed")
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT_DIR, "experiments")
ROOT_DIR = PROJECT_ROOT_DIR
EMBEDDINGS_DIR = os.path.join(PROJECT_ROOT_DIR, "embeddings")

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
