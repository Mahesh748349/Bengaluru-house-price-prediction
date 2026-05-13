from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SUBMISSION_PATH = OUTPUTS_DIR / "submission.csv"

TARGET_CANDIDATES = ["price", "Price", "target", "SalePrice", "sale_price"]
ID_CANDIDATES = ["id", "ID", "Id"]

RANDOM_STATE = 42
N_SPLITS = 5
