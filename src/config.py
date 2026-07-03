import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_GENERATED = ROOT / "data" / "generated"
DATA_MODELS = ROOT / "data" / "models"

MODEL_STRONG = os.getenv("MODEL_STRONG", "claude-sonnet-4-6")
MODEL_CHEAP = os.getenv("MODEL_CHEAP", "claude-haiku-4-5")

SANCTIONS_CSV = DATA_RAW / "opensanctions_targets.csv"
TRANSACTIONS_CSV = DATA_RAW / "HI-Small_Trans.csv"
AML_MODEL_PATH = DATA_MODELS / "lgbm_aml.pkl"
KYC_DOCS_PATH = DATA_GENERATED / "kyc_docs.json"
