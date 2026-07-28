import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_GENERATED = ROOT / "data" / "generated"
DATA_MODELS = ROOT / "data" / "models"

# Groq model IDs. gpt-oss-120b for reasoning/SAR, gpt-oss-20b for high-volume
# extraction. Both support structured/tool output. Override via .env.
MODEL_STRONG = os.getenv("MODEL_STRONG", "openai/gpt-oss-120b")
MODEL_CHEAP = os.getenv("MODEL_CHEAP", "openai/gpt-oss-20b")

SANCTIONS_CSV = DATA_RAW / "opensanctions_targets.csv"
TRANSACTIONS_CSV = DATA_RAW / "HI-Small_Trans.csv"
AML_MODEL_PATH = DATA_MODELS / "lgbm_aml.pkl"
KYC_DOCS_PATH = DATA_GENERATED / "kyc_docs.json"
