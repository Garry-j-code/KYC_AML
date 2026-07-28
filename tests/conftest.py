import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Ensure agent modules that construct a chat client at import time can be
# imported without real credentials. Tests mock the LLM itself.
os.environ.setdefault("GROQ_API_KEY", "test-key")
