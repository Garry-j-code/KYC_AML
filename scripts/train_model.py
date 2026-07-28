"""Train the LightGBM AML classifier on IBM transaction data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.transactions import train

if __name__ == "__main__":
    train()
