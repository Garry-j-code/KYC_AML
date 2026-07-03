"""Train the LightGBM AML classifier on IBM transaction data."""

from src.data.transactions import train

if __name__ == "__main__":
    train()
