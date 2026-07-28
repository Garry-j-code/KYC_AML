"""Tests for the transaction feature engineering, training, and scoring.

Uses a small synthetic dataset with separable features so the model is
learnable in-test, and no real data or network is required.
"""

import pandas as pd

from src.data.transactions import FEATS, _features, score_account, train


def _synthetic_csv(path, n: int = 300):
    """Half benign, half laundering, with clearly separable features so a
    tiny model can learn the boundary deterministically."""
    rows = []
    for i in range(n):
        laundering = i % 2 == 0
        if laundering:
            rows.append(
                {
                    "Timestamp": f"2022/09/01 0{i % 10}:20",
                    "From Bank": "001",
                    "Account": "LAUND001",
                    "To Bank": "999",  # cross-bank
                    "Account.1": "CP_L",
                    "Amount Received": 10000.0,
                    "Receiving Currency": "US Dollar",
                    "Amount Paid": 10000.0,  # round
                    "Payment Currency": "Euro",  # currency mismatch
                    "Payment Format": "Wire",
                    "Is Laundering": 1,
                }
            )
        else:
            rows.append(
                {
                    "Timestamp": f"2022/09/01 1{i % 10}:37",
                    "From Bank": "001",
                    "Account": "BENIGN01",
                    "To Bank": "001",  # same bank
                    "Account.1": "CP_B",
                    "Amount Received": 1234.56,
                    "Receiving Currency": "US Dollar",
                    "Amount Paid": 1234.56,  # not round
                    "Payment Currency": "US Dollar",  # no mismatch
                    "Payment Format": "Cheque",
                    "Is Laundering": 0,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_features_produces_expected_columns(tmp_path):
    csv = tmp_path / "trans.csv"
    _synthetic_csv(csv, n=10)
    feats = _features(pd.read_csv(csv))
    for col in FEATS:
        assert col in feats.columns
    assert set(feats["is_round"].unique()) <= {0, 1}


def test_train_and_score_discriminates(tmp_path):
    csv = tmp_path / "trans.csv"
    model = tmp_path / "model.pkl"
    _synthetic_csv(csv, n=300)

    train(raw_path=csv, model_path=model)

    launder_score, launder_flagged = score_account("LAUND001", raw_path=csv, model_path=model)
    benign_score, _ = score_account("BENIGN01", raw_path=csv, model_path=model)

    # Regression for the saturation bug: a benign account must score well below
    # a clearly-laundering one (they must not both saturate to ~1.0).
    assert launder_score > benign_score
    assert benign_score < 0.5
    assert len(launder_flagged) > 0


def test_score_account_unknown_returns_zero(tmp_path):
    csv = tmp_path / "trans.csv"
    model = tmp_path / "model.pkl"
    _synthetic_csv(csv, n=50)
    train(raw_path=csv, model_path=model)

    score, flagged = score_account("NO_SUCH_ACCOUNT", raw_path=csv, model_path=model)
    assert score == 0.0
    assert flagged == []
