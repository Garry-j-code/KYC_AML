from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

from src.config import AML_MODEL_PATH, TRANSACTIONS_CSV

FEATS = ["hour", "cross_bank", "currency_mismatch", "amount_paid", "is_round", "pay_format"]


def _features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ts"] = pd.to_datetime(df["Timestamp"])
    df["hour"] = df["ts"].dt.hour
    df["cross_bank"] = (df["From Bank"] != df["To Bank"]).astype(int)
    df["currency_mismatch"] = (df["Receiving Currency"] != df["Payment Currency"]).astype(int)
    df["amount_paid"] = pd.to_numeric(df["Amount Paid"], errors="coerce").fillna(0)
    df["is_round"] = (df["amount_paid"] % 1000 == 0).astype(int)
    df["pay_format"] = df["Payment Format"].astype("category").cat.codes
    return df


def train(raw_path: Path | None = None, model_path: Path | None = None) -> lgb.LGBMClassifier:
    raw_path = raw_path or TRANSACTIONS_CSV
    model_path = model_path or AML_MODEL_PATH
    df = _features(pd.read_csv(raw_path))
    X, y = df[FEATS], df["Is Laundering"].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pos_weight = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    clf = lgb.LGBMClassifier(scale_pos_weight=pos_weight, n_estimators=300, verbose=-1)
    clf.fit(Xtr, ytr)
    y_pred = clf.predict(Xte)
    recall = recall_score(yte, y_pred, zero_division=0)
    print(f"Positive-class recall on holdout: {recall:.3f}")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)
    return clf


def score_account(
    account_id: str,
    top_k: int = 5,
    raw_path: Path | None = None,
    model_path: Path | None = None,
) -> tuple[float, list[dict]]:
    """Return an account-level risk score and its most suspicious transactions."""
    raw_path = raw_path or TRANSACTIONS_CSV
    model_path = model_path or AML_MODEL_PATH
    if not model_path.exists():
        return 0.0, []
    clf = joblib.load(model_path)
    df = _features(pd.read_csv(raw_path))
    acct = df[(df["Account"] == account_id) | (df["Account.1"] == account_id)]
    if acct.empty:
        return 0.0, []
    proba = clf.predict_proba(acct[FEATS])[:, 1]
    acct = acct.assign(risk=proba).sort_values("risk", ascending=False)
    flagged = acct.head(top_k)[["Timestamp", "amount_paid", "risk"]].to_dict("records")
    return float(proba.max()), flagged
