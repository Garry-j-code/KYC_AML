from functools import lru_cache

import pandas as pd
from rapidfuzz import fuzz, process

from src.config import SANCTIONS_CSV
from src.state import ScreeningHit


@lru_cache(maxsize=1)
def _load() -> pd.DataFrame:
    df = pd.read_csv(SANCTIONS_CSV, low_memory=False)
    df = df[df["schema"] == "Person"].copy()
    df["name"] = df["name"].fillna("").str.strip()
    df = df[df["name"] != ""]
    return df.reset_index(drop=True)


def screen(name: str, threshold: int = 88, limit: int = 3) -> list[ScreeningHit]:
    if not name:
        return []
    df = _load()
    names = df["name"].tolist()
    matches = process.extract(name, names, scorer=fuzz.WRatio, limit=limit)
    hits = []
    for matched_name, score, idx in matches:
        if score < threshold:
            continue
        row = df.iloc[idx]
        topics = str(row.get("sanctions", "")) + str(row.get("dataset", ""))
        entity_type = "pep" if "pep" in topics.lower() else "sanction"
        hits.append(
            ScreeningHit(
                matched_name=matched_name,
                match_score=float(score),
                entity_type=entity_type,
                source=str(row.get("dataset", "opensanctions")),
                details={
                    "id": str(row.get("id", "")),
                    "countries": str(row.get("countries", "")),
                },
            )
        )
    return hits
