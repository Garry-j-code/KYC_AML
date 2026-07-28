import re
from functools import lru_cache

import pandas as pd
from rapidfuzz import fuzz, process

from src.config import SANCTIONS_CSV
from src.state import ScreeningHit


def _tokens(value: str) -> set[str]:
    """Alphanumeric tokens (len > 1), lowercased, for overlap checks."""
    return {t for t in re.findall(r"[a-z0-9]+", value.lower()) if len(t) > 1}


@lru_cache(maxsize=1)
def _load() -> pd.DataFrame:
    df = pd.read_csv(SANCTIONS_CSV, low_memory=False)
    df = df[df["schema"] == "Person"].copy()
    df["name"] = df["name"].fillna("").str.strip()
    # Drop empty and obvious placeholder/junk names (e.g. a bare "Person").
    df = df[df["name"].str.len() >= 3]
    df = df[df["name"].str.lower() != "person"]
    return df.reset_index(drop=True)


def screen(name: str, threshold: int = 88, limit: int = 3) -> list[ScreeningHit]:
    if not name:
        return []
    df = _load()
    names = df["name"].tolist()
    query_tokens = _tokens(name)
    # WRatio ranks candidates well but partial-matches short junk names against
    # long queries. Fetch extra candidates so the token-overlap guard below has
    # room to reject weak single-token matches without starving real ones.
    matches = process.extract(name, names, scorer=fuzz.WRatio, limit=limit * 3)
    hits = []
    for matched_name, score, idx in matches:
        if len(hits) >= limit:
            break
        if score < threshold:
            continue
        # Guard against single-common-token false positives: a multi-token name
        # must agree on at least two tokens (e.g. first + last), not just one.
        if len(query_tokens) >= 2 and len(query_tokens & _tokens(matched_name)) < 2:
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
