import pandas as pd
import pytest

from src.data.sanctions import screen
from src.state import ScreeningHit


@pytest.fixture
def mini_sanctions_csv(tmp_path, monkeypatch):
    csv = tmp_path / "opensanctions_targets.csv"
    df = pd.DataFrame(
        {
            "id": ["san-1", "pep-1", "clean-1", "junk-1"],
            "schema": ["Person", "Person", "Person", "Person"],
            "name": ["Vladimir Putin", "John Smith PEP Official", "Random Citizen", "Person"],
            "aliases": ["", "", "", ""],
            "birth_date": ["", "", "", ""],
            "countries": ["RU", "US", "US", ""],
            "addresses": ["", "", "", ""],
            "identifiers": ["", "", "", ""],
            "sanctions": ["OFAC SDN", "", "", ""],
            "phones": ["", "", "", ""],
            "emails": ["", "", "", ""],
            "dataset": ["us_ofac_sdn", "everypolitician", "test", "junk"],
            "first_seen": ["", "", "", ""],
            "last_seen": ["", "", "", ""],
            "last_change": ["", "", "", ""],
        }
    )
    df.to_csv(csv, index=False)
    monkeypatch.setattr("src.data.sanctions.SANCTIONS_CSV", csv)
    from src.data import sanctions

    sanctions._load.cache_clear()
    yield csv
    sanctions._load.cache_clear()


def test_sanctioned_name_matches(mini_sanctions_csv):
    hits = screen("Vladimir Putin", threshold=85)
    assert len(hits) >= 1
    assert hits[0].match_score >= 85
    assert hits[0].entity_type == "sanction"


def test_random_name_no_hit(mini_sanctions_csv):
    hits = screen("Totally Unknown Individual", threshold=88)
    assert hits == []


def test_junk_single_token_name_no_false_positive(mini_sanctions_csv):
    # A clean customer whose name shares only the generic token "Person" with a
    # junk sanctions row must NOT be flagged (regression for over-matching).
    hits = screen("Zxqwerty Nonexistent Person", threshold=88)
    assert all(h.matched_name.lower() != "person" for h in hits)
    assert hits == []


def test_reordered_name_still_matches(mini_sanctions_csv):
    # Token reordering with an extra middle name should still match on 2 tokens.
    hits = screen("Putin Vladimir Vladimirovich", threshold=85)
    assert any(h.matched_name == "Vladimir Putin" for h in hits)


def test_empty_name_returns_empty():
    assert screen("") == []
