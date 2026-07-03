import pandas as pd
import pytest

from src.data.sanctions import screen
from src.state import ScreeningHit


@pytest.fixture
def mini_sanctions_csv(tmp_path, monkeypatch):
    csv = tmp_path / "opensanctions_targets.csv"
    df = pd.DataFrame(
        {
            "id": ["san-1", "pep-1", "clean-1"],
            "schema": ["Person", "Person", "Person"],
            "name": ["Vladimir Putin", "John Smith PEP Official", "Random Citizen"],
            "aliases": ["", "", ""],
            "birth_date": ["", "", ""],
            "countries": ["RU", "US", "US"],
            "addresses": ["", "", ""],
            "identifiers": ["", "", ""],
            "sanctions": ["OFAC SDN", "", ""],
            "phones": ["", "", ""],
            "emails": ["", "", ""],
            "dataset": ["us_ofac_sdn", "everypolitician", "test"],
            "first_seen": ["", "", ""],
            "last_seen": ["", "", ""],
            "last_change": ["", "", ""],
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
    hits = screen("Totally Unknown Person XYZ", threshold=88)
    assert hits == []


def test_empty_name_returns_empty():
    assert screen("") == []
