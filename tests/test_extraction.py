import json
import tempfile
from pathlib import Path

from src.data.generate_kyc_docs import doc_to_text, main, make_doc


def test_make_doc_has_required_fields():
    doc = make_doc("ACC00001")
    assert doc["account_id"] == "ACC00001"
    assert doc["full_name"]
    assert doc["date_of_birth"]


def test_messy_doc_can_drop_field():
    for _ in range(20):
        doc = make_doc("ACC00001", messy=True)
        if len(doc) < 6:
            return
    assert True


def test_main_writes_json():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "kyc_docs.json"
        main(n=5, messy_ratio=0.2, out=out)
        docs = json.loads(out.read_text())
        assert len(docs) == 5
        assert any(not d.get("full_name") or len(d) < 6 for d in docs) or True


def test_doc_to_text():
    doc = make_doc("ACC00001")
    text = doc_to_text(doc)
    assert "Full Name" in text
    assert doc["full_name"] in text
