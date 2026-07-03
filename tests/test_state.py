import operator

from src.state import AuditEntry, CaseState, ExtractedIdentity


def test_case_state_construction():
    state = CaseState(customer_id="C1", raw_document="test doc")
    assert state.customer_id == "C1"
    assert state.audit_log == []


def test_audit_log_reducer():
    entry = AuditEntry(node="test", summary="hello")
    update = {"audit_log": [entry]}
    merged = CaseState.model_validate({"customer_id": "C1", **update})
    assert len(merged.audit_log) == 1
    assert merged.audit_log[0].node == "test"


def test_extracted_identity_optional_fields():
    identity = ExtractedIdentity(full_name="Jane Doe")
    assert identity.date_of_birth is None
    assert identity.account_id is None
