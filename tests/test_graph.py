from src.agents.risk_scoring import risk_scoring_node
from src.agents.screening import screening_node
from src.state import AuditEntry, CaseState, ExtractedIdentity, ScreeningHit


def test_risk_scoring_low_tier():
    state = CaseState(
        customer_id="C1",
        extracted_identity=ExtractedIdentity(full_name="Jane Doe", account_id="ACC00001"),
    )
    out = risk_scoring_node(state)
    assert out["risk_tier"] == "LOW"
    assert len(out["audit_log"]) == 1
    assert out["audit_log"][0].node == "risk_scoring"


def test_risk_scoring_high_sanctions():
    state = CaseState(
        customer_id="C1",
        screening_hits=[
            ScreeningHit(
                matched_name="Bad Actor",
                match_score=95.0,
                entity_type="sanction",
                source="OFAC",
            )
        ],
    )
    out = risk_scoring_node(state)
    assert out["risk_tier"] == "HIGH"
    assert "strong sanctions match" in out["audit_log"][0].payload["reasons"]


def test_risk_scoring_medium_incomplete_kyc():
    state = CaseState(
        customer_id="C1",
        completeness_issues=["nationality"],
    )
    out = risk_scoring_node(state)
    assert out["risk_tier"] == "MEDIUM"


def test_screening_node_audit_entry():
    state = CaseState(
        customer_id="C1",
        extracted_identity=ExtractedIdentity(full_name=""),
    )
    out = screening_node(state)
    assert "audit_log" in out
    assert out["audit_log"][0].node == "screening"
