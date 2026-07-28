"""Unit tests for the LLM-backed agent nodes.

The LLM (and the ML scorer) are mocked so these tests are hermetic and never
make network calls or require a trained model.
"""

from src.agents import intake, monitoring, sar_draft
from src.state import CaseState, ExtractedIdentity, ScreeningHit


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    """Stands in for a ChatAnthropic client; returns a canned response."""

    def __init__(self, response):
        self._response = response

    def invoke(self, _prompt):
        return self._response


# --- intake_node -----------------------------------------------------------


def test_intake_node_extracts_identity(monkeypatch):
    identity = ExtractedIdentity(
        full_name="Jane Doe",
        date_of_birth="1990-01-01",
        nationality="US",
        id_number="AB123456",
        account_id="ACC00001",
    )
    monkeypatch.setattr(intake, "llm", _FakeLLM(identity))

    out = intake.intake_node(CaseState(customer_id="C1", raw_document="..."))

    assert out["extracted_identity"].full_name == "Jane Doe"
    assert out["completeness_issues"] == []
    assert out["audit_log"][0].node == "intake"


def test_intake_node_flags_missing_fields(monkeypatch):
    identity = ExtractedIdentity(full_name="Jane Doe")  # missing dob/nationality/id
    monkeypatch.setattr(intake, "llm", _FakeLLM(identity))

    out = intake.intake_node(CaseState(customer_id="C1", raw_document="..."))

    assert set(out["completeness_issues"]) == {"date_of_birth", "nationality", "id_number"}
    assert out["audit_log"][0].payload["issues"] == out["completeness_issues"]


# --- monitoring_node -------------------------------------------------------


def test_monitoring_node_no_account_skips_llm(monkeypatch):
    # No account_id -> score_account is never called and the LLM is not invoked.
    def _fail(*_a, **_k):
        raise AssertionError("LLM should not be called when there are no flagged txns")

    monkeypatch.setattr(monitoring, "llm", _FakeLLM(_FakeMessage("unused")))
    monkeypatch.setattr(monitoring.llm, "invoke", _fail)

    state = CaseState(
        customer_id="C1",
        extracted_identity=ExtractedIdentity(full_name="Jane Doe", account_id=None),
    )
    out = monitoring.monitoring_node(state)

    assert out["transaction_risk_score"] == 0.0
    assert out["flagged_transactions"] == []
    assert out["audit_log"][0].node == "monitoring"
    assert out["audit_log"][0].payload["reasoning"] == ""


def test_monitoring_node_flagged_invokes_llm(monkeypatch):
    flagged = [{"Timestamp": "2022/09/01 00:20", "amount_paid": 9000.0, "risk": 0.97}]
    monkeypatch.setattr(monitoring, "score_account", lambda _acct: (0.97, flagged))
    monkeypatch.setattr(monitoring, "llm", _FakeLLM(_FakeMessage("Looks like structuring.")))

    state = CaseState(
        customer_id="C1",
        extracted_identity=ExtractedIdentity(full_name="Jane Doe", account_id="ACC00001"),
    )
    out = monitoring.monitoring_node(state)

    assert out["transaction_risk_score"] == 0.97
    assert out["flagged_transactions"] == flagged
    assert out["audit_log"][0].payload["reasoning"] == "Looks like structuring."


# --- sar_draft_node --------------------------------------------------------


def test_sar_draft_node_sets_narrative_and_decision(monkeypatch):
    monkeypatch.setattr(sar_draft, "llm", _FakeLLM(_FakeMessage("SAR narrative text.")))

    state = CaseState(
        customer_id="C1",
        extracted_identity=ExtractedIdentity(full_name="Bad Actor"),
        screening_hits=[
            ScreeningHit(
                matched_name="Bad Actor",
                match_score=95.0,
                entity_type="sanction",
                source="OFAC",
            )
        ],
    )
    out = sar_draft.sar_draft_node(state)

    assert out["sar_narrative"] == "SAR narrative text."
    assert out["decision"] == "ESCALATED_TO_HUMAN"
    assert out["audit_log"][0].node == "sar_draft"
