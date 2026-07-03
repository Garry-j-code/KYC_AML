from src.state import AuditEntry, CaseState


def risk_scoring_node(state: CaseState) -> dict:
    """Deterministic, explainable aggregation — NOT a black box."""
    tier = "LOW"
    reasons = []
    if any(h.entity_type == "sanction" and h.match_score >= 92 for h in state.screening_hits):
        tier = "HIGH"
        reasons.append("strong sanctions match")
    elif any(h.entity_type == "pep" for h in state.screening_hits):
        tier = "HIGH"
        reasons.append("PEP match")
    elif (state.transaction_risk_score or 0) >= 0.8:
        tier = "HIGH"
        reasons.append("high transaction risk")
    elif state.completeness_issues:
        tier = "MEDIUM"
        reasons.append("incomplete KYC")
    elif (state.transaction_risk_score or 0) >= 0.4:
        tier = "MEDIUM"
        reasons.append("moderate transaction risk")

    return {
        "risk_tier": tier,
        "audit_log": [
            AuditEntry(
                node="risk_scoring",
                summary=f"tier={tier}",
                payload={"reasons": reasons},
            )
        ],
    }
