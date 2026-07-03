from langchain_anthropic import ChatAnthropic

from src.config import MODEL_STRONG
from src.state import AuditEntry, CaseState

llm = ChatAnthropic(model=MODEL_STRONG, temperature=0.2)


def sar_draft_node(state: CaseState) -> dict:
    narrative = llm.invoke(
        "Draft a concise Suspicious Activity Report narrative for a compliance analyst. "
        f"Customer: {state.extracted_identity}. "
        f"Screening hits: {[h.model_dump() for h in state.screening_hits]}. "
        f"Flagged transactions: {state.flagged_transactions}. "
        "State the facts and the reason for escalation. Do not fabricate details."
    ).content
    return {
        "sar_narrative": narrative,
        "decision": "ESCALATED_TO_HUMAN",
        "audit_log": [AuditEntry(node="sar_draft", summary="SAR narrative drafted")],
    }
