from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.agents.intake import intake_node
from src.agents.monitoring import monitoring_node
from src.agents.risk_scoring import risk_scoring_node
from src.agents.sar_draft import sar_draft_node
from src.agents.screening import screening_node
from src.state import AuditEntry, CaseState


def auto_approve_node(state: CaseState) -> dict:
    return {
        "decision": "AUTO_APPROVED",
        "audit_log": [AuditEntry(node="auto_approve", summary="low risk, approved")],
    }


def request_info_node(state: CaseState) -> dict:
    return {
        "decision": "INFO_REQUESTED",
        "audit_log": [
            AuditEntry(node="request_info", summary="medium risk, more info needed")
        ],
    }


def human_review_node(state: CaseState) -> dict:
    verdict = interrupt(
        {
            "customer_id": state.customer_id,
            "risk_tier": state.risk_tier,
            "sar_narrative": state.sar_narrative,
        }
    )
    return {
        "decision": f"HUMAN_{verdict}",
        "audit_log": [
            AuditEntry(node="human_review", summary=f"analyst: {verdict}")
        ],
    }


def route_by_risk(state: CaseState) -> str:
    return {"LOW": "auto_approve", "MEDIUM": "request_info", "HIGH": "sar_draft"}[
        state.risk_tier
    ]


def build_graph():
    g = StateGraph(CaseState)
    for name, fn in [
        ("intake", intake_node),
        ("screening", screening_node),
        ("monitoring", monitoring_node),
        ("risk_scoring", risk_scoring_node),
        ("sar_draft", sar_draft_node),
        ("human_review", human_review_node),
        ("auto_approve", auto_approve_node),
        ("request_info", request_info_node),
    ]:
        g.add_node(name, fn)

    g.add_edge(START, "intake")
    g.add_edge("intake", "screening")
    g.add_edge("screening", "monitoring")
    g.add_edge("monitoring", "risk_scoring")
    g.add_conditional_edges(
        "risk_scoring",
        route_by_risk,
        {
            "auto_approve": "auto_approve",
            "request_info": "request_info",
            "sar_draft": "sar_draft",
        },
    )
    g.add_edge("sar_draft", "human_review")
    for terminal in ["auto_approve", "request_info", "human_review"]:
        g.add_edge(terminal, END)

    return g.compile(checkpointer=MemorySaver())
