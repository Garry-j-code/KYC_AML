from src.config import MODEL_STRONG
from src.data.transactions import score_account
from src.llm import get_chat_model
from src.state import AuditEntry, CaseState

llm = get_chat_model(MODEL_STRONG, temperature=0)


def monitoring_node(state: CaseState) -> dict:
    acct = state.extracted_identity.account_id if state.extracted_identity else None
    score, flagged = score_account(acct) if acct else (0.0, [])
    reasoning = ""
    if flagged:
        reasoning = llm.invoke(
            "These transactions were flagged as high-risk by an AML model. "
            "In 2-3 sentences, explain what laundering typology they might indicate "
            f"(structuring, layering, rapid movement, etc.):\n{flagged}"
        ).content
    return {
        "transaction_risk_score": score,
        "flagged_transactions": flagged,
        "audit_log": [
            AuditEntry(
                node="monitoring",
                summary=f"txn risk {score:.2f}, {len(flagged)} flagged",
                payload={"reasoning": reasoning},
            )
        ],
    }
