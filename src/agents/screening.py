from src.data.sanctions import screen
from src.state import AuditEntry, CaseState


def screening_node(state: CaseState) -> dict:
    name = state.extracted_identity.full_name if state.extracted_identity else ""
    hits = screen(name)
    return {
        "screening_hits": hits,
        "audit_log": [
            AuditEntry(
                node="screening",
                summary=f"{len(hits)} sanctions/PEP hits for '{name}'",
                payload={"top": hits[0].model_dump() if hits else None},
            )
        ],
    }
