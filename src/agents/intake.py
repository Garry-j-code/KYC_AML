from langchain_anthropic import ChatAnthropic

from src.config import MODEL_CHEAP
from src.state import AuditEntry, CaseState, ExtractedIdentity

llm = ChatAnthropic(model=MODEL_CHEAP, temperature=0).with_structured_output(ExtractedIdentity)


def intake_node(state: CaseState) -> dict:
    identity = llm.invoke(
        "Extract the identity fields from this onboarding document. "
        "Return null for any field not present.\n\n"
        f"{state.raw_document}"
    )
    issues = [
        f
        for f in ["full_name", "date_of_birth", "nationality", "id_number"]
        if not getattr(identity, f)
    ]
    return {
        "extracted_identity": identity,
        "completeness_issues": issues,
        "audit_log": [
            AuditEntry(
                node="intake",
                summary=f"extracted identity; {len(issues)} missing fields",
                payload={"issues": issues},
            )
        ],
    }
