"""End-to-end smoke test: runs the full LangGraph on three scenarios with real
data + real Groq calls, and prints the decision, risk tier, and audit trail.

Scenarios:
  A. Clean customer            -> LOW    -> AUTO_APPROVED
  B. Sanctions match           -> HIGH   -> SAR draft -> human review (resume)
  C. Suspicious transactions   -> HIGH   -> SAR draft -> human review (resume)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.types import Command  # noqa: E402

from src.data.generate_kyc_docs import doc_to_text  # noqa: E402
from src.graph import build_graph  # noqa: E402
from src.state import CaseState  # noqa: E402

# Real IDs discovered from the datasets:
BENIGN_ACCT = "8000EBD30"  # low model risk
LAUNDER_ACCT = "100428660"  # tied to Is Laundering=1 rows
SANCTIONED_NAME = "SANAVBARI NIKITENKO"  # real OpenSanctions person

SCENARIOS = [
    (
        "A. clean approve",
        {
            "full_name": "Gregory Fairbanks",
            "date_of_birth": "1985-04-12",
            "nationality": "United States",
            "address": "42 Maple Street, Springfield",
            "id_number": "US123456",
            "account_id": BENIGN_ACCT,
        },
    ),
    (
        "B. sanctions escalation",
        {
            "full_name": SANCTIONED_NAME,
            "date_of_birth": "1970-02-02",
            "nationality": "Russia",
            "address": "12 Nevsky Prospekt, St Petersburg",
            "id_number": "RU998877",
            "account_id": BENIGN_ACCT,
        },
    ),
    (
        "C. suspicious-transaction escalation",
        {
            "full_name": "Marcus Wellington",
            "date_of_birth": "1979-09-30",
            "nationality": "United Kingdom",
            "address": "8 Baker Street, London",
            "id_number": "UK445566",
            "account_id": LAUNDER_ACCT,
        },
    ),
]


def run():
    graph = build_graph()
    for i, (label, doc) in enumerate(SCENARIOS):
        print("\n" + "=" * 70)
        print(label)
        print("=" * 70)
        cfg = {"configurable": {"thread_id": f"smoke-{i}"}}
        result = graph.invoke(
            CaseState(customer_id=f"C{i}", raw_document=doc_to_text(doc)), cfg
        )
        if "__interrupt__" in result:
            print("  [paused at human_review interrupt -> analyst resumes: APPROVED]")
            result = graph.invoke(Command(resume="APPROVED"), cfg)

        print(f"  decision:   {result.get('decision')}")
        print(f"  risk_tier:  {result.get('risk_tier')}")
        ident = result.get("extracted_identity")
        print(f"  extracted:  name={getattr(ident, 'full_name', None)!r} "
              f"account={getattr(ident, 'account_id', None)!r}")
        print(f"  screening:  {len(result.get('screening_hits', []))} hit(s)")
        print(f"  txn_risk:   {result.get('transaction_risk_score')}")
        print("  audit trail:")
        for entry in result.get("audit_log", []):
            print(f"    - {entry.node}: {entry.summary}")


if __name__ == "__main__":
    run()
