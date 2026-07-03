import json

import streamlit as st

from src.data.generate_kyc_docs import doc_to_text
from src.config import KYC_DOCS_PATH
from src.graph import build_graph
from src.state import CaseState

st.set_page_config(page_title="KYC/AML Agent Demo", layout="wide")
st.title("Agentic KYC/AML Compliance System")

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

graph = st.session_state.graph

docs = []
if KYC_DOCS_PATH.exists():
    docs = json.loads(KYC_DOCS_PATH.read_text())

st.sidebar.header("Submit a case")
example_labels = ["Custom document"] + [
    f"{d.get('full_name', 'Unknown')} ({d.get('account_id', '')})" for d in docs[:10]
]
choice = st.sidebar.selectbox("Example customer", example_labels)

if choice == "Custom document":
    document = st.sidebar.text_area("Onboarding document", height=200)
else:
    idx = example_labels.index(choice) - 1
    document = doc_to_text(docs[idx])

customer_id = st.sidebar.text_input("Customer ID", value="case-001")
run = st.sidebar.button("Run screening", type="primary")

col1, col2 = st.columns(2)

if run and document:
    cfg = {"configurable": {"thread_id": customer_id}}
    with st.spinner("Running KYC/AML pipeline..."):
        result = graph.invoke(
            CaseState(customer_id=customer_id, raw_document=document), cfg
        )

    if isinstance(result, dict):
        decision = result.get("decision")
        risk_tier = result.get("risk_tier")
        audit_log = result.get("audit_log", [])
        interrupted = "__interrupt__" in result
    else:
        decision = result.decision
        risk_tier = result.risk_tier
        audit_log = result.audit_log
        interrupted = False

    with col1:
        st.metric("Risk tier", risk_tier or "—")
        st.metric("Decision", decision or "—")

    with col2:
        if interrupted:
            st.warning("Case paused for human review (HIGH risk)")
            verdict = st.selectbox("Analyst verdict", ["APPROVED", "REJECTED", "ESCALATED"])
            if st.button("Submit verdict"):
                from langgraph.types import Command

                final = graph.invoke(Command(resume=verdict), cfg)
                if isinstance(final, dict):
                    st.success(f"Final decision: {final.get('decision')}")
                else:
                    st.success(f"Final decision: {final.decision}")

    st.subheader("Audit trail")
    for entry in audit_log:
        data = entry.model_dump() if hasattr(entry, "model_dump") else entry
        with st.expander(f"{data['node']}: {data['summary']}"):
            st.json(data.get("payload", {}))
else:
    st.info("Select an example or paste a document, then click **Run screening**.")
