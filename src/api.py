from fastapi import FastAPI
from pydantic import BaseModel

from src.graph import build_graph
from src.state import CaseState

app = FastAPI(title="KYC/AML Agent")
graph = build_graph()


class SubmitReq(BaseModel):
    customer_id: str
    document: str


class ResumeReq(BaseModel):
    customer_id: str
    verdict: str


@app.post("/screen")
def screen_customer(req: SubmitReq):
    cfg = {"configurable": {"thread_id": req.customer_id}}
    out = graph.invoke(
        CaseState(customer_id=req.customer_id, raw_document=req.document), cfg
    )
    if isinstance(out, dict):
        audit = out.get("audit_log", [])
        return {
            "decision": out.get("decision"),
            "risk_tier": out.get("risk_tier"),
            "audit_log": [a.model_dump() if hasattr(a, "model_dump") else a for a in audit],
            "interrupted": "__interrupt__" in out,
        }
    audit = out.audit_log if hasattr(out, "audit_log") else []
    return {
        "decision": out.decision,
        "risk_tier": out.risk_tier,
        "audit_log": [a.model_dump() for a in audit],
        "interrupted": False,
    }


@app.post("/resume")
def resume_case(req: ResumeReq):
    from langgraph.types import Command

    cfg = {"configurable": {"thread_id": req.customer_id}}
    out = graph.invoke(Command(resume=req.verdict), cfg)
    if isinstance(out, dict):
        audit = out.get("audit_log", [])
        return {
            "decision": out.get("decision"),
            "risk_tier": out.get("risk_tier"),
            "audit_log": [a.model_dump() if hasattr(a, "model_dump") else a for a in audit],
        }
    return {
        "decision": out.decision,
        "risk_tier": out.risk_tier,
        "audit_log": [a.model_dump() for a in out.audit_log],
    }
