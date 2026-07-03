import operator
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field


class ExtractedIdentity(BaseModel):
    full_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    nationality: Optional[str] = None
    address: Optional[str] = None
    id_number: Optional[str] = None
    account_id: Optional[str] = None


class ScreeningHit(BaseModel):
    matched_name: str
    match_score: float
    entity_type: str
    source: str
    details: dict = Field(default_factory=dict)


class AuditEntry(BaseModel):
    node: str
    summary: str
    payload: dict = Field(default_factory=dict)


class CaseState(BaseModel):
    customer_id: str
    raw_document: str = ""
    extracted_identity: Optional[ExtractedIdentity] = None
    completeness_issues: list[str] = Field(default_factory=list)

    screening_hits: Annotated[list[ScreeningHit], operator.add] = Field(default_factory=list)
    flagged_transactions: list[dict] = Field(default_factory=list)
    transaction_risk_score: Optional[float] = None

    risk_tier: Optional[Literal["LOW", "MEDIUM", "HIGH"]] = None
    decision: Optional[str] = None
    sar_narrative: Optional[str] = None

    audit_log: Annotated[list[AuditEntry], operator.add] = Field(default_factory=list)
