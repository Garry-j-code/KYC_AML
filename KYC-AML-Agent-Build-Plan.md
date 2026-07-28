# Agentic KYC/AML Compliance System — Implementation Guide

An implementation-grade spec for building the system module by module. Every section is concrete: exact data sources, folder layout, real module code, and the build order.

> **Note on library versions:** LangChain/LangGraph APIs move fast. The code below is idiomatic and correct in shape. Before implementing a module, verify library signatures against the installed version (`pip show langgraph langchain`) and adjust. Treat the code as a precise spec of *intent*, not a frozen copy-paste.

---

## 0. What you're building (recap)

An agentic KYC onboarding + AML monitoring system on LangGraph. It ingests a customer onboarding document, screens the identity against real sanctions/PEP data, checks their transactions for suspicious patterns, scores risk, and routes to auto-approve / request-info / human-escalation — with a full audit trail on every decision.

The two things that make it hireable: **human-in-the-loop escalation** (agents don't auto-approve in regulated flows) and **hybrid ML+LLM** (trained classifier for high-volume detection, LLM only for reasoning/narrative).

---

## 1. Repo structure

```
kyc-aml-agent/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── .cursorrules                  # build invariants (see §9)
├── data/
│   ├── raw/                       # downloaded datasets (gitignored)
│   ├── generated/                 # Faker-generated KYC docs
│   └── models/                    # trained LightGBM model
├── src/
│   ├── config.py                  # settings, paths, model names
│   ├── state.py                   # CaseState schema
│   ├── data/
│   │   ├── generate_kyc_docs.py   # Faker synthetic onboarding docs
│   │   ├── sanctions.py           # OpenSanctions loader + fuzzy matcher
│   │   └── transactions.py        # IBM AML loader, feature eng, train + inference
│   ├── agents/
│   │   ├── intake.py              # document extraction node
│   │   ├── screening.py           # sanctions/PEP screening node
│   │   ├── monitoring.py          # transaction monitoring node
│   │   ├── risk_scoring.py        # risk tier decision node
│   │   └── sar_draft.py           # SAR narrative node
│   ├── graph.py                   # LangGraph wiring + HITL interrupt
│   ├── api.py                     # FastAPI endpoint
│   └── ui.py                      # Streamlit demo
├── scripts/
│   ├── download_data.sh
│   └── train_model.py
├── tests/
│   ├── test_extraction.py
│   ├── test_matcher.py
│   └── test_graph.py
└── notebooks/
    └── walkthrough.ipynb          # the demo you record
```

---

## 2. Environment & dependencies

**requirements.txt**
```
langgraph
langchain
langchain-anthropic
langchain-community
pydantic>=2
rapidfuzz
lightgbm
scikit-learn
pandas
numpy
faker
fastapi
uvicorn
streamlit
python-dotenv
kaggle
pytest
```

**.env.example**
```
ANTHROPIC_API_KEY=your_key_here
# swap for OPENAI_API_KEY if you prefer; or run Ollama locally for dev
MODEL_STRONG=claude-sonnet-4-6      # orchestration reasoning + SAR narrative
MODEL_CHEAP=claude-haiku-4-5        # high-volume extraction
```

Keep `MODEL_CHEAP` on the extraction node (runs on every doc) and `MODEL_STRONG` on risk reasoning and SAR narrative only. That's the cost discipline you'll talk about in interviews.

---

## 3. Data acquisition — exact sources

### 3a. Sanctions / PEP data — OpenSanctions (free, no login)

Bulk download the default collection as simplified CSV:

```bash
# scripts/download_data.sh (part 1)
mkdir -p data/raw
curl -L "https://data.opensanctions.org/datasets/latest/default/targets.simple.csv" \
  -o data/raw/opensanctions_targets.csv
```

Typical columns in `targets.simple.csv` (inspect the header first — schema evolves): `id, schema, name, aliases, birth_date, countries, addresses, identifiers, sanctions, phones, emails, dataset, first_seen, last_seen, last_change`. The `schema` column tells you entity type (Person, Company, etc.); `aliases` is a semicolon-ish delimited list you'll match against too.

For dev, filter to a few thousand rows (e.g. Persons only) so matching is fast. Upgrade path (stretch): self-host the **yente** matching API via Docker for production-grade fuzzy matching instead of local `rapidfuzz`.

### 3b. Transaction data — IBM synthetic AML (Kaggle, free account)

```bash
# scripts/download_data.sh (part 2)
# requires ~/.kaggle/kaggle.json credentials (free Kaggle account)
kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml \
  -p data/raw --unzip
```

Use **`HI-Small_Trans.csv`** for development (HI = higher illicit ratio, Small = manageable size). Columns:
`Timestamp, From Bank, Account, To Bank, Account.1, Amount Received, Receiving Currency, Amount Paid, Payment Currency, Payment Format, Is Laundering`

`Is Laundering` (0/1) is your label. It's per-transaction and heavily imbalanced (~0.1%), which you'll handle with class weighting.

### 3c. KYC documents — generated with Faker (you create these)

No real KYC docs are public (PII). Generate ~200 synthetic onboarding docs, inject messiness into ~20%. Code in §5a.

---

## 4. Core: state schema

**src/state.py**
```python
import operator
from typing import Annotated, Literal, Optional
from pydantic import BaseModel, Field


class ExtractedIdentity(BaseModel):
    full_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    nationality: Optional[str] = None
    address: Optional[str] = None
    id_number: Optional[str] = None
    account_id: Optional[str] = None   # links to transaction data


class ScreeningHit(BaseModel):
    matched_name: str
    match_score: float
    entity_type: str          # "sanction" | "pep" | "rca"
    source: str               # OFAC / EU / UN / ...
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

    # accumulates across every node — this is your explainability story
    audit_log: Annotated[list[AuditEntry], operator.add] = Field(default_factory=list)
```

The `Annotated[..., operator.add]` reducers let each node *append* to `audit_log` and `screening_hits` by returning just the new items, instead of rebuilding the whole list. Every node returns an `AuditEntry` — this is a hard invariant (see `.cursorrules` in §9).

---

## 5. Data modules

### 5a. src/data/generate_kyc_docs.py

```python
import json, random
from pathlib import Path
from faker import Faker

fake = Faker()
OUT = Path("data/generated")
OUT.mkdir(parents=True, exist_ok=True)

def make_doc(account_id: str, messy: bool = False) -> dict:
    doc = {
        "full_name": fake.name(),
        "date_of_birth": fake.date_of_birth(minimum_age=18, maximum_age=90).isoformat(),
        "nationality": fake.country(),
        "address": fake.address().replace("\n", ", "),
        "id_number": fake.bothify("??######"),
        "account_id": account_id,
    }
    if messy:
        # simulate real-world mess: drop a field, or corrupt date format
        choice = random.choice(["drop", "date_fmt", "blank_name"])
        if choice == "drop":
            doc.pop(random.choice(["nationality", "id_number", "address"]))
        elif choice == "date_fmt":
            doc["date_of_birth"] = fake.date(pattern="%m-%d-%y")  # inconsistent fmt
        elif choice == "blank_name":
            doc["full_name"] = ""
    return doc

def main(n: int = 200, messy_ratio: float = 0.2):
    docs = []
    for i in range(n):
        messy = random.random() < messy_ratio
        docs.append(make_doc(account_id=f"ACC{i:05d}", messy=messy))
    (OUT / "kyc_docs.json").write_text(json.dumps(docs, indent=2))
    print(f"wrote {n} docs ({int(n*messy_ratio)} messy) -> {OUT/'kyc_docs.json'}")

if __name__ == "__main__":
    main()
```

**Optional realism upgrade:** render each dict into free-text with an LLM ("write this as a scanned onboarding form with slightly inconsistent formatting") so the extraction agent parses prose, not clean JSON. Do this only after the happy path works.

To create test cases that actually hit the sanctions matcher, seed a handful of docs with real names pulled from `opensanctions_targets.csv`.

### 5b. src/data/sanctions.py

```python
import pandas as pd
from functools import lru_cache
from rapidfuzz import process, fuzz
from src.state import ScreeningHit

CSV = "data/raw/opensanctions_targets.csv"

@lru_cache(maxsize=1)
def _load() -> pd.DataFrame:
    df = pd.read_csv(CSV, low_memory=False)
    df = df[df["schema"] == "Person"].copy()      # persons for KYC
    df["name"] = df["name"].fillna("").str.strip()
    df = df[df["name"] != ""]
    return df.reset_index(drop=True)

def screen(name: str, threshold: int = 88, limit: int = 3) -> list[ScreeningHit]:
    if not name:
        return []
    df = _load()
    names = df["name"].tolist()
    matches = process.extract(name, names, scorer=fuzz.WRatio, limit=limit)
    hits = []
    for matched_name, score, idx in matches:
        if score < threshold:
            continue
        row = df.iloc[idx]
        topics = str(row.get("sanctions", "")) + str(row.get("dataset", ""))
        entity_type = "pep" if "pep" in topics.lower() else "sanction"
        hits.append(ScreeningHit(
            matched_name=matched_name,
            match_score=float(score),
            entity_type=entity_type,
            source=str(row.get("dataset", "opensanctions")),
            details={"id": str(row.get("id", "")),
                     "countries": str(row.get("countries", ""))},
        ))
    return hits
```

`WRatio` handles token reordering and partial names well (e.g. "Vladimir Putin" vs "Putin, Vladimir Vladimirovich"). Tune `threshold` on your seeded test cases — too low floods false positives, which is itself a realistic AML problem you can mention.

### 5c. src/data/transactions.py

```python
import pandas as pd, lightgbm as lgb, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW = "data/raw/HI-Small_Trans.csv"
MODEL = Path("data/models/lgbm_aml.pkl")

def _features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ts"] = pd.to_datetime(df["Timestamp"])
    df["hour"] = df["ts"].dt.hour
    df["cross_bank"] = (df["From Bank"] != df["To Bank"]).astype(int)
    df["currency_mismatch"] = (df["Receiving Currency"] != df["Payment Currency"]).astype(int)
    df["amount_paid"] = pd.to_numeric(df["Amount Paid"], errors="coerce").fillna(0)
    df["is_round"] = (df["amount_paid"] % 1000 == 0).astype(int)
    df["pay_format"] = df["Payment Format"].astype("category").cat.codes
    return df

FEATS = ["hour", "cross_bank", "currency_mismatch", "amount_paid", "is_round", "pay_format"]

def train():
    df = _features(pd.read_csv(RAW))
    X, y = df[FEATS], df["Is Laundering"].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pos_weight = (ytr == 0).sum() / max((ytr == 1).sum(), 1)   # handle imbalance
    clf = lgb.LGBMClassifier(scale_pos_weight=pos_weight, n_estimators=300)
    clf.fit(Xtr, ytr)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL)
    print("AUC-ish sanity — predict on holdout and eyeball recall on the positive class")
    return clf

def score_account(account_id: str, top_k: int = 5) -> tuple[float, list[dict]]:
    """Return an account-level risk score + its most suspicious transactions."""
    clf = joblib.load(MODEL)
    df = _features(pd.read_csv(RAW))
    acct = df[(df["Account"] == account_id) | (df["Account.1"] == account_id)]
    if acct.empty:
        return 0.0, []
    proba = clf.predict_proba(acct[FEATS])[:, 1]
    acct = acct.assign(risk=proba).sort_values("risk", ascending=False)
    flagged = acct.head(top_k)[["Timestamp", "amount_paid", "risk"]].to_dict("records")
    return float(proba.max()), flagged
```

Detection is a trained classifier, not an LLM call — that's deliberate and defensible on cost/latency. The LLM's job (in the monitoring node) is to *explain* why the flagged transactions look like laundering, not to score them.

---

## 6. Agent nodes

Each node is a function `(state: CaseState) -> dict` returning only the fields it updates (plus one `AuditEntry`).

### 6a. src/agents/intake.py

```python
from langchain_anthropic import ChatAnthropic
from src.config import MODEL_CHEAP
from src.state import CaseState, ExtractedIdentity, AuditEntry

llm = ChatAnthropic(model=MODEL_CHEAP, temperature=0).with_structured_output(ExtractedIdentity)

def intake_node(state: CaseState) -> dict:
    identity = llm.invoke(
        f"Extract the identity fields from this onboarding document. "
        f"Return null for any field not present.\n\n{state.raw_document}"
    )
    issues = [f for f in ["full_name", "date_of_birth", "nationality", "id_number"]
              if not getattr(identity, f)]
    return {
        "extracted_identity": identity,
        "completeness_issues": issues,
        "audit_log": [AuditEntry(node="intake",
                                 summary=f"extracted identity; {len(issues)} missing fields",
                                 payload={"issues": issues})],
    }
```

### 6b. src/agents/screening.py

```python
from src.data.sanctions import screen
from src.state import CaseState, AuditEntry

def screening_node(state: CaseState) -> dict:
    name = state.extracted_identity.full_name if state.extracted_identity else ""
    hits = screen(name)
    return {
        "screening_hits": hits,
        "audit_log": [AuditEntry(node="screening",
                                 summary=f"{len(hits)} sanctions/PEP hits for '{name}'",
                                 payload={"top": hits[0].model_dump() if hits else None})],
    }
```

### 6c. src/agents/monitoring.py

```python
from langchain_anthropic import ChatAnthropic
from src.config import MODEL_STRONG
from src.data.transactions import score_account
from src.state import CaseState, AuditEntry

llm = ChatAnthropic(model=MODEL_STRONG, temperature=0)

def monitoring_node(state: CaseState) -> dict:
    acct = state.extracted_identity.account_id if state.extracted_identity else None
    score, flagged = score_account(acct) if acct else (0.0, [])
    reasoning = ""
    if flagged:
        reasoning = llm.invoke(
            f"These transactions were flagged as high-risk by an AML model. "
            f"In 2-3 sentences, explain what laundering typology they might indicate "
            f"(structuring, layering, rapid movement, etc.):\n{flagged}"
        ).content
    return {
        "transaction_risk_score": score,
        "flagged_transactions": flagged,
        "audit_log": [AuditEntry(node="monitoring",
                                 summary=f"txn risk {score:.2f}, {len(flagged)} flagged",
                                 payload={"reasoning": reasoning})],
    }
```

### 6d. src/agents/risk_scoring.py

```python
from src.state import CaseState, AuditEntry

def risk_scoring_node(state: CaseState) -> dict:
    """Deterministic, explainable aggregation — NOT a black box."""
    tier = "LOW"
    reasons = []
    if any(h.entity_type == "sanction" and h.match_score >= 92 for h in state.screening_hits):
        tier = "HIGH"; reasons.append("strong sanctions match")
    elif any(h.entity_type == "pep" for h in state.screening_hits):
        tier = "HIGH"; reasons.append("PEP match")
    elif (state.transaction_risk_score or 0) >= 0.8:
        tier = "HIGH"; reasons.append("high transaction risk")
    elif state.completeness_issues:
        tier = "MEDIUM"; reasons.append("incomplete KYC")
    elif (state.transaction_risk_score or 0) >= 0.4:
        tier = "MEDIUM"; reasons.append("moderate transaction risk")

    return {
        "risk_tier": tier,
        "audit_log": [AuditEntry(node="risk_scoring",
                                 summary=f"tier={tier}",
                                 payload={"reasons": reasons})],
    }
```

Keep this rules-based and transparent. "Why was this customer flagged HIGH?" must have a one-line answer. A black-box risk score is a red flag in compliance, and interviewers know it.

### 6e. src/agents/sar_draft.py

```python
from langchain_anthropic import ChatAnthropic
from src.config import MODEL_STRONG
from src.state import CaseState, AuditEntry

llm = ChatAnthropic(model=MODEL_STRONG, temperature=0.2)

def sar_draft_node(state: CaseState) -> dict:
    narrative = llm.invoke(
        f"Draft a concise Suspicious Activity Report narrative for a compliance analyst. "
        f"Customer: {state.extracted_identity}. "
        f"Screening hits: {[h.model_dump() for h in state.screening_hits]}. "
        f"Flagged transactions: {state.flagged_transactions}. "
        f"State the facts and the reason for escalation. Do not fabricate details."
    ).content
    return {
        "sar_narrative": narrative,
        "decision": "ESCALATED_TO_HUMAN",
        "audit_log": [AuditEntry(node="sar_draft", summary="SAR narrative drafted")],
    }
```

---

## 7. Graph wiring + human-in-the-loop

**src/graph.py**
```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from src.state import CaseState, AuditEntry
from src.agents.intake import intake_node
from src.agents.screening import screening_node
from src.agents.monitoring import monitoring_node
from src.agents.risk_scoring import risk_scoring_node
from src.agents.sar_draft import sar_draft_node


def auto_approve_node(state: CaseState) -> dict:
    return {"decision": "AUTO_APPROVED",
            "audit_log": [AuditEntry(node="auto_approve", summary="low risk, approved")]}

def request_info_node(state: CaseState) -> dict:
    return {"decision": "INFO_REQUESTED",
            "audit_log": [AuditEntry(node="request_info", summary="medium risk, more info needed")]}

def human_review_node(state: CaseState) -> dict:
    # pauses the graph; resume with the analyst's decision
    verdict = interrupt({"customer_id": state.customer_id,
                         "risk_tier": state.risk_tier,
                         "sar_narrative": state.sar_narrative})
    return {"decision": f"HUMAN_{verdict}",
            "audit_log": [AuditEntry(node="human_review", summary=f"analyst: {verdict}")]}


def route_by_risk(state: CaseState) -> str:
    return {"LOW": "auto_approve", "MEDIUM": "request_info", "HIGH": "sar_draft"}[state.risk_tier]


def build_graph():
    g = StateGraph(CaseState)
    for name, fn in [("intake", intake_node), ("screening", screening_node),
                     ("monitoring", monitoring_node), ("risk_scoring", risk_scoring_node),
                     ("sar_draft", sar_draft_node), ("human_review", human_review_node),
                     ("auto_approve", auto_approve_node), ("request_info", request_info_node)]:
        g.add_node(name, fn)

    g.add_edge(START, "intake")
    g.add_edge("intake", "screening")
    g.add_edge("screening", "monitoring")
    g.add_edge("monitoring", "risk_scoring")
    g.add_conditional_edges("risk_scoring", route_by_risk,
                            {"auto_approve": "auto_approve",
                             "request_info": "request_info",
                             "sar_draft": "sar_draft"})
    g.add_edge("sar_draft", "human_review")
    for terminal in ["auto_approve", "request_info", "human_review"]:
        g.add_edge(terminal, END)

    return g.compile(checkpointer=MemorySaver())   # checkpointer required for interrupt()
```

**Running it (with resume for the human step):**
```python
from langgraph.types import Command
from src.graph import build_graph
from src.state import CaseState

graph = build_graph()
config = {"configurable": {"thread_id": "case-001"}}

result = graph.invoke(CaseState(customer_id="C1", raw_document=doc_text), config)
# if it paused on human_review, result will contain an __interrupt__ payload:
if "__interrupt__" in result:
    final = graph.invoke(Command(resume="APPROVED"), config)   # analyst decision
```

---

## 8. Demo layer

### 8a. src/api.py (FastAPI)
```python
from fastapi import FastAPI
from pydantic import BaseModel
from src.graph import build_graph
from src.state import CaseState

app = FastAPI()
graph = build_graph()

class SubmitReq(BaseModel):
    customer_id: str
    document: str

@app.post("/screen")
def screen_customer(req: SubmitReq):
    cfg = {"configurable": {"thread_id": req.customer_id}}
    out = graph.invoke(CaseState(customer_id=req.customer_id, raw_document=req.document), cfg)
    return {"decision": out.get("decision"),
            "risk_tier": out.get("risk_tier"),
            "audit_log": [a.model_dump() for a in out.get("audit_log", [])]}
```

### 8b. src/ui.py (Streamlit)
A form to submit a customer (or pick a canned example), a live view of the decision + risk tier, and an expandable audit trail. For the *deployed* version, ship canned examples that replay stored runs so visitors don't burn your API budget (see §11).

---

## 9. Build order & invariants

Build one module at a time, each with a test, committing between steps. Don't build the whole system in one pass — it sprawls and makes failures hard to isolate.

### Invariants (`.cursorrules`, placed at repo root)

These hold throughout the entire build:
```
- This is a LangGraph multi-agent KYC/AML system. Read src/state.py before touching any node.
- Every graph node returns a dict of ONLY the fields it updates, and always appends one
  AuditEntry to audit_log. Never rebuild the whole state.
- Detection uses the trained LightGBM model, never an LLM. LLMs are only for extraction,
  reasoning, and narrative.
- Risk scoring must stay deterministic and explainable — no LLM, no hidden logic.
- Verify LangChain/LangGraph API signatures against the installed version before writing code.
- Write a pytest test for every data module and agent node.
- Use MODEL_CHEAP for extraction, MODEL_STRONG for reasoning/SAR only.
```

### Build order

1. Scaffold the repo per §1; create empty modules and `requirements.txt`.
2. Implement `src/state.py` per §4; add a test that constructs a CaseState and appends to audit_log.
3. Implement `src/data/generate_kyc_docs.py`; run it and confirm 3 sample docs, including one messy one.
4. Implement `src/data/sanctions.py`; add a test where a seeded real sanctioned name scores above threshold and a random name does not.
5. Implement `src/data/transactions.py`; train the model and report positive-class recall on a holdout (not accuracy — see §5c).
6. Implement each agent node in `src/agents/`; unit-test each on a hand-built CaseState.
7. Wire `src/graph.py`; run the three scenarios (clean approve, PEP escalation, suspicious-txn escalation) and verify their audit logs.
8. Add `src/api.py` and `src/ui.py`.
9. Write the README with architecture diagram and demo GIF.

Run each step's test and commit before starting the next. The commits are your rollback points.

---

## 10. Phased timeline (~2 weeks, evenings/weekends)

- **Day 1** — Phase 0: scaffold + download data + empty graph runs
- **Days 2–3** — Phase 1: three data modules working & tested
- **Days 4–7** — Phase 2: all six nodes working in isolation
- **Days 8–9** — Phase 3: graph wired, HITL working, three scenarios pass
- **Days 10–11** — Phase 4: API + Streamlit demo
- **Day 12** — Phase 5: README, demo video, deploy

**Scope control:** MVP = 6 nodes + graph + HITL + audit log + one demo path. Cut first if behind: Streamlit UI, the request-info branch, deployment. Do **not** start a stretch goal (yente, GNN, adverse-media agent) until the MVP demos end-to-end.

---

## 11. Deployment & demo

Priority order (all free):

1. **Recorded demo video (non-negotiable, ~2–3 min).** Loom/screen recording linked at the *top* of the README, above install instructions. You control the narrative: clean approve → PEP escalation → audit trail. Many strong portfolios stop here.
2. **Streamlit Community Cloud (free, if you have a spare afternoon).** Ship with canned example customers or pre-computed runs the app *replays*, so visitors see the graph decide and the audit log populate without firing live LLM calls on your budget.
3. **Skip** containerized cloud deploy unless targeting infra/MLOps roles — your prior AWS EC2 work already covers "can deploy" on the resume.

Put the clickable link/video above the fold in the README. Hiding the demo three scrolls down is the most common own-goal in portfolio repos.

---

## 12. Interview talking points (prep these)

1. *"High-risk cases hit a human-in-the-loop interrupt — in a regulated flow the agent triages, it doesn't auto-approve. That's the actual deployment blocker in finance, not model quality."*
2. *"Detection is a trained classifier; the LLM only reasons and drafts narratives. LLMs are the wrong tool for high-volume scoring on cost and latency."*
3. *"Every node appends to an audit log, so any decision is fully traceable — which is what regulators and FINRA's 2026 agent-supervision stance actually require."*
4. *Honest limitations:* real data would need PII handling, real-time streaming, model monitoring, and drift detection. Naming the gaps reads as senior.