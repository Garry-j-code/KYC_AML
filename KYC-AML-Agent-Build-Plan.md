# Agentic KYC/AML Compliance System — Build Plan

**Goal:** One flagship portfolio project that proves *both* production-grade agentic system design *and* fintech domain fluency — the combination that separates you from candidates who've only built generic RAG chatbots.

**Timeline:** ~2 weeks of evenings/weekends. Scope is deliberately cut to ship, not to be exhaustive.

**One-line pitch (for your resume/LinkedIn):** *An agentic KYC onboarding and AML monitoring system built on LangGraph that screens customers against real sanctions/PEP data, detects suspicious transaction patterns, and routes cases through automated approval or human escalation with a full audit trail.*

---

## 1. Why this project stands out

Most portfolio projects are "answer" systems (retrieve → summarize). This is an "act" system: it ingests messy data, reasons across structured + unstructured inputs, makes a decision, and either acts or escalates. That maps directly onto what fintech firms are actually deploying in 2026, and onto four things interviewers probe for:

1. **Human-in-the-loop governance** — low-risk cases auto-approve, high-risk cases escalate to a review queue *with context attached*. This is the single biggest signal that you understand fintech AI can't just auto-approve.
2. **Auditability / explainability** — every decision carries a reasoning trail. Regulators (and FINRA's 2026 stance treating autonomous agents as a distinct supervisory category) care about this.
3. **A genuinely hard technical core** — fuzzy entity matching against sanctions lists across name variants is a real problem, not a toy.
4. **Hybrid ML + LLM** — a trained classifier does the high-volume detection; the LLM does reasoning and narrative. Shows you know when *not* to use an LLM.

---

## 2. The domain workflow you're modeling

Real banks run this pipeline. Your agents mirror it 1:1, which is what makes the project legible to a fintech hiring manager:

| Real bank stage | Your agent |
|---|---|
| Customer Identification Program (CIP) — collect & verify ID docs | Document Intake & Extraction agent |
| Sanctions / PEP / watchlist screening | Screening agent |
| Customer Due Diligence (CDD) risk rating | Risk Scoring agent |
| Ongoing transaction monitoring | Transaction Monitoring agent |
| Alert → investigation → SAR filing | Case Narrative / SAR-draft agent |
| Analyst review of escalated cases | Human-in-the-loop interrupt |

---

## 3. Architecture

### Agents (LangGraph nodes)

1. **Document Intake & Extraction** — ingests a synthetic onboarding doc (Faker-generated), extracts structured identity fields into a Pydantic schema, validates completeness, flags missing/malformed data. Deliberately handle messy inputs (missing fields, inconsistent date formats) — that's the realistic part.
2. **Sanctions & PEP Screening** — takes the extracted identity, runs fuzzy name matching against OpenSanctions data, returns hits with match scores and the matched entity's details.
3. **Transaction Monitoring** — runs the customer's transaction history through a trained classifier (LightGBM), surfaces flagged transactions, and has the LLM reason over *why* a pattern is suspicious (structuring, layering, rapid movement).
4. **Risk Scoring / Decision** — aggregates all signals (doc completeness, sanctions/PEP hits, transaction risk) into a risk tier: LOW / MEDIUM / HIGH.
5. **Case Narrative / SAR Draft** — for escalated cases, drafts a suspicious-activity narrative with the reasoning and evidence attached. This is your "credit memo" equivalent — high-value, shows you know what a compliance analyst actually needs to read.
6. **Orchestrator / Router** — conditional routing: LOW → auto-approve, MEDIUM → request more info, HIGH → human review queue.

### Graph shape

```
                    ┌─────────────────┐
   new customer ──> │ Document Intake │
                    │  & Extraction   │
                    └────────┬────────┘
                             │ (incomplete? → request info)
                             v
                    ┌─────────────────┐
                    │ Sanctions/PEP   │
                    │   Screening     │
                    └────────┬────────┘
                             v
                    ┌─────────────────┐
                    │  Transaction    │
                    │   Monitoring    │
                    └────────┬────────┘
                             v
                    ┌─────────────────┐
                    │  Risk Scoring   │
                    │   / Decision    │
                    └────────┬────────┘
              ┌──────────────┼──────────────┐
         LOW  v         MEDIUM v        HIGH v
     ┌────────────┐  ┌────────────┐  ┌──────────────┐
     │Auto-approve│  │Request more│  │ SAR Draft +  │
     │  + log     │  │   info     │  │ Human review │  ← interrupt()
     └────────────┘  └────────────┘  └──────────────┘
```

### State schema (the object flowing through the graph)

```python
from pydantic import BaseModel
from typing import Literal, Optional

class ScreeningHit(BaseModel):
    matched_name: str
    match_score: float
    entity_type: str          # sanction / PEP / rca
    source: str               # OFAC, EU, UN, etc.
    details: dict

class CaseState(BaseModel):
    customer_id: str
    raw_document: str
    extracted_identity: Optional[dict] = None
    completeness_issues: list[str] = []
    screening_hits: list[ScreeningHit] = []
    flagged_transactions: list[dict] = []
    transaction_risk_score: Optional[float] = None
    risk_tier: Optional[Literal["LOW", "MEDIUM", "HIGH"]] = None
    decision: Optional[str] = None
    sar_narrative: Optional[str] = None
    audit_log: list[dict] = []   # append at every node — this is your explainability
```

The `audit_log` is not decoration — append an entry (node name, inputs seen, output, timestamp) at every node. It *is* your explainability story in the interview.

---

## 4. Tech stack

- **Orchestration:** LangGraph (you already know it — lean into `interrupt()` for human-in-the-loop, and conditional edges for routing)
- **LLM wrappers / structured extraction:** LangChain + Pydantic (`with_structured_output`)
- **Sanctions matching:** OpenSanctions — either self-host the **yente** matching API via Docker (nicer, real fuzzy matching), or bulk-download the CSV and match locally with `rapidfuzz`. Start with `rapidfuzz`, upgrade to yente if time allows.
- **Transaction ML:** LightGBM on the IBM synthetic AML dataset (GNN is a stretch goal — see §7)
- **Synthetic KYC docs:** `Faker` for fields, optionally an LLM pass to make them read like real messy documents
- **Storage / audit:** SQLite (cases + audit log)
- **Demo layer:** FastAPI backend + a small Streamlit UI (submit a customer, watch the graph decide, see the audit trail)
- **LLM:** develop against a cheap/small model or local Ollama to keep costs ~$0; swap to a stronger model for the final demo. Reserve the strong model for the orchestrator reasoning and SAR narrative; use the cheap one for extraction.

---

## 5. Data setup

All free (portfolio = non-commercial). See earlier notes for links.

- **Sanctions/PEP:** `https://data.opensanctions.org/datasets/latest/default/` — start with `targets.simple.csv`. Filter to a manageable subset for dev.
- **Transactions:** IBM synthetic AML dataset on Kaggle — use the **small HI split** for development. Don't touch the large splits until everything works.
- **KYC docs:** generate ~200 with Faker. Inject missing/malformed fields into ~20% of them so the extraction agent has something real to validate.

Reminder: use the *small* versions of everything during the build. Data volume is a time sink that adds nothing to the story. Depth on agent orchestration beats volume on data.

---

## 6. Phased build plan

### Phase 0 — Setup (Day 1)
- Repo, virtualenv, dependency pins, `.env` for keys
- Download all three datasets, confirm they load
- Sketch the state schema and the graph skeleton (nodes as stubs that just pass state through)
- **Milestone:** empty graph runs end-to-end with dummy data

### Phase 1 — Data layer (Days 2–3)
- Faker doc generator with controlled "messiness"
- OpenSanctions loader + `rapidfuzz` matcher; sanity-check a known name (e.g. a real SDN entry) returns a hit
- Load IBM data, engineer a handful of features, train a LightGBM baseline, save the model
- **Milestone:** each data source is queryable via a clean function

### Phase 2 — Individual agents (Days 4–7)
- Extraction agent (structured output + completeness check)
- Screening agent (wraps the matcher, returns `ScreeningHit`s)
- Transaction monitoring agent (classifier + LLM reasoning over flags)
- Risk scoring agent (deterministic rules combining the signals — keep this explainable, not a black box)
- SAR narrative agent
- Test each **in isolation** before wiring them
- **Milestone:** every node works on a hand-crafted input

### Phase 3 — Orchestration + HITL (Days 8–9)
- Wire the conditional edges and routing logic
- Implement the `interrupt()` human-review escalation
- Populate the audit log at every node
- Run 3 scripted end-to-end scenarios: a clean approve, a PEP-match escalation, a suspicious-transaction escalation
- **Milestone:** all three scenarios flow correctly and produce an audit trail

### Phase 4 — Demo layer + polish (Days 10–11)
- FastAPI endpoint: submit customer → return decision + trail
- Streamlit UI: form → live decision view → expandable audit log
- **Milestone:** you can demo it live to a stranger in 3 minutes

### Phase 5 — Portfolio writeup (Day 12)
- README with architecture diagram, the "why it matters" framing, and a GIF/short video of the demo
- Deploy if feasible (Streamlit Community Cloud is free), or a recorded walkthrough
- **Milestone:** repo is something you'd link at the top of your resume

---

## 7. Scope control — protect the ship date

**MVP (must ship):** all 6 agents, LangGraph orchestration, `rapidfuzz` screening, LightGBM detection, HITL escalation, audit log, one working demo path.

**Cut first if behind:** the Streamlit UI (a clean Jupyter/CLI walkthrough is acceptable), the "request more info" branch, deployment.

**Stretch goals (only if ahead):**
- Swap `rapidfuzz` for self-hosted **yente** (real production-grade matching)
- Replace LightGBM with a **GNN** (money laundering is inherently a graph problem — the IBM paper uses GNNs; this plays well to your quant background but is a real time sink)
- Adverse-media search agent (LLM + web search) as a fourth screening signal
- Deploy to a cloud endpoint

Do **not** start a stretch goal until the MVP demos end-to-end.

---

## 8. Portfolio & interview framing

The build is half the value; the *story* is the other half. Prepare these:

- **The one-liner** (top of §0) — memorize it.
- **The architecture diagram** — the graph in §3, cleaned up. Interviewers will ask you to walk through it.
- **Three talking points that signal seniority:**
  1. *"I put a human-in-the-loop interrupt on high-risk cases because in a regulated workflow you can't have an agent auto-approving — the value is in triaging, not replacing the analyst."*
  2. *"I used a trained classifier for detection and reserved the LLM for reasoning and narrative — LLMs are the wrong tool for high-volume scoring on cost and latency."*
  3. *"Every node appends to an audit log, so any decision is fully traceable — which is the actual blocker for deploying agents in finance, not model quality."*
- **Honest limitations** — be ready to say what you'd do with real data, real-time streaming, and model monitoring. Naming the gaps reads as senior, not weak.

---

## 9. First concrete step

Create the repo, set up the state schema from §3, and stub out the six nodes so an empty graph runs end-to-end with dummy data. Getting the skeleton flowing first means every later piece has a place to plug in.
