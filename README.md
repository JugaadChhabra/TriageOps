---
title: TriageOps
emoji: 🎫
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
tags:
  - openenv
pinned: false
---

# TriageOps — AI Customer Support Ops

> **A multi-objective RL environment where AI agents triage and resolve customer support tickets under SLA pressure, capacity constraints, and competing priorities.**

Customer support is a **$400B+ industry** where teams process thousands of tickets daily under impossible trade-offs: speed vs quality, enterprise vs free-tier, escalation vs resolution, individual tickets vs queue-wide strategy. This environment turns that into a benchmark.

An agent sees a live ticket queue each step, must decide **which** tickets to act on, **how** to act (respond / escalate / defer / merge duplicates), and manage a finite action budget — while SLAs tick down, new tickets flood in, VIP customers demand attention, and departments go offline.

---

## Quick Start

### 1. Deploy to Hugging Face Spaces

```bash
# Clone and push to your HF Space
git clone https://github.com/YOUR_USERNAME/TriageOps.git
cd TriageOps

# Create HF Space (sdk: docker, tag: openenv)
huggingface-cli repo create YOUR_SPACE_NAME --type space -y
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
git push hf main
```

Your Space must be **Running** before submission. The HF metadata block at the top of this README configures it as a Docker Space on port 7860 with the `openenv` tag.

### 2. Run Locally with Docker

```bash
docker build -t triageops .
docker run -p 7860:7860 triageops
```

### 3. Run Locally without Docker

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### 4. Run the Inference Script

```bash
export API_BASE_URL="https://api.openai.com/v1"   # required, has default
export MODEL_NAME="gpt-4o-mini"                    # required, has default
export HF_TOKEN="your-api-key-here"                # MANDATORY — no default
export ENV_URL="http://localhost:7860"              # where the env server runs

python inference.py
```

`inference.py` uses the **OpenAI client** (`from openai import OpenAI`) with `HF_TOKEN` as the API key. It reads `API_BASE_URL` and `MODEL_NAME` from environment variables (both have defaults). `HF_TOKEN` is mandatory and will raise an error if not set.

---

## Validate Before Submitting

### Pre-submission checklist

```bash
# 1. Docker builds
docker build -t triageops .

# 2. Server responds
docker run -d -p 7860:7860 triageops
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" -d '{}'
# → 200 with JSON observation

# 3. openenv validate passes
pip install openenv-core
openenv validate                              # static: checks Dockerfile, pyproject.toml, openenv.yaml
openenv validate --url http://localhost:7860   # runtime: checks /health, /metadata, /schema, /mcp, /reset, /step, /state

# 4. Inference script runs (requires HF_TOKEN)
export HF_TOKEN="your-key"
export ENV_URL="http://localhost:7860"
python inference.py
# → emits [START], [STEP], [END] lines to stdout

# 5. HF Space is in "Running" state (check before submitting!)
```

### Validation results

```
openenv validate → [OK] Ready for multi-mode deployment (docker, openenv_serve, uv_run, python_module)
openenv validate --url → passed: true (6/6 criteria)
```

---

## Environment Overview

### Episode Loop

```
POST /reset → initial observation (ticket queue)
  └─ for each time step:
       agent sees: ticket queue + SLA warnings + dept status + capacity remaining
       agent sends: POST /step with action (respond/escalate/defer/merge)
       env returns: reward + updated observation + done flag
       world advances: SLAs tick, sentiment decays, new tickets arrive, bursts fire
  └─ episode ends when: time limit hit, all tickets resolved, or breach threshold exceeded
GET /grade → final score 0.0–1.0
```

### Architecture

```
                    ┌──────────────────────────────────┐
                    │          Agent (LLM)             │
                    │  reads observation → picks action │
                    └──────────┬───────────────────────┘
                               │ SupportAction (JSON)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Server (:7860)                    │
│  POST /reset ─── POST /step ─── GET /state ─── GET /grade  │
└──────────┬──────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│                  CustomerSupportEnv                          │
│                                                             │
│  ┌─────────┐   ┌──────────────┐   ┌──────────────────────┐ │
│  │ Ticket  │   │   Rewards    │   │   World Clock        │ │
│  │  Queue  │──▶│  per-action  │   │  SLA ticks           │ │
│  │         │   │  per-step    │   │  sentiment decays    │ │
│  │ 53 tpl  │   │  end-bonus   │   │  arrivals (Poisson)  │ │
│  └─────────┘   └──────────────┘   │  bursts (outage)     │ │
│                                    │  dept outages        │ │
│  ┌─────────┐   ┌──────────────┐   └──────────────────────┘ │
│  │Customer │   │  7 Graders   │                             │
│  │Profiles │   │  resolution  │   ┌──────────────────────┐ │
│  │tier/ltv │   │  priority    │   │  3 Tasks             │ │
│  │churn    │   │  SLA         │   │  easy → med → hard   │ │
│  │VIP flag │   │  quality     │   │  10 → 20 → 50+ tix  │ │
│  └─────────┘   │  duplicates  │   └──────────────────────┘ │
│                │  classific.  │                             │
│                │  critical    │                             │
│                └──────────────┘                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Observation Space

Each step, the agent sees a `QueueObservation`:

| Field | Type | Description |
|-------|------|-------------|
| `tickets` | `List[TicketView]` | Open tickets — subject, description, sentiment, SLA, tier, VIP flag, prior interactions |
| `current_step` | `int` | Current time step |
| `max_steps` | `int` | Total episode length |
| `actions_this_step` | `int` | Actions already taken this step |
| `capacity_per_step` | `int` | Max actions allowed per step |
| `department_status` | `List[DepartmentStatus]` | Each dept's queue size + whether it's accepting escalations |
| `sla_warnings` | `List[str]` | Ticket IDs with SLA <= 2 steps remaining |
| `total_reward` | `float` | Cumulative raw reward |
| `normalized_reward` | `float` | Reward normalized to 0.0–1.0 |
| `tickets_resolved` | `int` | Total resolved so far |
| `tickets_breached` | `int` | Total SLA breaches |
| `tickets_escalated` | `int` | Total escalated |

**The agent does NOT see ground-truth category or urgency.** It must infer these from the ticket description text.

## Action Space

Send a `SupportAction` JSON to `POST /step`:

| Action | Required Fields | Description |
|--------|----------------|-------------|
| `respond` | `ticket_id`, `response_text` | Resolve with a written response |
| `escalate` | `ticket_id`, `target_department` | Route to: `billing`, `engineering`, `security`, `account_management`, `general_support` |
| `defer` | `ticket_id` | Skip now, handle later (only smart if higher-priority tickets exist) |
| `merge` | `ticket_id`, `merge_with_id` | Merge a duplicate into another active ticket |

**Example action:**
```json
{
  "action_type": "respond",
  "ticket_id": "TKT-0001",
  "response_text": "I sincerely apologize for the inconvenience. I have processed your refund immediately and will follow up."
}
```

---

## Reward Design

### Per-action (dense signal)
| Action | Reward |
|--------|--------|
| Resolve ticket | `base × quality × urgency_mult × tier_mult × vip_mult + speed_bonus` (0.3–8+) |
| Correct escalation | +0.5 + speed bonus |
| Merge actual duplicate | +0.3 |
| Smart defer (SLA has room) | +0.1 |
| Wrong department | -0.3 |
| Empty close (< 10 chars) | -0.4 |
| Repeat same action on ticket | -0.3 |
| Invalid action | -0.2 |
| Escalate to overloaded/offline dept | -0.4 |

### Per-step (continuous pressure)
- P0/P1 ticket ignored (still OPEN): **-0.15 × urgency_mult** per step
- Sentiment decay: **-0.05/step** (1.5x for VIP, 2x for abusive tickets)
- Sentiment hits 0 (customer meltdown): **-1.5 × urgency_mult** and ticket auto-breaches
- SLA breach: **-1.0 × urgency_mult + -0.5** forced-escalation penalty

### Episode bonuses
- Zero breaches: **+2.0**
- All enterprise SLAs met: **+1.5**
- High resolution ratio: **up to +1.5**

### Why you can't maximize everything
- **Speed vs Quality:** Responding fast to hit SLAs means less thoughtful responses (lower quality score)
- **Coverage vs Prioritization:** Handling tickets in queue order maximizes throughput but ignores urgency ranking
- **Escalation vs Capacity:** Routing to the right department is correct, but departments have capacity limits and can go offline
- **Individual vs Batch:** Responding to 10 outage duplicates individually wastes 9 action slots; merging them saves capacity but requires recognition

---

## Tasks

### Task 1: Ticket Classification (Easy)

| Parameter | Value |
|-----------|-------|
| Tickets | 10 (no new arrivals) |
| Capacity | 10 actions/step (unlimited) |
| SLAs | Generous: P0=10, P1=12, P2=15, P3=20 steps |
| Steps | 15 |
| Focus | Can the agent correctly classify and route tickets? |
| Grader weights | Classification accuracy 50%, Resolution rate 30%, SLA 10%, Quality 10% |
| Target score | 0.80+ |

### Task 2: Triage & Prioritize (Medium)

| Parameter | Value |
|-----------|-------|
| Tickets | 20 (no new arrivals) |
| Capacity | 5 actions/step × 4 steps = exactly 20 slots |
| SLAs | Tight: P0=2, P1=3, P2=5, P3=8 steps |
| Steps | 4 |
| Focus | Handle P0/P1 first or they breach. Every action matters. |
| Grader weights | Prioritization 30%, Critical coverage 30%, Resolution 15%, SLA 15%, Classification 10% |
| Challenge | P0 tickets breach in 2 steps — must be handled immediately |

### Task 3: Full Resolution Pipeline (Hard)

| Parameter | Value |
|-----------|-------|
| Tickets | 20 initial + Poisson arrivals (rate 3.0) + 2 bursts = **50+ total** |
| Capacity | 5 actions/step × 10 steps = 50 slots (barely enough) |
| SLAs | Very tight: P0=2, P1=3, P2=5, P3=8 steps |
| Bursts | Step 2: 8 tickets (50% duplicates) / Step 5: 12 tickets (60% duplicates) |
| Surprises | VIP tickets (15%, 3x weight), Engineering dept goes offline at step 4 |
| Dept capacity | 6 tickets per department before overload |
| Grader weights | All 7 dimensions: 15% each + critical coverage 10% |
| Challenge | Even an omniscient agent scores ~0.78 — breaches are unavoidable |

---

## Grading

7 deterministic grading components, all in [0.0, 1.0]:

| Dimension | How it's measured |
|-----------|-------------------|
| **Resolution rate** | tickets handled / total tickets |
| **Prioritization** | Kendall-tau rank correlation vs ideal urgency order |
| **SLA compliance** | 1 - (breaches / total tickets) |
| **Response quality** | Keyword coverage (with synonym expansion) + empathy + actionability + sentiment alignment |
| **Duplicate detection** | F1 score: correct merges vs actual duplicates |
| **Classification accuracy** | Correct department routing + keyword relevance for responses |
| **Critical coverage** | P0/P1 tickets resolved before SLA breach |

Final score = weighted sum of components (weights vary by task). Same actions always produce the same score (seeded RNG + deterministic graders).

---

## Inference Script Output Format

`inference.py` emits exactly three line types to stdout:

```
[START] task=ticket_classification env=triageops model=gpt-4o-mini
[STEP] step=0 action={"action_type":"respond","ticket_id":"TKT-0001","response_text":"..."} reward=4.96 done=false error=null
[STEP] step=1 action={"action_type":"escalate","ticket_id":"TKT-0002","target_department":"billing"} reward=0.50 done=false error=null
...
[END] success=true steps=10 rewards=4.96,0.50,3.20,1.10,0.80,0.60,0.55,0.40,0.30,0.20
```

Rules:
- One `[START]` at episode begin
- One `[STEP]` per step, immediately after `env.step()` returns
- One `[END]` after episode ends — **always emitted**, even on exception (try/finally)
- `reward` formatted to 2 decimal places
- `done` and `success` are lowercase: `true` or `false`
- `error` is the error string or `null`
- All on single lines, no newlines within a line

---

## Test Endpoints

```bash
# Health check
curl http://localhost:7860/health
# → {"status": "healthy"}

# Metadata
curl http://localhost:7860/metadata
# → {"name": "TriageOps", "description": "...", "version": "1.0.0", "tasks": [...]}

# Schema (action/observation/state)
curl http://localhost:7860/schema

# Reset (empty body defaults to ticket_classification)
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{}'

# Reset with specific task
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task": "full_resolution"}'

# Take an action
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{
  "action_type": "respond",
  "ticket_id": "TKT-0001",
  "response_text": "I apologize for the inconvenience. I have processed your refund immediately."
}'

# Get current state
curl http://localhost:7860/state

# Get grade
curl http://localhost:7860/grade
```

---

## Baseline Scores

Fallback agent (no LLM, responds to tickets in SLA order with template responses):

| Model | Classification | Triage | Full Resolution | Average |
|-------|:---:|:---:|:---:|:---:|
| Fallback (no LLM) | 0.97 | 0.98 | 0.70 | 0.88 |

Expected LLM scores (with proper routing, response quality, and duplicate detection):

| Task | Frontier (GPT-4o) | Mid-tier (GPT-4o-mini) |
|------|:---:|:---:|
| ticket_classification | 0.90+ | 0.70–0.85 |
| triage_prioritize | 0.85+ | 0.60–0.75 |
| full_resolution | 0.50–0.78 | 0.30–0.50 |

*Scores are deterministic given the same model and temperature=0.*

---

## Why This Matters

### Domain Novelty

OpenEnv environments focus on code, data, and structured tasks. **Customer support ops is underrepresented** despite being one of the highest-volume AI deployment domains. This environment fills that gap:

- **Natural language understanding** — agent infers ticket category and urgency from free text, not labels
- **Multi-objective Pareto frontier** — no strategy maximizes all 7 grading dimensions. Speed vs quality. Coverage vs prioritization. Escalation vs department capacity.
- **Dynamic surprises** — VIP tickets (3x weight), department outages mid-episode, ticket bursts simulating outage floods, customer sentiment meltdowns
- **Adversarial inputs** — abusive customers, repeat callers, multi-issue tickets requiring decomposition
- **Real operational constraints** — capacity limits, department backlogs, SLA clocks, enterprise vs free-tier priority mirrors actual Zendesk/Freshdesk dynamics

### Connection to RL Research

This environment is a constrained multi-objective MDP with:
- **Partial observability** — ground-truth labels hidden, agent must infer from text
- **Stochastic transitions** — Poisson arrivals, burst events, sentiment dynamics
- **Competing reward signals** — dense per-action + continuous per-step pressure + episode bonuses
- **Planning horizon matters** — easy task is reactive, medium is tactical, hard requires strategic anticipation

---

## Project Structure

```
├── openenv.yaml          # OpenEnv metadata + action/observation schemas
├── pyproject.toml         # Python project config (openenv_serve, uv_run support)
├── inference.py           # Baseline inference script ([START]/[STEP]/[END] format)
├── Dockerfile             # python:3.11-slim, port 7860, uvicorn
├── .dockerignore          # Excludes venv, .git, __pycache__
├── requirements.txt       # fastapi, uvicorn, pydantic, httpx, openai, pyyaml
├── README.md              # This file (also HF Space metadata)
├── uv.lock                # Lockfile for uv_run deployment
├── server/
│   ├── __init__.py
│   ├── app.py             # FastAPI: /reset, /step, /state, /grade, /health, /metadata, /schema, /mcp
│   ├── environment.py     # Core state machine (CustomerSupportEnv)
│   ├── models.py          # Pydantic schemas (all typed with Field descriptions)
│   ├── tickets.py         # Ticket generation engine (53 templates, 8 categories)
│   └── tasks/
│       ├── task1.json     # ticket_classification (Easy)
│       ├── task2.json     # triage_prioritize (Medium)
│       └── task3.json     # full_resolution (Hard)
└── tests/
    ├── __init__.py
    └── test_env.py        # 31 unit tests (pytest)
```

---

## Hardware Requirements

Runs within hackathon constraints:
- **2 vCPU / 8 GB RAM** — no heavy models, no compilation-required deps
- **Base image:** `python:3.11-slim`
- **Inference runtime:** < 5 minutes for all 3 tasks (well under 20-minute limit)
- **Memory footprint:** < 200 MB (ticket data is lightweight)

---

## License

MIT
