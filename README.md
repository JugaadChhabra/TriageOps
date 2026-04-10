---
title: TriageOps
emoji: 🎫
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
tags:
  - openenv
---

# TriageOps — AI Customer Support Ops

> **An OpenEnv RL environment where AI agents triage and resolve customer support tickets under SLA pressure, capacity constraints, and competing priorities.**

**Team Pied Piper**

| Name | Email | Role |
|------|-------|------|
| Muaaz Shaikh | muaaz5731@gmail.com | Team Lead |
| Mantek Singh Burn | manteksburn@gmail.com | Member |
| Jugaad Chhabra | jugaad.chhabra@gmail.com | Member |

**Live Space:** [huggingface.co/spaces/MuaazS/TriageOps](https://huggingface.co/spaces/MuaazS/TriageOps)

---

## Problem Statement

> "Build a complete, real-world OpenEnv environment that an AI agent can learn from through the standard step() / reset() / state() API."

### Our Solution

Customer support is a **$400B+ industry** where teams process thousands of tickets daily under impossible trade-offs: speed vs quality, enterprise vs free-tier, escalation vs resolution, individual tickets vs queue-wide strategy. **TriageOps** turns that into a benchmark.

An agent sees a live ticket queue each step, must decide **which** tickets to act on, **how** to act (respond / escalate / defer / merge duplicates), and manage a finite action budget — while SLAs tick down, new tickets flood in, VIP customers demand attention, and departments go offline.

**What we built:**

- A real-world **customer support triage simulation** (not a game or toy)
- Full **OpenEnv-core SDK integration** — `TriageEnvironment` inherits from `openenv.core.env_server.interfaces.Environment`, server uses `create_app()`, and the typed `TriageOpsEnv` client subclasses `EnvClient[TriageAction, TriageObservation, TriageState]` with `from_docker_image()` support
- **Typed Pydantic models** at project root (`models.py`) inheriting from `openenv.core.env_server.types.{Action, Observation, State}`
- **3 tasks** with progressive difficulty (easy → medium → hard) and 7 deterministic graders scoring 0.0–1.0
- **Dense reward function** with per-action rewards, per-step penalties, and episode bonuses — penalizes gaming, rewards genuine triage skill, includes **Jaccard semantic overlap** to defeat keyword templating
- **Hard task** features: 50+ tickets, 4 actions/step (deficit by design), VIP weighting, mid-episode department outage, and a buried **compliance landmine** ticket worth -5.0 if missed
- **Baseline inference script** using OpenAI client + the typed `TriageOpsEnv` SDK over WebSocket, with reproducible scores across all 3 tasks
- **31 unit tests** (`pytest tests/test_env.py`) covering reset/step/state, all 7 graders, edge cases, ticket generation, and end-to-end WebSocket round-trip
- **Deployed on Hugging Face Spaces** with working Dockerfile and `openenv validate` passing

---

## Environment Overview & Motivation

### Why Customer Support Triage?

Customer support triage is a **multi-objective constrained decision problem**. You cannot maximize all objectives simultaneously:

- Responding faster means lower quality
- Handling enterprise customers first means free-tier SLAs breach
- Escalating everything avoids bad responses but overloads departments
- Merging duplicates saves capacity but wrong merges lose tickets

This creates a realistic benchmark for evaluating AI agents on **planning under uncertainty**, **prioritization**, and **human-centric decision-making**.

### Domain Novelty

Existing OpenEnv environments focus on code generation, data processing, and structured tasks. **Customer support ops is underrepresented** despite being one of the highest-volume AI deployment domains. TriageOps fills that gap with:

- **Natural language understanding required** — the agent must infer ticket category and urgency from free-text descriptions, not metadata labels
- **Multi-objective Pareto frontier** — no strategy can maximize all 7 grading dimensions simultaneously
- **Dynamic surprise mechanics** — VIP tickets (3x weight), departments going offline mid-episode, ticket bursts simulating outage floods, customer sentiment meltdowns
- **Adversarial inputs** — abusive customers, repeat callers, multi-issue tickets requiring decomposition
- **Real operational constraints** — capacity limits, department backlogs, SLA clocks, enterprise vs free-tier prioritization mirrors actual Zendesk/Freshdesk dynamics

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

Each step, the agent receives a `QueueObservation` (all fields typed via Pydantic):

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

Each `TicketView` contains: `id`, `subject`, `description`, `customer_name`, `customer_tier` (free/pro/enterprise), `status`, `sla_remaining`, `created_step`, `sentiment` (0.0–1.0), `is_vip`, `prior_interactions`.

---

## Action Space

Send a `SupportAction` JSON to `POST /step`:

| Action | Required Fields | Description |
|--------|----------------|-------------|
| `respond` | `ticket_id`, `response_text` | Resolve with a written response |
| `escalate` | `ticket_id`, `target_department` | Route to: `billing`, `engineering`, `security`, `account_management`, `general_support` |
| `defer` | `ticket_id` | Skip now, handle later (only smart if higher-priority tickets exist) |
| `merge` | `ticket_id`, `merge_with_id` | Merge a duplicate into another active ticket |

**Example:**
```json
{
  "action_type": "respond",
  "ticket_id": "TKT-0001",
  "response_text": "I sincerely apologize for the inconvenience. I have processed your refund immediately and will follow up."
}
```

---

## Task Descriptions

### Task 1: Ticket Classification (Easy)

**Objective:** Classify 10 clear-cut tickets by routing to the correct department or resolving with category-appropriate responses.

| Parameter | Value |
|-----------|-------|
| Tickets | 10 (no new arrivals) |
| Capacity | 10 actions/step (unlimited) |
| SLAs | Generous: P0=10, P1=12, P2=15, P3=20 steps |
| Steps | 15 |
| Grader weights | Classification accuracy 50%, Resolution rate 30%, SLA 10%, Quality 10% |
| Expected difficulty | Frontier models: 0.90+, mid-tier: 0.70–0.85 |

**Why it's easy:** No time pressure, no arrivals, unlimited capacity. Tests basic reading comprehension and routing.

### Task 2: Triage & Prioritize (Medium)

**Objective:** Handle 20 mixed-urgency tickets with exactly 20 action slots (5/step x 4 steps). P0/P1 must be handled first or they breach.

| Parameter | Value |
|-----------|-------|
| Tickets | 20 (no new arrivals) |
| Capacity | 5 actions/step x 4 steps = exactly 20 slots |
| SLAs | Tight: P0=2, P1=3, P2=5, P3=8 steps |
| Grader weights | Prioritization 30%, Critical coverage 30%, Resolution 15%, SLA 15%, Classification 10% |
| Expected difficulty | Frontier models: 0.85+, mid-tier: 0.60–0.75 |

**Why it's medium:** Every action matters. P0 tickets breach in 2 steps if ignored. Requires actual triage strategy, not just sequential processing.

### Task 3: Full Resolution Pipeline (Hard)

**Objective:** Handle 50+ streaming tickets across all dimensions simultaneously — classify, prioritize, respond with quality, detect duplicates, manage escalations — under extreme time pressure with surprise events.

| Parameter | Value |
|-----------|-------|
| Tickets | 20 initial + Poisson arrivals (rate 3.0) + 2 bursts = **50+ total** |
| Capacity | 5 actions/step x 10 steps = 50 slots (barely enough) |
| SLAs | Very tight: P0=2, P1=3, P2=5, P3=8 steps |
| Bursts | Step 2: 8 tickets (50% duplicates), Step 5: 12 tickets (60% duplicates) |
| Surprises | VIP tickets (15%, 3x reward weight), Engineering dept goes offline at step 4 |
| Dept capacity | 6 tickets per department before overload |
| Grader weights | All 7 dimensions at 15% each + critical coverage 10% |
| Expected difficulty | Frontier models: 0.50–0.78, mid-tier: 0.30–0.50 |

**Why it's hard:** Even an omniscient agent scores ~0.78 because capacity (50 actions) barely covers ticket volume (50+), tight SLAs guarantee some breaches, and the engineering department going offline mid-episode forces strategy adaptation. Duplicate detection, VIP prioritization, and sentiment management all compete for the same action slots.

---

## Reward Design

### Per-action (dense signal)

| Action | Reward |
|--------|--------|
| Resolve ticket | `base x quality x urgency_mult x tier_mult x vip_mult + speed_bonus` (0.3–8+) |
| Correct escalation | +0.5 + speed bonus |
| Merge actual duplicate | +0.3 |
| Smart defer (SLA has room) | +0.1 |
| Wrong department | -0.3 |
| Empty close (< 10 chars) | -0.4 |
| Repeat same action on ticket | -0.3 |
| Invalid action | -0.2 |
| Escalate to overloaded/offline dept | -0.4 |

### Per-step (continuous pressure)

- P0/P1 ticket ignored (still OPEN): **-0.15 x urgency_mult** per step
- Sentiment decay: **-0.05/step** (1.5x for VIP, 2x for abusive tickets)
- Sentiment hits 0 (customer meltdown): **-1.5 x urgency_mult**, ticket auto-breaches
- SLA breach: **-1.0 x urgency_mult + -0.5** forced-escalation penalty

### Episode bonuses

- Zero breaches: **+2.0**
- All enterprise SLAs met: **+1.5**
- High resolution ratio: **up to +1.5**

### Multi-objective trade-offs (why you can't maximize everything)

- **Speed vs Quality:** Responding fast to hit SLAs means less thoughtful responses (lower quality score)
- **Coverage vs Prioritization:** Handling tickets in queue order maximizes throughput but ignores urgency ranking
- **Escalation vs Capacity:** Routing to the right department is correct, but departments have capacity limits and can go offline
- **Individual vs Batch:** Responding to 10 outage duplicates individually wastes 9 action slots; merging saves capacity but requires recognition

---

## Grading

7 deterministic grading components, all producing scores in [0.0, 1.0]:

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

## Baseline Performance Scores

Tested with Llama 3.3 70B (via Groq) and the deterministic fallback agent. Each task is run with `temperature=0` and a fixed RNG seed, so scores reproduce exactly.

| Model | Classification (Easy) | Triage (Medium) | Full Resolution (Hard) | Average |
|-------|:---:|:---:|:---:|:---:|
| **Llama 3.3 70B** (via Groq, free tier) | 0.92 | 0.96 | 0.65 | 0.84 |
| **Fallback (no LLM)** | 0.97 | 0.97 | 0.64 | 0.86 |

The fallback agent is a deterministic heuristic (handle tickets in SLA order with template responses). It scores high on easy/medium tasks because they reward correct ordering, but struggles on the **hard task** because it can't:
- Recognise the **compliance landmine** ticket (one buried high-stakes ticket worth -5.0 if missed)
- Detect duplicate floods and merge them
- Handle the **engineering department outage** at step 4
- Stay within the **4-action/step capacity** (50+ tickets vs 40 action slots = strategic deferrals required)

This is what makes the hard task genuinely challenging — even an omniscient agent caps out around 0.65–0.70 because resource constraints mathematically prevent perfect coverage.

---

## Using the Typed SDK Client (Python)

After deployment, the environment is consumable as a typed Python client (over WebSocket):

```python
import asyncio
from client import TriageOpsEnv
from models import TriageAction, ActionType

async def main():
    # Option A: spin up a Docker container and connect
    env = await TriageOpsEnv.from_docker_image("triageops:latest")

    # Option B: connect to an already-running server (HF Space, local docker, uv run)
    # async with TriageOpsEnv(base_url="https://muaazs-triageops.hf.space") as env:

    try:
        # Reset selects the task; default is ticket_classification
        result = await env.reset(task="full_resolution")
        print(f"{len(result.observation.tickets)} initial tickets")

        # Step with a fully typed action
        tid = result.observation.tickets[0].id
        result = await env.step(TriageAction(
            action_type=ActionType.RESPOND,
            ticket_id=tid,
            response_text="I sincerely apologize. I have resolved this immediately and will follow up.",
        ))
        print(f"reward={result.reward:.2f}, done={result.done}")

        # Get full episode state with grade breakdown
        state = await env.state()
        print(f"step_count={state.step_count}, score={state.final_score}")
    finally:
        await env.close()

asyncio.run(main())
```

For sync code, use the `.sync()` wrapper:
```python
with TriageOpsEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset()
    result = env.step(TriageAction(action_type=ActionType.RESPOND, ticket_id="TKT-0001", response_text="..."))
```

---

## Setup & Usage Instructions

### 1. Run via uv (the OpenEnv way)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync deps and start the server (uses [project.scripts] server entry point)
uv sync
uv run server
# → Uvicorn running on http://0.0.0.0:8000
```

### 2. Run with Docker (HF Spaces deployment path)

```bash
docker build -t triageops .
docker run -p 8000:8000 triageops
```

### 3. Run Locally without uv

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 4. Run the Inference Script

```bash
export API_BASE_URL="https://api.openai.com/v1"   # required, has default
export MODEL_NAME="gpt-4o-mini"                    # required, has default
export HF_TOKEN="your-api-key-here"                # MANDATORY
export ENV_URL="http://localhost:8000"             # if connecting to running server
# OR set LOCAL_IMAGE_NAME="triageops:latest"       # to spin up via Docker

python inference.py
```

The inference script uses the typed `TriageOpsEnv` client over WebSocket and emits the exact OpenEnv `[START]` / `[STEP]` / `[END]` plain-text format to stdout.

### 5. Test Endpoints

```bash
# Health check
curl http://localhost:8000/health
# {"status":"healthy"}

# Metadata
curl http://localhost:8000/metadata

# Schema (action / observation / state JSON Schemas)
curl http://localhost:8000/schema

# Reset (HTTP path is stateless — use the WebSocket client for real episodes)
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{}'

# Get current state
curl http://localhost:8000/state
```

### 6. Validate Before Submitting

```bash
pip install openenv-core
openenv validate                                # static validation (5/5 deployment modes)
openenv validate --url http://localhost:8000     # runtime validation (6/6 criteria)
```

### 7. Run the unit tests

```bash
pytest tests/test_env.py -v
# 31 passed in ~3s
```

---

## Inference Script Output Format

`inference.py` emits exactly three line types to stdout:

```
[START] task=ticket_classification env=triageops model=gpt-4o-mini
[STEP] step=0 action={"action_type":"respond","ticket_id":"TKT-0001","response_text":"..."} reward=4.96 done=false error=null
[STEP] step=1 action={"action_type":"escalate","ticket_id":"TKT-0002","target_department":"billing"} reward=0.50 done=false error=null
[END] success=true steps=10 score=0.917 rewards=4.96,0.50,3.20,1.10,0.80,0.60,0.55,0.40,0.30,0.20
```

---

## Connection to RL Research

This environment is a constrained multi-objective MDP with:
- **Partial observability** — ground-truth labels hidden, agent must infer from text
- **Stochastic transitions** — Poisson arrivals, burst events, sentiment dynamics
- **Competing reward signals** — dense per-action + continuous per-step pressure + episode bonuses
- **Planning horizon matters** — easy task is reactive, medium is tactical, hard requires strategic anticipation

---

## Project Structure

Follows the official OpenEnv project layout (from `openenv init`):

```
TriageOps/
├── openenv.yaml              # OpenEnv manifest (spec_version, runtime, port, tasks)
├── pyproject.toml             # Python project + [project.scripts] server entry point
├── uv.lock                    # Reproducible dependency lockfile
├── requirements.txt           # Pip-installable deps (mirrors pyproject)
├── Dockerfile                 # Mirror of server/Dockerfile (root + server/ both work)
├── .dockerignore
├── README.md                  # This file (also HF Space metadata header)
│
├── models.py                  # ◉ Action/Observation/State at ROOT (OpenEnv convention)
│                              #   TriageAction(Action), TriageObservation(Observation),
│                              #   TriageState(State) — all inherit from openenv.core.env_server.types
│
├── client.py                  # ◉ TriageOpsEnv(EnvClient[...]) at ROOT (OpenEnv convention)
│                              #   Implements _step_payload, _parse_result, _parse_state
│                              #   Inherits from_docker_image(), reset(), step(), close() for free
│
├── __init__.py                # Re-exports TriageOpsEnv + models for `from triageops import ...`
│
├── inference.py               # Baseline LLM agent — uses async TriageOpsEnv client over WebSocket
│                              #   Emits exact [START]/[STEP]/[END] plain-text format
│
├── server/
│   ├── __init__.py
│   ├── app.py                 # FastAPI app via openenv create_app(TriageEnvironment, ...)
│   │                          #   Auto-creates /reset, /step, /state, /ws, /health, /schema, /docs
│   ├── triage_environment.py  # TriageEnvironment(Environment) — OpenEnv-spec wrapper
│   ├── environment.py         # CustomerSupportEnv — the rich domain engine (8 enums, 53 tpl,
│   │                          #   7 graders, sentiment, VIP, dept outage, landmines)
│   ├── models.py              # Backward-compat shim that re-exports root models.py
│   ├── tickets.py             # Ticket generation engine (53 templates, Poisson arrivals, bursts)
│   ├── Dockerfile             # python:3.11-slim, port 8000, uv-based deps
│   └── tasks/
│       ├── task1.json         # ticket_classification (Easy)
│       ├── task2.json         # triage_prioritize (Medium)
│       └── task3.json         # full_resolution (Hard)
│
└── tests/
    ├── __init__.py
    └── test_env.py            # 31 unit tests (reset/step/state, all 7 graders, edge cases,
                               #   ticket generation, end-to-end WebSocket round-trip)
```

### Key Architectural Decisions

- **`models.py` and `client.py` at project root** — matches the OpenEnv scaffold convention so the typed Python client is `from triageops.client import TriageOpsEnv` (or `from client import TriageOpsEnv` when running locally)
- **`server/triage_environment.py`** wraps the existing `CustomerSupportEnv` engine in a thin `Environment` subclass — preserves all the rich domain logic while exposing the OpenEnv contract
- **`server/app.py` is a 30-line factory call** — `create_app(TriageEnvironment, TriageAction, TriageObservation, env_name="triageops")` — no custom routing, no manual session handling
- **Backward-compat shim** (`server/models.py` re-exports root `models.py`) lets the legacy domain engine keep its `from .models import ...` imports working without changes

---

## Hardware Requirements

- **2 vCPU / 8 GB RAM** — no heavy models, no compilation-required deps
- **Base image:** `python:3.11-slim`
- **Inference runtime:** < 5 minutes for all 3 tasks
- **Memory footprint:** < 200 MB

---

## License

MIT
