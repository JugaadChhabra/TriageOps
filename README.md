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
- Full **OpenEnv spec compliance** — typed models, step/reset/state, openenv.yaml, passes `openenv validate`
- **3 tasks** with increasing difficulty (easy, medium, hard) and 7 deterministic graders scoring 0.0–1.0
- **Dense reward function** with per-action rewards, per-step penalties, and episode bonuses — penalizes gaming, rewards genuine triage skill
- **Baseline inference script** using OpenAI client with reproducible scores across all 3 tasks
- **Deployed on Hugging Face Spaces** with working Dockerfile

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

Tested with Llama 3.3 70B (via Groq) and fallback agent:

| Model | Classification | Triage | Full Resolution | Average |
|-------|:---:|:---:|:---:|:---:|
| Llama 3.3 70B | 0.92 | 0.96 | 0.71 | 0.86 |
| Fallback (no LLM) | 0.97 | 0.98 | 0.70 | 0.88 |

*Scores are deterministic given the same model and temperature=0. The fallback agent scores high on easy/medium tasks (just responds in SLA order) but cannot do smart routing, duplicate detection, or quality responses that an LLM agent can.*

---

## Setup & Usage Instructions

### 1. Run with Docker (recommended)

```bash
docker build -t triageops .
docker run -p 7860:7860 triageops
```

### 2. Run Locally

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### 3. Run Inference Script

```bash
export API_BASE_URL="https://api.openai.com/v1"   # required, has default
export MODEL_NAME="gpt-4o-mini"                    # required, has default
export HF_TOKEN="your-api-key-here"                # MANDATORY
export ENV_URL="http://localhost:7860"

python inference.py
```

### 4. Test Endpoints

```bash
# Health check
curl http://localhost:7860/health

# Reset (empty body defaults to ticket_classification)
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{}'

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

### 5. Validate Before Submitting

```bash
pip install openenv-core
openenv validate                              # static validation
openenv validate --url http://localhost:7860   # runtime validation (6/6 criteria)
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

```
TriageOps/
├── openenv.yaml          # OpenEnv metadata + action/observation schemas
├── pyproject.toml         # Python project config
├── inference.py           # Baseline inference script
├── Dockerfile             # python:3.11-slim, port 7860
├── .dockerignore
├── requirements.txt       # fastapi, uvicorn, pydantic, httpx, openai, pyyaml
├── README.md              # This file
├── uv.lock
├── server/
│   ├── __init__.py
│   ├── app.py             # FastAPI endpoints
│   ├── environment.py     # Core state machine
│   ├── models.py          # Pydantic schemas (all typed)
│   ├── tickets.py         # Ticket generation (53 templates)
│   └── tasks/
│       ├── task1.json     # ticket_classification (Easy)
│       ├── task2.json     # triage_prioritize (Medium)
│       └── task3.json     # full_resolution (Hard)
└── tests/
    └── test_env.py        # 31 unit tests
```

---

## Hardware Requirements

- **2 vCPU / 8 GB RAM** — no heavy models, no compilation-required deps
- **Base image:** `python:3.11-slim`
- **Inference runtime:** < 5 minutes for all 3 tasks
- **Memory footprint:** < 200 MB

---

## License

MIT
