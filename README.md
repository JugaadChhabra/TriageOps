# AI Customer Support Ops — OpenEnv Environment

> **A multi-objective reinforcement learning environment that benchmarks AI agents on real-world customer support triage and resolution.**

Customer support teams process thousands of tickets daily under competing pressures: speed, quality, SLA compliance, and customer retention. This environment models that reality. An agent must classify incoming tickets, prioritize under capacity constraints, generate quality responses, detect duplicates, route escalations correctly, and manage crisis events — all while the clock ticks and SLAs expire.

---

## Why This Matters

Customer support triage is a **multi-objective constrained decision problem**. You cannot maximize all objectives simultaneously:

- Responding faster means lower quality
- Handling enterprise customers first means free-tier SLAs breach
- Escalating everything avoids bad responses but overloads departments
- Merging duplicates saves capacity but wrong merges lose tickets

This creates a realistic benchmark for evaluating AI agents on **planning under uncertainty**, **prioritization**, and **human-centric decision-making**.

---

## Environment Overview

### Episode Loop

```
reset(task) → initial observation
  └─ for each step:
       agent sees ticket queue + constraints
       agent picks action (respond / escalate / defer / merge)
       environment returns reward + updated observation
       time advances: SLAs tick, new tickets arrive, sentiment decays
  └─ episode ends: time limit, all resolved, or catastrophic breach
grade() → final score 0.0–1.0
```

### Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `tickets` | `List[TicketView]` | Open tickets with subject, description, sentiment, SLA, customer info |
| `current_step` | `int` | Current time step |
| `max_steps` | `int` | Total episode length |
| `actions_this_step` | `int` | Actions taken this step |
| `capacity_per_step` | `int` | Max actions per step |
| `department_status` | `List[DepartmentStatus]` | Department backlog and availability |
| `sla_warnings` | `List[str]` | Ticket IDs with SLA ≤ 2 steps |
| `total_reward` | `float` | Cumulative raw reward |
| `normalized_reward` | `float` | Reward normalized to 0.0–1.0 |
| `tickets_resolved` | `int` | Total resolved so far |
| `tickets_breached` | `int` | Total SLA breaches |
| `tickets_escalated` | `int` | Total escalated |

**Note:** The agent does NOT see ground-truth category or urgency labels. It must infer these from the ticket description text.

### Action Space

| Action | Required Fields | Description |
|--------|----------------|-------------|
| `respond` | `ticket_id`, `response_text` | Resolve a ticket with a written response |
| `escalate` | `ticket_id`, `target_department` | Route to: billing, engineering, security, account_management, general_support |
| `defer` | `ticket_id` | Skip this ticket to handle higher-priority ones |
| `merge` | `ticket_id`, `merge_with_id` | Merge a duplicate into another ticket |

### Reward Design

**Per-action (dense):**
- Resolve: `base × quality × urgency_mult × tier_mult + speed_bonus` (0.3–8.0)
- Correct escalation: +0.5 + speed bonus
- Good merge (actual duplicate): +0.3
- Smart defer (SLA has room): +0.1
- Wrong department: -0.3
- Empty close (< 10 chars): -0.4
- Repeat action on same ticket: -0.3
- Invalid action: -0.2

**Per-step (continuous pressure):**
- Critical ticket ignored (P0/P1 still OPEN): -0.15 × urgency multiplier
- Sentiment decay: -0.05 per waiting ticket per step
- SLA breach: -1.0 × urgency multiplier + -0.5 forced-escalation penalty

**Episode bonuses:**
- Zero breaches: +2.0
- All enterprise SLAs met: +1.5
- High resolution ratio: up to +1.5

---

## Tasks

### Task 1: Ticket Classification (Easy)
- **Scenario:** 10 clear-cut tickets, no new arrivals, generous SLAs (10–20 steps), unlimited capacity
- **Tests:** Correct department routing, category-appropriate responses
- **Grader emphasis:** Classification accuracy (50%), resolution rate (30%)
- **Threshold:** 0.8+ for success

### Task 2: Triage & Prioritize (Medium)
- **Scenario:** 20 mixed-urgency tickets, 5 actions/step × 4 steps = exactly 20 slots, no arrivals
- **Tests:** Urgency-appropriate ordering, critical ticket coverage, no P0/P1 left unhandled
- **Grader emphasis:** Prioritization (30%), critical coverage (30%)
- **Challenge:** Tight SLAs mean P0 tickets breach in 2 steps if ignored

### Task 3: Full Resolution Pipeline (Hard)
- **Scenario:** 20 initial + Poisson arrivals (rate 3.0) + 2 burst events (step 2: 8 tickets, step 5: 12 tickets) = 50+ total
- **Capacity:** 5 actions/step × 10 steps = 50 slots (barely enough)
- **Tests:** Classification, prioritization, response quality, duplicate detection, escalation routing — all simultaneously
- **Grader emphasis:** All 7 dimensions weighted equally (15% each + 10% critical coverage)
- **Challenge:** Even an omniscient agent scores ~0.78 due to unavoidable SLA breaches

---

## Grading

Each task uses a weighted composite grader. All scores are in [0.0, 1.0]:

| Dimension | Description |
|-----------|-------------|
| Resolution rate | Fraction of tickets handled (resolved + escalated + merged) |
| Prioritization | Kendall-tau ordering — were urgent tickets handled first? |
| SLA compliance | 1 - (breaches / total tickets) |
| Response quality | Keyword coverage + length + empathy + actionability + sentiment alignment |
| Duplicate detection | F1 score of correct merges vs actual duplicates |
| Classification accuracy | Correct dept routing for escalations + keyword quality for responses |
| Critical coverage | Fraction of P0/P1 tickets handled before SLA breach |

Weights vary by task (see task JSON configs for exact weights).

---

## Setup & Usage

### Docker (recommended)

```bash
docker build -t supportbench .
docker run -p 7860:7860 supportbench
```

### Local

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Test endpoints

```bash
# Health check
curl http://localhost:7860/

# List tasks
curl http://localhost:7860/tasks

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

### Run inference

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export OPENAI_API_KEY="sk-..."
export HF_TOKEN="$OPENAI_API_KEY"
export ENV_URL="http://localhost:7860"

python inference.py
```

---

## Project Structure

```
├── openenv.yaml          # OpenEnv metadata + action/observation schemas
├── inference.py           # Baseline inference script ([START]/[STEP]/[END] logs)
├── Dockerfile             # python:3.11-slim, port 7860
├── requirements.txt       # fastapi, uvicorn, pydantic, httpx, openai, pyyaml
├── README.md              # This file
└── server/
    ├── __init__.py
    ├── app.py             # FastAPI endpoints (/reset, /step, /state, /grade)
    ├── environment.py     # Core state machine (CustomerSupportEnv)
    ├── models.py          # Pydantic schemas (all typed)
    ├── tickets.py         # Ticket generation engine (50+ templates)
    └── tasks/
        ├── task1.json     # ticket_classification (Easy)
        ├── task2.json     # triage_prioritize (Medium)
        └── task3.json     # full_resolution (Hard)
```

---

## License

MIT
