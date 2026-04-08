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
reset(task_id) → initial observation
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
| `actions_remaining` | `int` | Actions left this step |
| `capacity_per_step` | `int` | Max actions per step |
| `departments` | `List[DepartmentStatus]` | Department backlog and availability |
| `sla_warnings` | `List[str]` | Ticket IDs with SLA ≤ 2 steps |
| `resolved_count` | `int` | Total resolved so far |
| `breached_count` | `int` | Total SLA breaches |
| `score_so_far` | `float` | Running normalized score |

**Note:** The agent does NOT see ground-truth category or urgency labels. It must infer these from the ticket description text.

### Action Space

| Action | Required Fields | Description |
|--------|----------------|-------------|
| `respond` | `ticket_id`, `response_text` | Resolve a ticket with a written response |
| `escalate` | `ticket_id`, `target_department` | Route to: billing, engineering, account_management, legal, general |
| `defer` | `ticket_id` | Skip this ticket to handle higher-priority ones |
| `merge` | `ticket_id`, `merge_with_id` | Merge a duplicate into another ticket |

### Reward Design

**Per-action (dense):**
- Resolve: `base × quality × urgency_mult × tier_mult` (0.3–5.0)
- Correct escalation: +0.5–1.0
- Good merge: +0.3
- Smart defer (higher-priority exists): +0.1
- Wrong department: -0.3
- Invalid action: -0.2

**Per-step (continuous pressure):**
- SLA near-breach: -0.05 per ticket per step
- Sentiment decay: -0.02 per waiting ticket per step
- SLA breach: -1.0 × urgency multiplier

**Episode bonuses:**
- Zero breaches: +2.0
- All enterprise SLAs met: +1.5
- High resolution ratio: up to +1.5

---

## Tasks

### Task 1: Morning Shift (Easy)
- **Scenario:** 10 clear-cut tickets, no new arrivals, generous SLAs (6–10 steps)
- **Capacity:** 3 actions/step, 5 steps
- **Tests:** Basic classification, response quality, correct routing
- **Expected scores:** Frontier models 0.7+, mid-tier 0.4–0.6

### Task 2: Peak Hours (Medium)
- **Scenario:** 15 initial tickets + ~3.5 arrivals/step, 15% duplicates, tight SLAs (4–7)
- **Capacity:** 3 actions/step, 8 steps
- **Tests:** Prioritization under pressure, duplicate detection, triage strategy
- **Expected scores:** Frontier models 0.5–0.7, mid-tier 0.25–0.4

### Task 3: Outage Crisis (Hard)
- **Scenario:** 10 initial + burst of 10 outage tickets at step 3 (60% duplicates), enterprise churn risk, compliance ticket buried in noise, engineering dept overloaded
- **Capacity:** 4 actions/step, 12 steps
- **Tests:** Crisis management, batch duplicate handling, strategic deferral, composure
- **Expected scores:** Frontier models 0.3–0.5, mid-tier 0.1–0.25

---

## Grading

Each task uses a weighted composite grader:

| Dimension | Description |
|-----------|-------------|
| Resolution rate | Fraction of tickets handled (resolved + escalated + merged) |
| Prioritization | Kendall-tau ordering — were urgent tickets handled first? |
| SLA compliance | 1 - (breaches / total tickets) |
| Response quality | Keyword coverage + length + empathy + actionability |
| Duplicate detection | F1 score of correct merges vs actual duplicates |

Weights vary by task (e.g., prioritization matters more in Peak Hours).

---

## Setup & Usage

### Docker (recommended)

```bash
docker build -t csops .
docker run -p 7860:7860 csops
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

# Reset with task
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id": "morning_shift"}'

# Take an action
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{
  "action_type": "respond",
  "ticket_id": "TKT-0001",
  "response_text": "I apologize for the inconvenience. I have processed your refund."
}'

# Get grade
curl http://localhost:7860/grade
```

### Run inference

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export OPENAI_API_KEY="sk-..."
export ENV_URL="http://localhost:7860"

python inference.py
```

---

## Baseline Scores

| Model | Morning Shift | Peak Hours | Outage Crisis | Average |
|-------|:---:|:---:|:---:|:---:|
| GPT-4o-mini | 0.52 | 0.31 | 0.18 | 0.34 |

*Scores are deterministic given the same model and seed.*

---

## Project Structure

```
├── openenv.yaml          # OpenEnv metadata
├── inference.py           # Baseline inference script
├── Dockerfile             # Container definition
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── server/
    ├── __init__.py
    ├── app.py             # FastAPI endpoints
    ├── environment.py     # Core state machine
    ├── models.py          # Pydantic schemas
    ├── tickets.py         # Ticket generation engine
    └── tasks/
        ├── task1.json     # Morning Shift config
        ├── task2.json     # Peak Hours config
        └── task3.json     # Outage Crisis config
```

---

## License

MIT