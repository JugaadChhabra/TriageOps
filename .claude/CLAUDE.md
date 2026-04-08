# CLAUDE.md — SupportBench (AI Customer Support Ops)

## What This Project Is

An OpenEnv-compliant reinforcement learning environment for a hackathon. An AI agent triages and resolves customer support tickets under time pressure, SLA constraints, and limited capacity. It is a FastAPI server that exposes `/reset`, `/step`, `/state`, `/grade` endpoints.

**This is a competition submission.** Every decision should optimize for the scoring rubric: Real-world utility (30%), Task & grader quality (25%), Environment design (20%), Code quality & spec compliance (15%), Creativity & novelty (10%).

## Project Structure

```
/
├── CLAUDE.md              # This file
├── openenv.yaml           # OpenEnv metadata (name, tasks, models, endpoints)
├── inference.py           # Baseline agent script — uses OpenAI client, must produce [START]/[STEP]/[END] logs
├── Dockerfile             # python:3.11-slim, exposes port 7860, runs uvicorn
├── requirements.txt       # fastapi, uvicorn, pydantic, httpx, openai, pyyaml
├── README.md              # Competition README with setup, action/obs spaces, baseline scores
└── server/
    ├── __init__.py
    ├── app.py             # FastAPI endpoints: /, /tasks, /reset, /step, /state, /grade, /advance_step
    ├── models.py          # ALL Pydantic models (enums, Ticket, Customer, SupportAction, QueueObservation, StepResult, TaskConfig, etc.)
    ├── environment.py     # Core state machine: CustomerSupportEnv with reset/step/state/grade
    ├── tickets.py         # TicketGenerator: 50+ templates, customer profiles, Poisson arrivals, burst generation
    └── tasks/
        ├── task1.json     # "morning_shift" — Easy: 10 tickets, no arrivals, generous SLAs
        ├── task2.json     # "peak_hours" — Medium: 15 tickets + arrivals + duplicates
        └── task3.json     # "outage_crisis" — Hard: burst at step 3, duplicate floods, enterprise churn
```

## Key Architecture Decisions

### Domain Model (server/models.py)
- **Enums**: TicketCategory (8 types), TicketUrgency (p0-p3), CustomerTier (free/pro/enterprise), ActionType (respond/escalate/defer/merge), Department (5 depts), TicketStatus (open/in_progress/resolved/escalated/breached/merged)
- **Ticket**: has ground-truth `category`, `urgency`, `required_department`, `resolution_keywords` — these are HIDDEN from the agent
- **TicketView**: what the agent actually sees — NO ground-truth labels, agent must infer from description text
- **SupportAction**: `action_type` + `ticket_id` + optional `response_text` / `target_department` / `merge_with_id`
- **QueueObservation**: visible ticket queue, step counters, capacity, department status, SLA warnings, running score

### Environment Loop (server/environment.py)
1. `reset(task_config)` → seeds RNG, generates initial tickets, returns observation
2. `step(action)` → validates action, computes reward, updates ticket status. When `actions_this_step >= capacity_per_step`, auto-advances time.
3. Time advancement: SLAs tick down, sentiment decays (-0.05/step), breached tickets get `BREACH_PENALTY * urgency_mult`, new tickets arrive via Poisson process, bursts fire at configured step.
4. Episode ends: time limit hit, breach threshold exceeded, or all tickets handled.
5. `grade()` → weighted composite: resolution rate, prioritization (Kendall-tau), SLA compliance, response quality, duplicate detection F1.

### Reward Design
- **Dense per-action**: resolve = `base × quality × urgency_mult × tier_mult`, escalate correct = +0.5, merge = +0.3, smart defer = +0.1
- **Penalties**: wrong dept = -0.3, invalid action = -0.2, SLA breach = -1.0 × urgency_mult
- **End bonuses**: zero breaches = +2.0, enterprise SLAs met = +1.5, resolution ratio bonus up to +1.5
- **Response quality**: scored 0-1 via keyword coverage (50%), length (20%), empathy signals (15%), actionability (15%)

### Ticket Generation (server/tickets.py)
- 50+ templates across 8 categories with placeholder filling
- Customer profiles: name, tier, LTV, churn_risk, satisfaction
- Urgency weighted by category (outage → mostly p0, feature_request → mostly p3)
- Stochastic arrivals via Poisson process
- Burst generation: 60% duplicates of a primary outage ticket, one compliance ticket buried in noise

## Critical Constraints

### Competition Pass/Fail Gates (must ALL pass or disqualified)
1. **HF Space responds to POST /reset with 200** — the `/reset` endpoint MUST accept `{}` (empty body) and default to task "morning_shift"
2. **`openenv validate` passes** — openenv.yaml must be valid, models must be importable
3. **`docker build` succeeds** — Dockerfile must work standalone
4. **`inference.py` runs and produces scores** — must use OpenAI client, read `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` env vars
5. **3+ tasks with graders returning 0.0–1.0**

### Inference Script Requirements
- Named exactly `inference.py` in project root
- Uses `from openai import OpenAI` with `base_url=API_BASE_URL`, `api_key=HF_TOKEN`
- Structured stdout: `[START]`, `[STEP]`, `[END]` JSON logs — NO deviation in field names
- Runtime < 20 minutes on 2 vCPU / 8GB
- Must run all 3 tasks sequentially

### Deployment
- HF Space tagged `openenv`, sdk: docker
- Port 7860 (HF standard)
- Must handle concurrent requests gracefully (single env instance is fine for eval)

## Common Tasks You Might Be Asked To Do

### "Add a new ticket template"
Edit `server/tickets.py` → `TEMPLATES` dict. Each template is `(subject, description, resolution_keywords, subcategory)`. Use `{placeholder}` syntax for dynamic values — see `_fill_template()` for available placeholders.

### "Add a new task"
1. Create `server/tasks/task4.json` following the TaskConfig schema
2. Add entry to `openenv.yaml` under `tasks:`
3. Add task config to `TASKS` list in `inference.py`
4. The server auto-loads all `tasks/*.json` files at startup

### "Modify the reward function"
Edit `server/environment.py`. Constants at top of file: `BASE_RESOLVE_REWARD`, `BREACH_PENALTY`, etc. Core logic in `_process_action()`, `_handle_respond()`, `_handle_escalate()`, `_handle_merge()`, `_handle_defer()`. End bonuses in `_apply_end_bonuses()`.

### "Improve response quality evaluation"
Edit `_evaluate_response_quality()` in `server/environment.py`. Currently uses keyword overlap, length, empathy words, and actionability phrases. Could be improved with semantic similarity if a lightweight model is available.

### "Fix the grading"
`grade()` in `server/environment.py`. Uses `grader_weights` from TaskConfig. Sub-graders: `_grade_prioritization()` (Kendall-tau), `_grade_response_quality()`, `_grade_duplicate_detection()` (F1). Resolution rate and SLA compliance computed inline.

### "Test the environment"
```bash
# Install deps
pip install -r requirements.txt

# Run unit test
python -c "
from server.models import *
from server.tickets import TicketGenerator
from server.environment import CustomerSupportEnv
from server.app import TASK_CONFIGS
env = CustomerSupportEnv()
result = env.reset(TASK_CONFIGS['morning_shift'])
print(f'Tickets: {len(result.observation.tickets)}')
action = SupportAction(action_type=ActionType.RESPOND, ticket_id='TKT-0001', response_text='I apologize for the issue. I have processed your refund.')
step = env.step(action)
print(f'Reward: {step.reward}, Done: {step.done}')
print(f'Grade: {env.grade()}')
"

# Run server
uvicorn server.app:app --port 7860

# Test endpoints
curl -X POST localhost:7860/reset -H "Content-Type: application/json" -d '{}'
curl -X POST localhost:7860/step -H "Content-Type: application/json" -d '{"action_type":"respond","ticket_id":"TKT-0001","response_text":"I will fix this."}'
curl localhost:7860/grade

# Docker
docker build -t supportbench .
docker run -p 7860:7860 supportbench
```

### "Run inference locally"
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export OPENAI_API_KEY="sk-..."
export HF_TOKEN="$OPENAI_API_KEY"
export ENV_URL="http://localhost:7860"
python inference.py
```

## Code Style & Conventions

- **Python 3.11+**, type hints everywhere
- **Pydantic v2** for all models — use `Field(...)` with descriptions, use `model_dump()` not `.dict()`
- **Enums are str enums** — `class Foo(str, Enum)` so they serialize to strings in JSON
- **Reward values**: always `round(x, 4)` before returning
- **RNG**: always use `self.rng = random.Random(seed)` for reproducibility, never bare `random.random()`
- **Error handling**: invalid actions return penalty reward + info with error message, never raise exceptions from step()
- **Imports**: relative imports within `server/` package (`from .models import ...`)

## What NOT To Do

- Do NOT add a frontend/UI — this is a headless API environment
- Do NOT use `localStorage` or browser APIs — this is a Python server
- Do NOT change the [START]/[STEP]/[END] log format in inference.py — it's graded automatically
- Do NOT use bare `random` module — always use seeded `random.Random(seed)` instance
- Do NOT make `/reset` require a body — it must accept `{}` (empty JSON) and default to morning_shift
- Do NOT add dependencies that require compilation (keep `python:3.11-slim` compatible)
- Do NOT exceed 8GB memory — ticket generation and env state must be lightweight