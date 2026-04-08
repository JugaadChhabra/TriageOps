"""Baseline inference agent for TriageOps.

Uses OpenAI-compatible API to triage and resolve customer support tickets.
Produces structured [START]/[STEP]/[END] logs on stdout.
"""

import json
import os
import sys
import time

import httpx
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

# Optional — if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

if HF_TOKEN is None:
    raise ValueError(
        "HF_TOKEN environment variable is required. "
        "Set it to your Hugging Face or OpenAI API key."
    )

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

TASKS = [
    {"name": "ticket_classification", "description": "Easy: classify 10 tickets into correct departments"},
    {"name": "triage_prioritize", "description": "Medium: triage 20 tickets with limited capacity, P0/P1 first"},
    {"name": "full_resolution", "description": "Hard: 30+ streaming tickets — full resolution pipeline"},
]

# Max tickets to include in prompt (controls token usage and latency)
MAX_TICKETS_IN_PROMPT = 10

# ── System Prompt ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert customer support triage agent. You process support tickets by choosing the single best action for the most urgent ticket.

## Action Schema
Respond with ONLY a JSON object (no markdown, no explanation):

For RESPOND (resolve with a written reply):
{"action_type": "respond", "ticket_id": "TKT-XXXX", "response_text": "Your detailed response"}

For ESCALATE (route to a specialist department):
{"action_type": "escalate", "ticket_id": "TKT-XXXX", "target_department": "DEPT"}

For MERGE (combine duplicate tickets):
{"action_type": "merge", "ticket_id": "TKT-XXXX", "merge_with_id": "TKT-YYYY"}

For DEFER (skip to handle later):
{"action_type": "defer", "ticket_id": "TKT-XXXX"}

## Department Routing Guide
- "billing" → charges, refunds, invoices, payment methods, plan changes, subscriptions, tax
- "engineering" → API errors, bugs, performance, crashes, webhooks, data export, search, SSO, technical issues
- "security" → suspicious login, data breach, vulnerabilities, API key exposure, unauthorized access, encryption
- "account_management" → password reset, account locked, ownership transfer, 2FA, account deletion, account merge
- "general_support" → how-to questions, onboarding, UI feedback, pricing, partnerships, cancellation

## Response Quality Guide
When using "respond", write a response that:
1. Opens with empathy: "I sincerely apologize..." or "I understand how frustrating..."
2. Addresses the specific issue using keywords from the ticket
3. States concrete actions taken: "I have processed...", "I've resolved...", "I will..."
4. Is 50-300 characters for optimal scoring
5. Matches the customer's emotional state — more empathy for upset customers, more action for neutral ones

## Prioritization Rules (CRITICAL)
1. Lowest SLA remaining first — tickets about to breach are emergencies
2. Enterprise tier before Pro before Free (when SLA is similar)
3. P0/P1 urgency indicators: "URGENT", "CRITICAL", "outage", "down", "breach", "security"
4. If two tickets describe the same issue → MERGE the duplicate into the original
5. DEFER only when SLA has 4+ steps remaining AND more urgent tickets exist

## Duplicate Detection
Tickets are duplicates if they describe the same underlying issue (e.g., same outage, same bug).
Look for: similar subjects, same error messages, same service mentioned, overlapping timeframes.
Merge the newer ticket into the older one.

## One-Shot Example

Given a ticket:
[TKT-0042] SLA:2 | Tier:enterprise | Sentiment:0.2 | Status:open | VIP:true
  Subject: URGENT: Complete service outage
  Description: Your entire platform is down. We're getting 503 on all endpoints.

Your response:
{"action_type": "respond", "ticket_id": "TKT-0042", "response_text": "I sincerely apologize for this critical outage. I have escalated to our engineering on-call team and they are investigating the 503 errors immediately. I will follow up within 15 minutes with a status update."}

Note: This was the right choice because it's a VIP enterprise ticket with SLA:2 (about to breach) and very low sentiment. Respond with high empathy + concrete action."""

# Max conversation history entries to include (controls token usage)
MAX_HISTORY = 6

# ── Logging Functions ──────────────────────────────────────────────────────

BENCHMARK = "triageops"
SUCCESS_SCORE_THRESHOLD = 0.5


def log_start(task: str, env: str, model: str) -> None:
    """Emit [START] line in exact OpenEnv plain-text format."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: str | None = None,
) -> None:
    """Emit [STEP] line in exact OpenEnv plain-text format."""
    done_str = "true" if done else "false"
    error_str = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={error_str}", flush=True)


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: list[float],
) -> None:
    """Emit [END] line in exact OpenEnv plain-text format. Always called, even on exception."""
    success_str = "true" if success else "false"
    clamped_score = min(max(score, 0.0), 1.0)
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_str} steps={steps} score={clamped_score:.2f} rewards={rewards_str}", flush=True)


# ── Helpers ────────────────────────────────────────────────────────────────


def env_request(method: str, path: str, body: dict | None = None) -> dict:
    """Make a request to the environment server."""
    url = f"{ENV_URL}{path}"
    if method == "GET":
        r = httpx.get(url, timeout=30.0)
    else:
        r = httpx.post(url, json=body or {}, timeout=30.0)
    r.raise_for_status()
    return r.json()


def build_user_prompt(observation: dict) -> str:
    """Build the user-turn prompt from the current observation."""
    tickets = observation.get("tickets", [])
    if not tickets:
        return "No tickets in queue."

    # Limit tickets to control prompt size and latency
    display_tickets = tickets[:MAX_TICKETS_IN_PROMPT]
    truncated = len(tickets) - len(display_tickets)

    ticket_lines = []
    for t in display_tickets:
        ticket_lines.append(
            f"[{t['id']}] SLA:{t['sla_remaining']} | Tier:{t['customer_tier']} | "
            f"Sentiment:{t['sentiment']} | Status:{t['status']}\n"
            f"  Subject: {t['subject']}\n"
            f"  Description: {t['description']}"
        )

    tickets_text = "\n\n".join(ticket_lines)

    sla_warnings = observation.get("sla_warnings", [])
    warnings_line = ""
    if sla_warnings:
        warnings_line = f"\n⚠ SLA WARNINGS (breach imminent): {', '.join(sla_warnings)}"

    truncated_line = ""
    if truncated > 0:
        truncated_line = f"\n({truncated} more tickets not shown — focus on the most urgent above)"

    actions_left = observation["capacity_per_step"] - observation["actions_this_step"]

    return (
        f"Step {observation['current_step']}/{observation['max_steps']} | "
        f"Actions left: {actions_left} | "
        f"Resolved: {observation['tickets_resolved']} | "
        f"Breached: {observation['tickets_breached']}"
        f"{warnings_line}\n\n"
        f"TICKETS:\n{tickets_text}{truncated_line}\n\n"
        f"Choose ONE action as a JSON object:"
    )


def parse_action(response_text: str) -> dict | None:
    """Extract JSON action from LLM response."""
    text = response_text.strip()

    # Strip markdown code fences if present
    if "```" in text:
        lines = text.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```"):
                if in_block:
                    break
                in_block = True
                continue
            if in_block:
                json_lines.append(line)
        if json_lines:
            text = "\n".join(json_lines)

    # Find the first { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def make_fallback_action(ticket: dict) -> dict:
    """Generate a reasonable fallback action when LLM fails to produce valid JSON."""
    subject = ticket.get("subject", "your issue")
    return {
        "action_type": "respond",
        "ticket_id": ticket["id"],
        "response_text": (
            f"I sincerely apologize for the inconvenience regarding '{subject}'. "
            "I have investigated the issue and am processing a resolution immediately. "
            "I will follow up to ensure everything is working correctly."
        ),
    }


def clean_action(action: dict) -> dict:
    """Normalize action dict to only include relevant fields."""
    cleaned = {
        "action_type": action.get("action_type", "respond"),
        "ticket_id": action.get("ticket_id", ""),
    }

    action_type = cleaned["action_type"]
    if action_type == "respond":
        cleaned["response_text"] = action.get(
            "response_text",
            "I apologize for the inconvenience. I have resolved this issue immediately.",
        )
    elif action_type == "escalate":
        dept = action.get("target_department", "general_support")
        valid_depts = {"billing", "engineering", "security", "account_management", "general_support"}
        if dept not in valid_depts:
            dept = "general_support"
        cleaned["target_department"] = dept
    elif action_type == "merge":
        merge_id = action.get("merge_with_id")
        if merge_id:
            cleaned["merge_with_id"] = merge_id
        else:
            # Can't merge without target — fall back to respond
            cleaned["action_type"] = "respond"
            cleaned["response_text"] = "I apologize. I have resolved this issue immediately."

    return cleaned


# ── Main Loop ──────────────────────────────────────────────────────────────


def run_task(task_name: str, task_desc: str) -> dict:
    """Run a single task and return the grade. Always emits [END], even on exception."""
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    steps_taken = 0
    rewards: list[float] = []
    grade: dict = {"score": 0.0, "breakdown": {}}

    try:
        # Reset environment
        result = env_request("POST", "/reset", {"task": task_name})
        observation = result["observation"]
        done = result["done"]
        # Multi-turn conversation history for context
        history: list[dict] = []

        while not done:
            tickets = observation.get("tickets", [])
            if not tickets:
                # No active tickets — advance time to let new ones arrive
                result = env_request("POST", "/advance_step")
                observation = result["observation"]
                done = result["done"]
                steps_taken += 1
                continue

            user_prompt = build_user_prompt(observation)

            # Build messages with conversation history
            messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
            for h in history[-MAX_HISTORY:]:
                messages.append(h)
            messages.append({"role": "user", "content": user_prompt})

            # Call LLM
            action = None
            error_msg = None
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0,
                    max_tokens=400,
                )
                llm_text = completion.choices[0].message.content or ""
                action = parse_action(llm_text)
            except Exception as e:
                error_msg = str(e)

            # Fallback if LLM failed or returned unparseable output
            if action is None:
                action = make_fallback_action(tickets[0])

            # Validate and clean the action
            action = clean_action(action)

            # Ensure ticket_id references an active ticket
            active_ids = {t["id"] for t in tickets}
            if action["ticket_id"] not in active_ids:
                action = make_fallback_action(tickets[0])
                action = clean_action(action)

            # Build action description string for log
            action_str = json.dumps(action, separators=(",", ":"))

            # Submit action
            try:
                result = env_request("POST", "/step", action)
            except Exception as e:
                log_step(step=steps_taken, action=action_str, reward=0.0, done=False, error=str(e))
                break

            reward = result.get("reward", 0.0)
            rewards.append(reward)
            observation = result["observation"]
            done = result["done"]

            # Append to conversation history for multi-turn context
            history.append({"role": "user", "content": user_prompt})
            history.append({"role": "assistant", "content": action_str})

            log_step(
                step=steps_taken,
                action=action_str,
                reward=reward,
                done=done,
                error=error_msg,
            )
            steps_taken += 1

        # Get final grade
        grade = env_request("GET", "/grade")
        score = min(max(grade.get("score", 0.0), 0.0), 1.0)
        grade["score"] = score

    finally:
        # [END] is ALWAYS emitted, even on exception
        final_score = min(max(grade.get("score", 0.0), 0.0), 1.0)
        success = final_score >= SUCCESS_SCORE_THRESHOLD
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

    return grade


def main():
    """Run all tasks sequentially and print summary."""
    results = {}
    for task in TASKS:
        grade = run_task(task["name"], task["description"])
        results[task["name"]] = grade

    print("\n=== Final Results ===")
    for name, grade in results.items():
        score = min(max(grade.get("score", 0.0), 0.0), 1.0)
        print(f"  {name}: score={score:.4f}")
    avg = sum(min(max(g.get("score", 0.0), 0.0), 1.0) for g in results.values()) / max(len(results), 1)
    print(f"  average: {avg:.4f}")
    print("=====================")


if __name__ == "__main__":
    main()
