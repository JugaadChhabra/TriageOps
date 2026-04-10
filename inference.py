"""
Baseline inference agent for TriageOps.

Uses the official OpenEnv `EnvClient` SDK (via the `TriageOpsEnv` client class
defined in client.py at project root) to drive the environment over WebSocket.

Reads environment variables (per hackathon spec):
  - API_BASE_URL  (default: https://api.openai.com/v1)
  - MODEL_NAME    (default: gpt-4o-mini)
  - HF_TOKEN      (mandatory — used as the OpenAI/HF API key)

Optional:
  - LOCAL_IMAGE_NAME  — Docker image name. If set, the env is launched via
                        TriageOpsEnv.from_docker_image(IMAGE_NAME). Otherwise
                        the script connects to ENV_URL (default localhost:8000).
  - ENV_URL           — direct base URL of an already-running TriageOps server

Emits exactly the OpenEnv structured stdout format:
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Optional

from openai import OpenAI

from client import TriageOpsEnv
from models import ActionType, Department, TriageAction

# ── Configuration ──────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

# Optional: spin up the env via Docker image instead of connecting to a URL
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

BENCHMARK = "triageops"
SUCCESS_SCORE_THRESHOLD = 0.5
MAX_TICKETS_IN_PROMPT = 10
MAX_HISTORY = 6

TASKS = [
    {"name": "ticket_classification", "description": "Easy: classify 10 tickets into correct departments"},
    {"name": "triage_prioritize", "description": "Medium: triage 20 tickets with limited capacity, P0/P1 first"},
    {"name": "full_resolution", "description": "Hard: 50+ streaming tickets — full resolution pipeline"},
]

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

# ── OpenAI client ──────────────────────────────────────────────────────────

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ── Logging Functions (exact OpenEnv plain-text format) ───────────────────


def log_start(task: str, env: str, model: str) -> None:
    """[START] task=<task_name> env=<benchmark> model=<model_name>"""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str] = None,
) -> None:
    """[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>"""
    done_str = "true" if done else "false"
    error_str = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: list[float],
) -> None:
    """[END] success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>"""
    success_str = "true" if success else "false"
    clamped_score = min(max(score, 0.0), 1.0)
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_str} steps={steps} score={clamped_score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Prompt building ────────────────────────────────────────────────────────


def _ticket_to_dict(t: Any) -> dict:
    """Convert a TicketView (Pydantic) into a plain dict for prompt building."""
    if hasattr(t, "model_dump"):
        return t.model_dump()
    if isinstance(t, dict):
        return t
    return {
        "id": getattr(t, "id", ""),
        "subject": getattr(t, "subject", ""),
        "description": getattr(t, "description", ""),
        "customer_tier": getattr(t, "customer_tier", "free"),
        "status": getattr(t, "status", "open"),
        "sla_remaining": getattr(t, "sla_remaining", 0),
        "sentiment": getattr(t, "sentiment", 0.5),
        "is_vip": getattr(t, "is_vip", False),
    }


def build_user_prompt(observation: Any) -> str:
    """Build the user-turn prompt from a TriageObservation."""
    raw_tickets = list(getattr(observation, "tickets", []) or [])
    if not raw_tickets:
        return "No tickets in queue. Episode may be ending soon."

    tickets = [_ticket_to_dict(t) for t in raw_tickets]
    display = tickets[:MAX_TICKETS_IN_PROMPT]
    truncated = len(tickets) - len(display)

    lines = []
    for t in display:
        tier = t.get("customer_tier")
        tier_str = tier.value if hasattr(tier, "value") else str(tier)
        status = t.get("status")
        status_str = status.value if hasattr(status, "value") else str(status)
        vip_marker = " VIP" if t.get("is_vip") else ""
        lines.append(
            f"[{t.get('id')}] SLA:{t.get('sla_remaining')} | Tier:{tier_str}{vip_marker} | "
            f"Sentiment:{t.get('sentiment')} | Status:{status_str}\n"
            f"  Subject: {t.get('subject')}\n"
            f"  Description: {t.get('description')}"
        )

    tickets_text = "\n\n".join(lines)

    sla_warnings = list(getattr(observation, "sla_warnings", []) or [])
    warnings_line = ""
    if sla_warnings:
        warnings_line = f"\nSLA WARNINGS (breach imminent): {', '.join(sla_warnings)}"

    truncated_line = ""
    if truncated > 0:
        truncated_line = f"\n({truncated} more tickets not shown — focus on the most urgent above)"

    cap = getattr(observation, "capacity_per_step", 0)
    used = getattr(observation, "actions_this_step", 0)
    actions_left = max(0, cap - used)

    return (
        f"Step {getattr(observation, 'current_step', 0)}/{getattr(observation, 'max_steps', 0)} | "
        f"Actions left: {actions_left} | "
        f"Resolved: {getattr(observation, 'tickets_resolved', 0)} | "
        f"Breached: {getattr(observation, 'tickets_breached', 0)}"
        f"{warnings_line}\n\n"
        f"TICKETS:\n{tickets_text}{truncated_line}\n\n"
        f"Choose ONE action as a JSON object:"
    )


# ── Action parsing & cleaning ──────────────────────────────────────────────


def parse_action(response_text: str) -> Optional[dict]:
    """Extract a JSON action object from the LLM response."""
    text = response_text.strip()

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

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        return json.loads(text[start : end + 1])
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


VALID_DEPTS = {"billing", "engineering", "security", "account_management", "general_support"}


def clean_action_dict(action: dict) -> dict:
    """Validate and normalize an action dict before constructing TriageAction."""
    cleaned: dict = {
        "action_type": action.get("action_type", "respond"),
        "ticket_id": action.get("ticket_id", ""),
    }
    at = cleaned["action_type"]
    if at == "respond":
        cleaned["response_text"] = action.get(
            "response_text",
            "I apologize for the inconvenience. I have resolved this issue immediately.",
        )
    elif at == "escalate":
        dept = action.get("target_department", "general_support")
        if dept not in VALID_DEPTS:
            dept = "general_support"
        cleaned["target_department"] = dept
    elif at == "merge":
        merge_id = action.get("merge_with_id")
        if merge_id:
            cleaned["merge_with_id"] = merge_id
        else:
            cleaned["action_type"] = "respond"
            cleaned["response_text"] = "I apologize. I have resolved this issue immediately."
    return cleaned


def dict_to_triage_action(d: dict) -> TriageAction:
    """Convert a cleaned action dict into a typed TriageAction."""
    kwargs: dict[str, Any] = {
        "action_type": ActionType(d["action_type"]),
        "ticket_id": d["ticket_id"],
    }
    if d.get("response_text"):
        kwargs["response_text"] = d["response_text"]
    if d.get("target_department"):
        kwargs["target_department"] = Department(d["target_department"])
    if d.get("merge_with_id"):
        kwargs["merge_with_id"] = d["merge_with_id"]
    return TriageAction(**kwargs)


# ── Main loop (async, uses TriageOpsEnv SDK) ──────────────────────────────


async def run_task(env: TriageOpsEnv, task_name: str) -> dict:
    """
    Run a single task end-to-end via the typed TriageOpsEnv client.
    Always emits a [START]/[END] pair, even on exception.
    """
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    steps_taken = 0
    rewards: list[float] = []
    final_score = 0.0

    try:
        # Reset for this task. The Environment.reset(task=...) kwarg routes
        # via WebSocket to TriageEnvironment.reset().
        result = await env.reset(task=task_name)
        observation = result.observation
        done = bool(result.done)

        history: list[dict] = []

        while not done:
            tickets = list(getattr(observation, "tickets", []) or [])
            if not tickets:
                # Empty queue but episode not done — shouldn't normally happen,
                # but bail out cleanly.
                break

            user_prompt = build_user_prompt(observation)

            # Build messages with rolling conversation history for context
            messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
            for h in history[-MAX_HISTORY:]:
                messages.append(h)
            messages.append({"role": "user", "content": user_prompt})

            # Call LLM
            action_dict: Optional[dict] = None
            error_msg: Optional[str] = None
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=0,
                    max_tokens=400,
                )
                llm_text = completion.choices[0].message.content or ""
                action_dict = parse_action(llm_text)
            except Exception as e:
                error_msg = str(e)

            # Fallback if LLM failed or returned unparseable output
            if action_dict is None:
                first = _ticket_to_dict(tickets[0])
                action_dict = make_fallback_action(first)

            action_dict = clean_action_dict(action_dict)

            # Ensure ticket_id references an active ticket
            active_ids = {_ticket_to_dict(t).get("id") for t in tickets}
            if action_dict["ticket_id"] not in active_ids:
                action_dict = clean_action_dict(make_fallback_action(_ticket_to_dict(tickets[0])))

            # Build typed action and submit
            action_str = json.dumps(action_dict, separators=(",", ":"))
            try:
                triage_action = dict_to_triage_action(action_dict)
                step_result = await env.step(triage_action)
            except Exception as e:
                # Surface the error in the step log and stop the episode
                log_step(step=steps_taken, action=action_str, reward=0.0, done=False, error=str(e))
                break

            reward = float(step_result.reward or 0.0)
            rewards.append(reward)
            observation = step_result.observation
            done = bool(step_result.done)

            # Surface any per-action error from the env (e.g. invalid action)
            obs_err = getattr(observation, "last_action_error", None)
            log_error = error_msg or obs_err

            history.append({"role": "user", "content": user_prompt})
            history.append({"role": "assistant", "content": action_str})

            log_step(
                step=steps_taken,
                action=action_str,
                reward=reward,
                done=done,
                error=log_error,
            )
            steps_taken += 1

        # Episode complete — pull final score from the env state
        final_state = await env.state()
        final_score = float(getattr(final_state, "final_score", 0.0))
        # Fall back to normalized cumulative reward if grade wasn't computed
        if final_score == 0.0:
            final_score = float(getattr(final_state, "normalized_reward", 0.0))

    finally:
        success = final_score >= SUCCESS_SCORE_THRESHOLD
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

    return {"score": final_score, "steps": steps_taken, "rewards": rewards}


async def main_async() -> None:
    """Run all tasks sequentially. Connects via Docker image OR base URL."""
    # Construct the env client. If LOCAL_IMAGE_NAME is set, spin up Docker;
    # otherwise connect to an already-running server at ENV_URL.
    if LOCAL_IMAGE_NAME:
        env = await TriageOpsEnv.from_docker_image(LOCAL_IMAGE_NAME)
        owns_env = True
    else:
        env = TriageOpsEnv(base_url=ENV_URL)
        await env.connect()
        owns_env = True

    results: dict[str, dict] = {}
    try:
        for task in TASKS:
            try:
                result = await run_task(env, task["name"])
                results[task["name"]] = result
            except Exception as exc:
                # Make sure even a hard exception leaves a clean end log
                print(f"[DEBUG] Task {task['name']} crashed: {exc}", file=sys.stderr, flush=True)
                results[task["name"]] = {"score": 0.0, "steps": 0, "rewards": []}
    finally:
        if owns_env:
            try:
                await env.close()
            except Exception as exc:
                print(f"[DEBUG] env.close() error: {exc}", file=sys.stderr, flush=True)

    # Final summary on stderr (so we don't pollute the structured stdout log)
    print("\n=== Final Results ===", file=sys.stderr)
    total = 0.0
    for name, r in results.items():
        s = min(max(r["score"], 0.0), 1.0)
        total += s
        print(f"  {name}: score={s:.4f}", file=sys.stderr)
    if results:
        print(f"  average: {total / len(results):.4f}", file=sys.stderr)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
