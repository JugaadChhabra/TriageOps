"""Baseline inference agent for SupportBench.

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

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

TASKS = [
    {"name": "morning_shift", "description": "Easy: 10 tickets, no arrivals"},
    {"name": "peak_hours", "description": "Medium: 15 tickets + arrivals + duplicates"},
    {"name": "outage_crisis", "description": "Hard: burst + duplicate floods + enterprise churn"},
]

# ── Helpers ────────────────────────────────────────────────────────────────


def env_request(method: str, path: str, body: dict | None = None) -> dict:
    url = f"{ENV_URL}{path}"
    if method == "GET":
        r = httpx.get(url, timeout=30.0)
    else:
        r = httpx.post(url, json=body or {}, timeout=30.0)
    r.raise_for_status()
    return r.json()


def build_prompt(observation: dict) -> str:
    tickets = observation.get("tickets", [])
    if not tickets:
        return "No tickets in queue. The episode may be over."

    ticket_descriptions = []
    for t in tickets:
        ticket_descriptions.append(
            f"- [{t['id']}] (SLA: {t['sla_remaining']} steps, Tier: {t['customer_tier']}, "
            f"Sentiment: {t['sentiment']}, Status: {t['status']})\n"
            f"  Subject: {t['subject']}\n"
            f"  Description: {t['description']}"
        )

    tickets_text = "\n\n".join(ticket_descriptions)
    sla_warnings = observation.get("sla_warnings", [])
    warnings_text = f"\nSLA WARNINGS: {', '.join(sla_warnings)}" if sla_warnings else ""

    return f"""You are a customer support agent. You must triage and resolve support tickets efficiently.

Current state:
- Step: {observation['current_step']}/{observation['max_steps']}
- Actions remaining this step: {observation['capacity_per_step'] - observation['actions_this_step']}
- Tickets resolved: {observation['tickets_resolved']}
- Tickets breached: {observation['tickets_breached']}
- Score so far: {observation['total_reward']}{warnings_text}

Active tickets:
{tickets_text}

Choose ONE action. Respond with a JSON object (no markdown):
{{
  "action_type": "respond" | "escalate" | "defer" | "merge",
  "ticket_id": "TKT-XXXX",
  "response_text": "Your response to the customer (required for respond)",
  "target_department": "billing" | "engineering" | "security" | "account_management" | "general_support" (required for escalate),
  "merge_with_id": "TKT-YYYY" (required for merge)
}}

Strategy:
1. Prioritize by SLA urgency — tickets with low sla_remaining first
2. Enterprise tier customers get priority
3. For respond: include empathy, address the issue, mention specific actions taken
4. For escalate: route to the correct department based on ticket content
5. For merge: if two tickets describe the same issue from different customers
6. For defer: only if SLA has plenty of room and other tickets are more urgent

Pick the most urgent ticket and take the best action."""


def parse_action(response_text: str) -> dict | None:
    """Extract JSON action from LLM response."""
    text = response_text.strip()
    # Try to find JSON in the response
    if text.startswith("```"):
        lines = text.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.startswith("```") and not in_block:
                in_block = True
                continue
            elif line.startswith("```") and in_block:
                break
            elif in_block:
                json_lines.append(line)
        text = "\n".join(json_lines)

    # Find the first { ... } block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None

    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


# ── Main Loop ──────────────────────────────────────────────────────────────


def run_task(task_name: str, task_desc: str) -> dict:
    print(json.dumps({
        "type": "[START]",
        "task": task_name,
        "description": task_desc,
        "timestamp": time.time(),
    }))
    sys.stdout.flush()

    # Reset environment
    result = env_request("POST", "/reset", {"task": task_name})
    observation = result["observation"]
    done = result["done"]
    total_steps = 0
    total_reward = 0.0

    while not done:
        tickets = observation.get("tickets", [])
        if not tickets:
            # No tickets — advance time
            result = env_request("POST", "/advance_step")
            observation = result["observation"]
            done = result["done"]
            total_steps += 1
            continue

        prompt = build_prompt(observation)

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=512,
            )
            llm_response = completion.choices[0].message.content or ""
        except Exception as e:
            print(json.dumps({"type": "[STEP]", "error": str(e), "step": total_steps}))
            sys.stdout.flush()
            # Fallback: respond to most urgent ticket
            t = tickets[0]
            llm_response = json.dumps({
                "action_type": "respond",
                "ticket_id": t["id"],
                "response_text": f"I apologize for the inconvenience. I'm looking into your issue regarding '{t['subject']}' and will resolve it promptly.",
            })

        action = parse_action(llm_response)
        if action is None:
            # Fallback action
            t = tickets[0]
            action = {
                "action_type": "respond",
                "ticket_id": t["id"],
                "response_text": f"I understand your concern about '{t['subject']}'. I apologize for the inconvenience and I've escalated this internally for immediate resolution.",
            }

        # Clean up action — only include relevant fields
        clean_action = {
            "action_type": action["action_type"],
            "ticket_id": action["ticket_id"],
        }
        if action["action_type"] == "respond":
            clean_action["response_text"] = action.get("response_text", "I will resolve this issue.")
        elif action["action_type"] == "escalate":
            clean_action["target_department"] = action.get("target_department", "general_support")
        elif action["action_type"] == "merge":
            clean_action["merge_with_id"] = action.get("merge_with_id")

        try:
            result = env_request("POST", "/step", clean_action)
        except Exception as e:
            print(json.dumps({"type": "[STEP]", "error": str(e), "step": total_steps}))
            sys.stdout.flush()
            break

        reward = result.get("reward", 0.0)
        total_reward += reward
        observation = result["observation"]
        done = result["done"]

        print(json.dumps({
            "type": "[STEP]",
            "step": total_steps,
            "action": clean_action["action_type"],
            "ticket_id": clean_action["ticket_id"],
            "reward": reward,
            "cumulative_reward": round(total_reward, 4),
            "tickets_remaining": len(observation.get("tickets", [])),
            "done": done,
        }))
        sys.stdout.flush()
        total_steps += 1

    # Get final grade
    grade = env_request("GET", "/grade")

    print(json.dumps({
        "type": "[END]",
        "task": task_name,
        "score": grade.get("score", 0.0),
        "breakdown": grade.get("breakdown", {}),
        "total_reward": round(total_reward, 4),
        "total_steps": total_steps,
        "timestamp": time.time(),
    }))
    sys.stdout.flush()

    return grade


def main():
    results = {}
    for task in TASKS:
        grade = run_task(task["name"], task["description"])
        results[task["name"]] = grade

    print("\n=== Final Results ===")
    for name, grade in results.items():
        print(f"  {name}: score={grade.get('score', 0.0):.4f}")
    print("=====================")


if __name__ == "__main__":
    main()
