"""FastAPI endpoints for the TriageOps environment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .environment import CustomerSupportEnv
from .models import GradeResult, QueueObservation, StepResult, SupportAction, TaskConfig

# ── Load Task Configs ──────────────────────────────────────────────────────

TASKS_DIR = Path(__file__).parent / "tasks"
TASK_CONFIGS: dict[str, TaskConfig] = {}


def _load_tasks() -> None:
    for path in sorted(TASKS_DIR.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        config = TaskConfig(**data)
        TASK_CONFIGS[config.name] = config


_load_tasks()

# ── App Setup ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="TriageOps",
    description="AI Customer Support Ops — RL Environment",
    version="1.0.0",
)

env = CustomerSupportEnv()


# ── Request Models ─────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: Optional[str] = None


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "TriageOps",
        "description": "AI Customer Support Ops RL Environment",
        "tasks": list(TASK_CONFIGS.keys()),
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    return {
        "name": "TriageOps",
        "description": "AI Customer Support Ops — triage and resolve tickets under SLA pressure",
        "version": "1.0.0",
        "tasks": list(TASK_CONFIGS.keys()),
    }


@app.get("/schema")
def schema():
    return {
        "action": SupportAction.model_json_schema(),
        "observation": QueueObservation.model_json_schema(),
        "state": QueueObservation.model_json_schema(),
    }


@app.post("/mcp")
def mcp(body: dict = {}):
    """Minimal JSON-RPC endpoint for OpenEnv MCP compatibility."""
    return {
        "jsonrpc": "2.0",
        "id": body.get("id", 1),
        "result": {
            "name": "TriageOps",
            "version": "1.0.0",
        },
    }


@app.get("/tasks")
def list_tasks():
    return {
        name: {"description": cfg.description, "initial_tickets": cfg.initial_tickets, "max_steps": cfg.max_steps}
        for name, cfg in TASK_CONFIGS.items()
    }


@app.post("/reset")
def reset(request: ResetRequest = ResetRequest()) -> StepResult:
    task_name = request.task or "ticket_classification"
    if task_name not in TASK_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Task '{task_name}' not found. Available: {list(TASK_CONFIGS.keys())}")
    config = TASK_CONFIGS[task_name]
    return env.reset(config)


@app.post("/step")
def step(action: SupportAction) -> StepResult:
    if env.config is None:
        raise HTTPException(status_code=400, detail="No episode in progress. Call /reset first.")
    return env.step(action)


@app.post("/advance_step")
def advance_step() -> StepResult:
    if env.config is None:
        raise HTTPException(status_code=400, detail="No episode in progress. Call /reset first.")
    return env.advance_step()


@app.get("/state")
def get_state() -> QueueObservation:
    if env.config is None:
        raise HTTPException(status_code=400, detail="No episode in progress. Call /reset first.")
    return env.state()


@app.get("/grade")
def get_grade() -> GradeResult:
    return env.grade()


def main() -> None:
    """Entry point for running the server directly."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
