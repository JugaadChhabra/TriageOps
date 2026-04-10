"""
TriageOps Environment — OpenEnv-spec wrapper.

This module provides `TriageEnvironment`, which inherits from
`openenv.core.env_server.interfaces.Environment` and wraps the rich
`CustomerSupportEnv` engine (in `server/environment.py`) so the standard
OpenEnv lifecycle (`reset()`, `step()`, `state` property) is exposed.

Task selection works via `reset(task=...)` kwarg. Available tasks are loaded
from `server/tasks/*.json` at startup.

Example:
    >>> env = TriageEnvironment()
    >>> obs = env.reset(task="ticket_classification")
    >>> obs = env.step(TriageAction(action_type="respond", ticket_id="TKT-0001",
    ...                              response_text="I have resolved this."))
    >>> state = env.state
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from models import (
    ActionType,
    Department,
    TaskConfig,
    TicketStatus,
    TriageAction,
    TriageObservation,
    TriageState,
)

from .environment import CustomerSupportEnv

# ── Load task configs at module import ─────────────────────────────────────

_TASKS_DIR = Path(__file__).parent / "tasks"
_TASK_CONFIGS: dict[str, TaskConfig] = {}


def _load_tasks() -> None:
    for path in sorted(_TASKS_DIR.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        config = TaskConfig(**data)
        _TASK_CONFIGS[config.name] = config


_load_tasks()

# Default task when reset() is called with no task kwarg
DEFAULT_TASK = "ticket_classification"


def get_task_configs() -> dict[str, TaskConfig]:
    """Public accessor for the loaded task configs."""
    return dict(_TASK_CONFIGS)


# ── OpenEnv Environment wrapper ────────────────────────────────────────────


class TriageEnvironment(Environment):
    """
    OpenEnv-compatible wrapper around the TriageOps customer support engine.

    Each WebSocket session gets its own TriageEnvironment instance (controlled
    by SUPPORTS_CONCURRENT_SESSIONS=True), so multiple agents can evaluate
    against the env in parallel without state leakage.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        """Initialize an empty environment. Call reset() to start an episode."""
        self._engine = CustomerSupportEnv()
        self._current_task: str = DEFAULT_TASK
        self._state = TriageState(
            episode_id=str(uuid4()),
            step_count=0,
            task_name=DEFAULT_TASK,
        )

    # ── Required OpenEnv interface ─────────────────────────────────────────

    def reset(self, task: Optional[str] = None, **kwargs: Any) -> TriageObservation:  # type: ignore[override]
        """
        Reset the environment to start a new episode.

        Args:
            task: Task identifier (one of the keys in server/tasks/*.json).
                  Defaults to `ticket_classification` if not provided.
            **kwargs: Ignored (forward-compat for openenv reset kwargs like seed/episode_id).

        Returns:
            Initial TriageObservation with the starting ticket queue.
        """
        _ = kwargs  # accepted for forward-compat with openenv reset signature
        task_name = task or DEFAULT_TASK
        if task_name not in _TASK_CONFIGS:
            raise ValueError(
                f"Unknown task '{task_name}'. Available: {list(_TASK_CONFIGS.keys())}"
            )

        config = _TASK_CONFIGS[task_name]
        self._current_task = task_name

        # Reset the underlying engine
        result = self._engine.reset(config)

        # Reset our state tracking
        self._state = TriageState(
            episode_id=str(uuid4()),
            step_count=0,
            task_name=task_name,
            total_reward=0.0,
            normalized_reward=0.0,
        )

        # The engine's observation already matches TriageObservation
        # (QueueObservation is now an alias). Just stamp done/reward fields
        # that the OpenEnv base class expects.
        obs = result.observation
        obs.done = bool(result.done)
        obs.reward = float(result.reward)
        obs.last_action_error = None
        return obs

    def step(self, action: TriageAction) -> TriageObservation:  # type: ignore[override]
        """
        Execute one agent action.

        Args:
            action: TriageAction with action_type, ticket_id, and optional fields.

        Returns:
            TriageObservation with updated queue, reward, done flag.
        """
        # Normalize string enum values that may come over the wire
        if isinstance(action.action_type, str):
            action.action_type = ActionType(action.action_type)
        if isinstance(action.target_department, str):
            action.target_department = Department(action.target_department)

        # Delegate to the engine. The engine returns a StepResult-like object
        # whose .observation is already a TriageObservation (QueueObservation alias).
        result = self._engine.step(action)

        # Update OpenEnv state
        self._state.step_count += 1
        self._state.total_reward = float(self._engine.total_reward)
        self._state.normalized_reward = float(self._engine._normalized_reward)
        self._state.tickets_resolved = len(self._engine.resolved_tickets)
        self._state.tickets_breached = sum(
            1
            for t in self._engine.tickets.values()
            if t.status == TicketStatus.BREACHED
        )

        obs = result.observation
        obs.done = bool(result.done)
        obs.reward = float(result.reward)

        # Surface any error from the action's info dict
        info = result.info or {}
        err = info.get("error")
        obs.last_action_error = str(err) if err else None

        # If the episode is done, compute final grade and stash in state
        if obs.done:
            try:
                grade = self._engine.grade()
                self._state.final_score = float(grade.score)
                self._state.grade_breakdown = dict(grade.breakdown)
            except Exception:
                pass

        return obs

    @property
    def state(self) -> TriageState:  # type: ignore[override]
        """Return the current episode state."""
        return self._state

    # ── Optional helpers for the server / inference ────────────────────────

    def get_metadata(self) -> EnvironmentMetadata:
        """Return environment metadata (used by /metadata endpoint)."""
        tasks_str = ", ".join(_TASK_CONFIGS.keys())
        return EnvironmentMetadata(
            name="triageops",
            description=(
                "AI Customer Support Ops — triage and resolve tickets under SLA pressure. "
                f"Tasks: {tasks_str}. Active: {self._current_task}."
            ),
            version="1.0.0",
            author="Pied Piper (Muaaz Shaikh, Mantek Singh Burn, Jugaad Chhabra)",
        )

    def grade(self) -> dict[str, Any]:
        """Return final grade — used by inference for the [END] log line."""
        try:
            g = self._engine.grade()
            return {
                "score": float(g.score),
                "breakdown": dict(g.breakdown),
                "details": dict(g.details),
            }
        except Exception as e:
            return {"score": 0.0, "breakdown": {}, "details": {"error": str(e)}}

    def close(self) -> None:
        """Cleanup hook (called on session end)."""
        pass
