"""
TriageOps Environment Client.

A typed OpenEnv client for the TriageOps customer support triage environment.
Inherits from openenv.core.EnvClient and provides:
  - WebSocket-based persistent session management
  - from_docker_image() classmethod for spinning up the env in Docker
  - Async + sync (.sync()) usage modes
  - Typed reset/step/state with TriageAction/TriageObservation/TriageState
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from models import (
    DepartmentStatus,
    TicketStatus,
    TicketView,
    TriageAction,
    TriageObservation,
    TriageState,
)


class TriageOpsEnv(EnvClient[TriageAction, TriageObservation, TriageState]):
    """
    Client for the TriageOps customer support environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance has its own dedicated environment session.

    Example (Docker):
        >>> import asyncio
        >>> from client import TriageOpsEnv
        >>> from models import TriageAction, ActionType
        >>>
        >>> async def main():
        ...     env = await TriageOpsEnv.from_docker_image("triageops:latest")
        ...     try:
        ...         result = await env.reset()
        ...         tickets = result.observation.tickets
        ...         result = await env.step(TriageAction(
        ...             action_type=ActionType.RESPOND,
        ...             ticket_id=tickets[0].id,
        ...             response_text="I sincerely apologize. I have resolved this immediately.",
        ...         ))
        ...         print(f"reward={result.reward}, done={result.done}")
        ...     finally:
        ...         await env.close()
        >>>
        >>> asyncio.run(main())

    Example (existing server):
        >>> async with TriageOpsEnv(base_url="http://localhost:8000") as env:
        ...     result = await env.reset()

    Example (sync wrapper):
        >>> with TriageOpsEnv(base_url="http://localhost:8000").sync() as env:
        ...     result = env.reset()
    """

    def _step_payload(self, action: TriageAction) -> Dict[str, Any]:
        """Convert TriageAction to JSON payload for the step message."""
        payload: Dict[str, Any] = {
            "action_type": action.action_type.value
            if hasattr(action.action_type, "value")
            else str(action.action_type),
            "ticket_id": action.ticket_id,
        }
        if action.response_text is not None:
            payload["response_text"] = action.response_text
        if action.target_department is not None:
            payload["target_department"] = (
                action.target_department.value
                if hasattr(action.target_department, "value")
                else str(action.target_department)
            )
        if action.merge_with_id is not None:
            payload["merge_with_id"] = action.merge_with_id
        return payload

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[TriageObservation]:
        """Parse server response into a typed StepResult[TriageObservation]."""
        obs_data = payload.get("observation", {}) or {}

        # Reconstruct ticket views (server sends as plain dicts)
        ticket_views: list[TicketView] = []
        for t in obs_data.get("tickets", []) or []:
            try:
                ticket_views.append(TicketView(**t))
            except Exception:
                # Be lenient — if the server adds fields, ignore unknowns
                ticket_views.append(
                    TicketView(
                        id=t.get("id", ""),
                        subject=t.get("subject", ""),
                        description=t.get("description", ""),
                        customer_name=t.get("customer_name", ""),
                        customer_tier=t.get("customer_tier", "free"),
                        status=t.get("status", TicketStatus.OPEN.value),
                        sla_remaining=int(t.get("sla_remaining", 0)),
                        created_step=int(t.get("created_step", 0)),
                        sentiment=float(t.get("sentiment", 0.5)),
                        is_vip=bool(t.get("is_vip", False)),
                        prior_interactions=int(t.get("prior_interactions", 0)),
                    )
                )

        dept_status: list[DepartmentStatus] = []
        for d in obs_data.get("department_status", []) or []:
            try:
                dept_status.append(DepartmentStatus(**d))
            except Exception:
                continue

        observation = TriageObservation(
            tickets=ticket_views,
            current_step=int(obs_data.get("current_step", 0)),
            max_steps=int(obs_data.get("max_steps", 0)),
            actions_this_step=int(obs_data.get("actions_this_step", 0)),
            capacity_per_step=int(obs_data.get("capacity_per_step", 0)),
            department_status=dept_status,
            sla_warnings=list(obs_data.get("sla_warnings", []) or []),
            total_reward=float(obs_data.get("total_reward", 0.0)),
            normalized_reward=float(obs_data.get("normalized_reward", 0.0)),
            tickets_resolved=int(obs_data.get("tickets_resolved", 0)),
            tickets_breached=int(obs_data.get("tickets_breached", 0)),
            tickets_escalated=int(obs_data.get("tickets_escalated", 0)),
            last_action_error=obs_data.get("last_action_error"),
            done=bool(payload.get("done", False)),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}) or {},
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> TriageState:
        """Parse server response into a TriageState object."""
        return TriageState(
            episode_id=payload.get("episode_id"),
            step_count=int(payload.get("step_count", 0)),
            task_name=payload.get("task_name", ""),
            total_reward=float(payload.get("total_reward", 0.0)),
            normalized_reward=float(payload.get("normalized_reward", 0.0)),
            tickets_resolved=int(payload.get("tickets_resolved", 0)),
            tickets_breached=int(payload.get("tickets_breached", 0)),
            tickets_escalated=int(payload.get("tickets_escalated", 0)),
            grade_breakdown=payload.get("grade_breakdown", {}) or {},
            final_score=float(payload.get("final_score", 0.0)),
        )
