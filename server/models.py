"""
Backward-compat shim. The canonical models live in /models.py at project root
(per OpenEnv convention). This module re-exports them so existing
`from server.models import ...` imports keep working during the transition.
"""

from typing import Any

from pydantic import BaseModel, Field

from models import (
    ActionType,
    BurstConfig,
    Customer,
    CustomerTier,
    Department,
    DepartmentStatus,
    GradeResult,
    QueueObservation,
    SupportAction,
    TaskConfig,
    Ticket,
    TicketCategory,
    TicketStatus,
    TicketUrgency,
    TicketView,
    TriageAction,
    TriageObservation,
    TriageState,
)


class StepResult(BaseModel):
    """
    Internal step-result wrapper used by the legacy CustomerSupportEnv engine.
    NOT the same as `openenv.core.client_types.StepResult` (which is what the
    OpenEnv client uses). Kept here for backward-compat with environment.py.
    """

    observation: QueueObservation
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)

__all__ = [
    "ActionType",
    "BurstConfig",
    "Customer",
    "CustomerTier",
    "Department",
    "DepartmentStatus",
    "GradeResult",
    "QueueObservation",
    "StepResult",
    "SupportAction",
    "TaskConfig",
    "Ticket",
    "TicketCategory",
    "TicketStatus",
    "TicketUrgency",
    "TicketView",
    "TriageAction",
    "TriageObservation",
    "TriageState",
]
