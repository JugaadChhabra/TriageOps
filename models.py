"""
Data models for the TriageOps Customer Support environment.

TriageOps simulates a customer support triage queue. Agents must classify,
prioritize, respond to, escalate, merge, or defer tickets under SLA pressure.

This module defines:
  - TriageAction: what the agent submits each step
  - TriageObservation: what the agent sees each step
  - TriageState: episode-level state
  - Supporting enums and helper models (Customer, Ticket, TicketView, etc.)

All Action/Observation/State classes inherit from openenv-core base types.
"""

from __future__ import annotations

import enum
from typing import Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────


class TicketCategory(str, enum.Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    FEATURE_REQUEST = "feature_request"
    OUTAGE = "outage"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    GENERAL = "general"


class TicketUrgency(str, enum.Enum):
    P0 = "p0"  # Critical
    P1 = "p1"  # High
    P2 = "p2"  # Medium
    P3 = "p3"  # Low


class CustomerTier(str, enum.Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class ActionType(str, enum.Enum):
    RESPOND = "respond"
    ESCALATE = "escalate"
    DEFER = "defer"
    MERGE = "merge"


class Department(str, enum.Enum):
    BILLING = "billing"
    ENGINEERING = "engineering"
    SECURITY = "security"
    ACCOUNT_MANAGEMENT = "account_management"
    GENERAL_SUPPORT = "general_support"


class TicketStatus(str, enum.Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    BREACHED = "breached"
    MERGED = "merged"


# ── Internal Domain Models (server-side) ───────────────────────────────────


class Customer(BaseModel):
    """Customer profile (full ground truth, server-side)."""

    name: str = Field(description="Customer display name")
    tier: CustomerTier = Field(description="Subscription tier")
    ltv: float = Field(description="Lifetime value in USD")
    churn_risk: float = Field(ge=0.0, le=1.0, description="Probability of churning")
    satisfaction: float = Field(ge=0.0, le=1.0, description="Current satisfaction score")
    prior_interactions: int = Field(default=0, description="Number of prior support interactions")


class Ticket(BaseModel):
    """Full ticket with ground-truth labels (hidden from agent)."""

    id: str = Field(description="Unique ticket ID, e.g. TKT-0001")
    subject: str = Field(description="Ticket subject line")
    description: str = Field(description="Full ticket body text")
    customer: Customer = Field(description="Customer who filed the ticket")
    category: TicketCategory = Field(description="Ground-truth category")
    urgency: TicketUrgency = Field(description="Ground-truth urgency")
    required_department: Department = Field(description="Correct department for escalation")
    resolution_keywords: list[str] = Field(description="Keywords that indicate a quality response")
    status: TicketStatus = Field(default=TicketStatus.OPEN, description="Current status")
    sla_remaining: int = Field(description="Time steps until SLA breach")
    created_step: int = Field(default=0, description="Step when ticket was created")
    sentiment: float = Field(default=0.5, ge=0.0, le=1.0, description="Customer sentiment")
    duplicate_of: Optional[str] = Field(default=None, description="ID of ticket this is a duplicate of")
    subcategory: str = Field(default="", description="More specific categorization")
    is_vip: bool = Field(default=False, description="VIP ticket — carries 3x reward weight")
    is_abusive: bool = Field(default=False, description="Contains abusive language — requires de-escalation")
    is_landmine: bool = Field(
        default=False,
        description="Compliance landmine — heavy penalty if missed (hard task only)",
    )


class TicketView(BaseModel):
    """What the agent sees — no ground-truth labels."""

    id: str = Field(description="Ticket ID")
    subject: str = Field(description="Subject line")
    description: str = Field(description="Full ticket body")
    customer_name: str = Field(description="Customer display name")
    customer_tier: CustomerTier = Field(description="Subscription tier (free/pro/enterprise)")
    status: TicketStatus = Field(description="Current ticket status")
    sla_remaining: int = Field(description="Time steps remaining before SLA breach")
    created_step: int = Field(description="Step when ticket was created")
    sentiment: float = Field(ge=0.0, le=1.0, description="Customer sentiment 0-1 (lower = angrier)")
    is_vip: bool = Field(default=False, description="VIP ticket flag")
    prior_interactions: int = Field(default=0, description="Prior support contacts from this customer")

    @classmethod
    def from_ticket(cls, ticket: Ticket) -> "TicketView":
        return cls(
            id=ticket.id,
            subject=ticket.subject,
            description=ticket.description,
            customer_name=ticket.customer.name,
            customer_tier=ticket.customer.tier,
            status=ticket.status,
            sla_remaining=ticket.sla_remaining,
            created_step=ticket.created_step,
            sentiment=ticket.sentiment,
            is_vip=ticket.is_vip,
            prior_interactions=ticket.customer.prior_interactions,
        )


class DepartmentStatus(BaseModel):
    """Status of an escalation department."""

    name: Department = Field(description="Department identifier")
    queue_size: int = Field(default=0, description="Number of tickets currently queued at this department")
    available: bool = Field(default=True, description="Whether department is accepting escalations")


# ── OpenEnv Action / Observation / State ───────────────────────────────────


class TriageAction(Action):
    """
    Action submitted by the agent each step.

    The agent picks ONE ticket and ONE action type per step:
      - respond: resolve with a written response (requires response_text)
      - escalate: route to a specialist department (requires target_department)
      - defer: skip for now to handle higher-priority tickets first
      - merge: merge a duplicate into another active ticket (requires merge_with_id)
    """

    action_type: ActionType = Field(description="Action to perform on the ticket")
    ticket_id: str = Field(description="Target ticket ID (e.g. TKT-0001)")
    response_text: Optional[str] = Field(
        default=None, description="Response text — required for action_type=respond"
    )
    target_department: Optional[Department] = Field(
        default=None,
        description="Department to escalate to — required for action_type=escalate",
    )
    merge_with_id: Optional[str] = Field(
        default=None,
        description="Target ticket ID to merge into — required for action_type=merge",
    )


class TriageObservation(Observation):
    """
    Observation returned to the agent each step.

    Contains the visible ticket queue (without ground-truth labels), capacity
    state, department availability, SLA warnings, and running score.
    """

    tickets: list[TicketView] = Field(
        default_factory=list, description="Visible ticket queue (sorted by SLA urgency)"
    )
    current_step: int = Field(default=0, description="Current time step")
    max_steps: int = Field(default=0, description="Total episode length")
    actions_this_step: int = Field(default=0, description="Actions already taken this step")
    capacity_per_step: int = Field(default=0, description="Maximum actions allowed per step")
    department_status: list[DepartmentStatus] = Field(
        default_factory=list, description="Per-department queue size and availability"
    )
    sla_warnings: list[str] = Field(
        default_factory=list, description="Ticket IDs with SLA <= 2 steps remaining"
    )
    total_reward: float = Field(default=0.0, description="Cumulative raw reward so far")
    normalized_reward: float = Field(
        default=0.0, description="Cumulative reward normalized to [0.0, 1.0]"
    )
    tickets_resolved: int = Field(default=0, description="Total tickets resolved so far")
    tickets_breached: int = Field(default=0, description="Total tickets that breached SLA")
    tickets_escalated: int = Field(default=0, description="Total tickets escalated")
    last_action_error: Optional[str] = Field(
        default=None, description="Error message from last action, if any"
    )


class TriageState(State):
    """
    Episode-level state for inspection / debugging.

    Inherits `episode_id` and `step_count` from openenv State.
    """

    task_name: str = Field(default="", description="Active task name")
    total_reward: float = Field(default=0.0, description="Cumulative raw reward")
    normalized_reward: float = Field(default=0.0, description="Reward normalized to [0, 1]")
    tickets_resolved: int = Field(default=0)
    tickets_breached: int = Field(default=0)
    tickets_escalated: int = Field(default=0)
    grade_breakdown: dict[str, float] = Field(
        default_factory=dict, description="Per-component grade scores"
    )
    final_score: float = Field(default=0.0, description="Final grade score for episode")


# ── Internal Task Configuration (loaded from server/tasks/*.json) ──────────


class BurstConfig(BaseModel):
    """Configuration for a scheduled ticket burst event."""

    step: int = Field(description="Step at which the burst occurs")
    count: int = Field(description="Number of tickets in the burst")
    duplicate_ratio: float = Field(default=0.6, description="Fraction that are duplicates")


class TaskConfig(BaseModel):
    """Full task configuration loaded from JSON."""

    name: str = Field(description="Task identifier")
    description: str = Field(default="", description="Human-readable description")
    initial_tickets: int = Field(description="Number of tickets at reset")
    max_steps: int = Field(description="Episode length in time steps")
    capacity_per_step: int = Field(description="Actions allowed per time step")
    arrival_rate: float = Field(default=0.0, description="Poisson arrival rate per step")
    breach_threshold: int = Field(default=5, description="Max breaches before forced episode end")
    sla_steps: dict[str, int] = Field(
        default_factory=lambda: {"p0": 3, "p1": 5, "p2": 8, "p3": 12},
        description="SLA deadlines per urgency level (in steps)",
    )
    seed: int = Field(default=42, description="Random seed for reproducibility")
    bursts: list[BurstConfig] = Field(default_factory=list, description="Scheduled burst events")
    enable_duplicates: bool = Field(default=False, description="Whether duplicate tickets can appear")
    vip_ratio: float = Field(default=0.0, description="Fraction of tickets that are VIP (3x weight)")
    landmine_count: int = Field(
        default=0, description="Number of compliance landmine tickets (heavy penalty if missed)"
    )
    department_outage: Optional[dict[str, int]] = Field(
        default=None,
        description="Department outage event: {department_name: step_when_it_goes_down}",
    )
    department_capacity: int = Field(
        default=10, description="Max tickets per department before overload"
    )
    grader_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "resolution_rate": 0.25,
            "prioritization": 0.20,
            "sla_compliance": 0.25,
            "response_quality": 0.15,
            "duplicate_detection": 0.15,
        },
        description="Weights for each grading component (must sum to 1.0)",
    )


# ── Grade Result (used internally by the grader) ───────────────────────────


class GradeResult(BaseModel):
    """Result from the episode grader."""

    score: float = Field(ge=0.0, le=1.0, description="Overall score 0-1")
    breakdown: dict[str, float] = Field(description="Per-component scores")
    details: dict[str, str] = Field(default_factory=dict, description="Human-readable details")


# Backward-compatible aliases (so existing imports keep working during transition)
SupportAction = TriageAction
QueueObservation = TriageObservation
