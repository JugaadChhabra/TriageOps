"""Pydantic models for the SupportBench environment."""
import enum
from typing import Optional

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


# ── Domain Models ──────────────────────────────────────────────────────────

class Customer(BaseModel):
    """Customer profile."""
    name: str = Field(description="Customer display name")
    tier: CustomerTier = Field(description="Subscription tier")
    ltv: float = Field(description="Lifetime value in USD")
    churn_risk: float = Field(ge=0.0, le=1.0, description="Probability of churning")
    satisfaction: float = Field(ge=0.0, le=1.0, description="Current satisfaction score")


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


class TicketView(BaseModel):
    """What the agent sees — no ground-truth labels."""
    id: str
    subject: str
    description: str
    customer_name: str
    customer_tier: CustomerTier
    status: TicketStatus
    sla_remaining: int
    created_step: int
    sentiment: float

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
        )


# ── Action & Observation ──────────────────────────────────────────────────

class SupportAction(BaseModel):
    """Action the agent takes on a ticket."""
    action_type: ActionType = Field(description="Type of action")
    ticket_id: str = Field(description="Target ticket ID")
    response_text: Optional[str] = Field(default=None, description="Response text (for respond action)")
    target_department: Optional[Department] = Field(default=None, description="Department to escalate to")
    merge_with_id: Optional[str] = Field(default=None, description="Ticket to merge into")


class DepartmentStatus(BaseModel):
    """Status of an escalation department."""
    name: Department
    queue_size: int = Field(default=0, description="Number of tickets currently in this department's queue")
    available: bool = Field(default=True, description="Whether department is accepting escalations")


class QueueObservation(BaseModel):
    """What the agent observes each step."""
    tickets: list[TicketView] = Field(description="Visible ticket queue")
    current_step: int = Field(description="Current time step")
    max_steps: int = Field(description="Maximum time steps")
    actions_this_step: int = Field(default=0, description="Actions taken this step")
    capacity_per_step: int = Field(description="Max actions per step")
    department_status: list[DepartmentStatus] = Field(default_factory=list)
    sla_warnings: list[str] = Field(default_factory=list, description="Tickets close to SLA breach")
    total_reward: float = Field(default=0.0, description="Cumulative reward so far")
    normalized_reward: float = Field(default=0.0, description="Reward normalized to 0.0-1.0 range")
    tickets_resolved: int = Field(default=0)
    tickets_breached: int = Field(default=0)
    tickets_escalated: int = Field(default=0)


# ── Step Result ────────────────────────────────────────────────────────────

class StepResult(BaseModel):
    """Returned from env.step()."""
    observation: QueueObservation
    reward: float
    done: bool
    info: dict = Field(default_factory=dict)


# ── Task Configuration ─────────────────────────────────────────────────────

class BurstConfig(BaseModel):
    """Configuration for a ticket burst event."""
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
    arrival_rate: float = Field(default=0.0, description="Poisson arrival rate for new tickets per step")
    breach_threshold: int = Field(default=5, description="Max breaches before forced episode end")
    sla_steps: dict[str, int] = Field(
        default_factory=lambda: {"p0": 3, "p1": 5, "p2": 8, "p3": 12},
        description="SLA deadlines per urgency level (in steps)",
    )
    seed: int = Field(default=42, description="Random seed for reproducibility")
    bursts: list[BurstConfig] = Field(default_factory=list, description="Scheduled burst events")
    enable_duplicates: bool = Field(default=False, description="Whether duplicate tickets can appear")
    grader_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "resolution_rate": 0.25,
            "prioritization": 0.20,
            "sla_compliance": 0.25,
            "response_quality": 0.15,
            "duplicate_detection": 0.15,
        },
        description="Weights for each grading component",
    )


# ── Grade Result ───────────────────────────────────────────────────────────

class GradeResult(BaseModel):
    """Result from the grader."""
    score: float = Field(ge=0.0, le=1.0, description="Overall score 0-1")
    breakdown: dict[str, float] = Field(description="Per-component scores")
    details: dict[str, str] = Field(default_factory=dict, description="Human-readable details")
