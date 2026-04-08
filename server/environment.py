"""Core state machine: CustomerSupportEnv with reset/step/state/grade."""

import random
from typing import Optional

from .models import (
    ActionType,
    Department,
    DepartmentStatus,
    GradeResult,
    QueueObservation,
    StepResult,
    SupportAction,
    TaskConfig,
    Ticket,
    TicketStatus,
    TicketUrgency,
    TicketView,
)
from .tickets import TicketGenerator

# ── Reward Constants ───────────────────────────────────────────────────────

BASE_RESOLVE_REWARD = 1.0
ESCALATE_CORRECT_REWARD = 0.5
ESCALATE_WRONG_PENALTY = -0.3
MERGE_REWARD = 0.3
DEFER_REWARD = 0.1
INVALID_ACTION_PENALTY = -0.2
BREACH_PENALTY = -1.0

URGENCY_MULT = {
    TicketUrgency.P0: 3.0,
    TicketUrgency.P1: 2.0,
    TicketUrgency.P2: 1.0,
    TicketUrgency.P3: 0.5,
}
TIER_MULT = {
    "free": 1.0,
    "pro": 1.5,
    "enterprise": 2.5,
}

SENTIMENT_DECAY = 0.05
ZERO_BREACH_BONUS = 2.0
ENTERPRISE_SLA_BONUS = 1.5
MAX_RESOLUTION_BONUS = 1.5

# Empathy and actionability words for response quality scoring
EMPATHY_WORDS = [
    "sorry", "apologize", "understand", "frustrating", "inconvenience",
    "appreciate", "patience", "help", "concern", "regret",
]
ACTIONABILITY_PHRASES = [
    "i have", "i've", "we will", "we'll", "i will", "i'll",
    "processed", "completed", "resolved", "fixed", "refund",
    "next step", "follow up", "immediately", "right away",
]


class CustomerSupportEnv:
    """RL environment for customer support ticket triage."""

    def __init__(self) -> None:
        self.tickets: dict[str, Ticket] = {}
        self.resolved_tickets: list[Ticket] = []
        self.action_log: list[dict] = []
        self.current_step: int = 0
        self.actions_this_step: int = 0
        self.total_reward: float = 0.0
        self.done: bool = False
        self.config: Optional[TaskConfig] = None
        self.generator: Optional[TicketGenerator] = None
        self.rng: Optional[random.Random] = None
        self.department_queues: dict[Department, int] = {d: 0 for d in Department}
        self.merge_map: dict[str, str] = {}  # merged_id → target_id
        self._response_qualities: list[float] = []
        self._resolution_order: list[str] = []
        self._detected_duplicates: set[str] = set()
        self._actual_duplicates: set[str] = set()

    def reset(self, config: TaskConfig) -> StepResult:
        """Initialize a new episode."""
        self.config = config
        self.rng = random.Random(config.seed)
        self.generator = TicketGenerator(seed=config.seed, sla_steps=config.sla_steps)
        self.tickets = {}
        self.resolved_tickets = []
        self.action_log = []
        self.current_step = 0
        self.actions_this_step = 0
        self.total_reward = 0.0
        self.done = False
        self.department_queues = {d: 0 for d in Department}
        self.merge_map = {}
        self._response_qualities = []
        self._resolution_order = []
        self._detected_duplicates = set()
        self._actual_duplicates = set()

        # Generate initial tickets
        initial = self.generator.generate_batch(config.initial_tickets, current_step=0)
        for t in initial:
            self.tickets[t.id] = t
            if t.duplicate_of is not None:
                self._actual_duplicates.add(t.id)

        obs = self._build_observation()
        return StepResult(observation=obs, reward=0.0, done=False, info={"message": "Episode started"})

    def step(self, action: SupportAction) -> StepResult:
        """Process one agent action."""
        if self.done:
            obs = self._build_observation()
            return StepResult(
                observation=obs, reward=0.0, done=True,
                info={"error": "Episode already done"},
            )

        reward, info = self._process_action(action)
        self.total_reward = round(self.total_reward + reward, 4)
        self.actions_this_step += 1
        self.action_log.append({
            "step": self.current_step,
            "action": action.model_dump(),
            "reward": reward,
            "info": info,
        })

        # Auto-advance time if capacity reached
        if self.actions_this_step >= self.config.capacity_per_step:
            self._advance_time()

        # Check termination
        self._check_done()

        obs = self._build_observation()
        return StepResult(observation=obs, reward=round(reward, 4), done=self.done, info=info)

    def advance_step(self) -> StepResult:
        """Manually advance to the next time step."""
        if self.done:
            obs = self._build_observation()
            return StepResult(observation=obs, reward=0.0, done=True, info={"error": "Episode already done"})

        self._advance_time()
        self._check_done()
        obs = self._build_observation()
        return StepResult(observation=obs, reward=0.0, done=self.done, info={"message": f"Advanced to step {self.current_step}"})

    def state(self) -> QueueObservation:
        """Return current observation."""
        return self._build_observation()

    def grade(self) -> GradeResult:
        """Compute final grade for the episode."""
        if self.config is None:
            return GradeResult(score=0.0, breakdown={}, details={"error": "No episode started"})

        weights = self.config.grader_weights

        # Resolution rate
        total_tickets = len(self.tickets) + len(self.resolved_tickets)
        resolved_count = len(self.resolved_tickets)
        resolution_rate = resolved_count / max(total_tickets, 1)

        # SLA compliance
        breached = sum(
            1 for t in list(self.tickets.values()) + self.resolved_tickets
            if t.status == TicketStatus.BREACHED
        )
        sla_compliance = 1.0 - (breached / max(total_tickets, 1))

        # Prioritization (Kendall-tau on resolution order)
        prioritization = self._grade_prioritization()

        # Response quality
        response_quality = self._grade_response_quality()

        # Duplicate detection F1
        duplicate_f1 = self._grade_duplicate_detection()

        breakdown = {
            "resolution_rate": round(resolution_rate, 4),
            "prioritization": round(prioritization, 4),
            "sla_compliance": round(sla_compliance, 4),
            "response_quality": round(response_quality, 4),
            "duplicate_detection": round(duplicate_f1, 4),
        }

        score = sum(breakdown[k] * weights.get(k, 0.0) for k in breakdown)
        score = round(max(0.0, min(1.0, score)), 4)

        details = {
            "total_tickets": str(total_tickets),
            "resolved": str(resolved_count),
            "breached": str(breached),
            "total_reward": str(round(self.total_reward, 4)),
        }

        return GradeResult(score=score, breakdown=breakdown, details=details)

    # ── Action Handlers ────────────────────────────────────────────────────

    def _process_action(self, action: SupportAction) -> tuple[float, dict]:
        ticket = self.tickets.get(action.ticket_id)
        if ticket is None:
            return INVALID_ACTION_PENALTY, {"error": f"Ticket {action.ticket_id} not found or already resolved"}

        if ticket.status in (TicketStatus.RESOLVED, TicketStatus.MERGED, TicketStatus.ESCALATED):
            return INVALID_ACTION_PENALTY, {"error": f"Ticket {action.ticket_id} already {ticket.status.value}"}

        if action.action_type == ActionType.RESPOND:
            return self._handle_respond(ticket, action)
        elif action.action_type == ActionType.ESCALATE:
            return self._handle_escalate(ticket, action)
        elif action.action_type == ActionType.MERGE:
            return self._handle_merge(ticket, action)
        elif action.action_type == ActionType.DEFER:
            return self._handle_defer(ticket, action)
        else:
            return INVALID_ACTION_PENALTY, {"error": f"Unknown action type: {action.action_type}"}

    def _handle_respond(self, ticket: Ticket, action: SupportAction) -> tuple[float, dict]:
        response_text = action.response_text or ""
        quality = self._evaluate_response_quality(response_text, ticket)
        self._response_qualities.append(quality)

        u_mult = URGENCY_MULT[ticket.urgency]
        t_mult = TIER_MULT[ticket.customer.tier.value]
        reward = round(BASE_RESOLVE_REWARD * quality * u_mult * t_mult, 4)

        ticket.status = TicketStatus.RESOLVED
        self._resolution_order.append(ticket.id)
        self.resolved_tickets.append(ticket)
        del self.tickets[ticket.id]

        return reward, {
            "action": "respond",
            "ticket_id": ticket.id,
            "quality": round(quality, 4),
            "reward": reward,
        }

    def _handle_escalate(self, ticket: Ticket, action: SupportAction) -> tuple[float, dict]:
        target_dept = action.target_department
        if target_dept is None:
            return INVALID_ACTION_PENALTY, {"error": "Escalation requires target_department"}

        correct = target_dept == ticket.required_department
        if correct:
            reward = ESCALATE_CORRECT_REWARD
        else:
            reward = ESCALATE_WRONG_PENALTY

        ticket.status = TicketStatus.ESCALATED
        self.department_queues[target_dept] = self.department_queues.get(target_dept, 0) + 1
        self._resolution_order.append(ticket.id)
        self.resolved_tickets.append(ticket)
        del self.tickets[ticket.id]

        return round(reward, 4), {
            "action": "escalate",
            "ticket_id": ticket.id,
            "target_department": target_dept.value,
            "correct_department": correct,
            "reward": round(reward, 4),
        }

    def _handle_merge(self, ticket: Ticket, action: SupportAction) -> tuple[float, dict]:
        merge_target_id = action.merge_with_id
        if merge_target_id is None:
            return INVALID_ACTION_PENALTY, {"error": "Merge requires merge_with_id"}

        target = self.tickets.get(merge_target_id)
        if target is None:
            # Target might already be resolved
            return INVALID_ACTION_PENALTY, {"error": f"Merge target {merge_target_id} not found in active tickets"}

        is_actual_dup = ticket.duplicate_of == merge_target_id or target.duplicate_of == ticket.id
        if is_actual_dup:
            self._detected_duplicates.add(ticket.id)

        ticket.status = TicketStatus.MERGED
        self.merge_map[ticket.id] = merge_target_id
        self._resolution_order.append(ticket.id)
        self.resolved_tickets.append(ticket)
        del self.tickets[ticket.id]

        reward = MERGE_REWARD if is_actual_dup else round(MERGE_REWARD * 0.3, 4)
        return round(reward, 4), {
            "action": "merge",
            "ticket_id": ticket.id,
            "merged_with": merge_target_id,
            "was_duplicate": is_actual_dup,
            "reward": round(reward, 4),
        }

    def _handle_defer(self, ticket: Ticket, action: SupportAction) -> tuple[float, dict]:
        """Defer: mark as in-progress but don't resolve. Smart if SLA has room."""
        if ticket.sla_remaining <= 1:
            # Bad defer — SLA about to breach
            reward = INVALID_ACTION_PENALTY
            info_msg = "Deferred ticket about to breach SLA"
        else:
            reward = DEFER_REWARD
            info_msg = "Ticket deferred"

        ticket.status = TicketStatus.IN_PROGRESS

        return round(reward, 4), {
            "action": "defer",
            "ticket_id": ticket.id,
            "sla_remaining": ticket.sla_remaining,
            "reward": round(reward, 4),
            "message": info_msg,
        }

    # ── Response Quality Evaluation ────────────────────────────────────────

    def _evaluate_response_quality(self, response: str, ticket: Ticket) -> float:
        if not response.strip():
            return 0.0

        response_lower = response.lower()

        # Keyword coverage (50%)
        keywords = ticket.resolution_keywords
        if keywords:
            matched = sum(1 for kw in keywords if kw.lower() in response_lower)
            keyword_score = matched / len(keywords)
        else:
            keyword_score = 0.5

        # Length score (20%) — sweet spot 50-300 chars
        length = len(response)
        if length < 20:
            length_score = 0.1
        elif length < 50:
            length_score = 0.4
        elif length <= 300:
            length_score = 1.0
        elif length <= 500:
            length_score = 0.8
        else:
            length_score = 0.6

        # Empathy (15%)
        empathy_count = sum(1 for w in EMPATHY_WORDS if w in response_lower)
        empathy_score = min(1.0, empathy_count / 2)

        # Actionability (15%)
        action_count = sum(1 for p in ACTIONABILITY_PHRASES if p in response_lower)
        actionability_score = min(1.0, action_count / 2)

        quality = (
            0.50 * keyword_score
            + 0.20 * length_score
            + 0.15 * empathy_score
            + 0.15 * actionability_score
        )
        return round(max(0.0, min(1.0, quality)), 4)

    # ── Time & State Management ────────────────────────────────────────────

    def _advance_time(self) -> None:
        self.current_step += 1
        self.actions_this_step = 0

        # Tick SLAs and decay sentiment
        for ticket in list(self.tickets.values()):
            ticket.sla_remaining -= 1
            ticket.sentiment = round(max(0.0, ticket.sentiment - SENTIMENT_DECAY), 2)

            # SLA breach
            if ticket.sla_remaining <= 0 and ticket.status not in (
                TicketStatus.RESOLVED, TicketStatus.ESCALATED, TicketStatus.MERGED
            ):
                ticket.status = TicketStatus.BREACHED
                penalty = round(BREACH_PENALTY * URGENCY_MULT[ticket.urgency], 4)
                self.total_reward = round(self.total_reward + penalty, 4)

        # New arrivals via Poisson process
        if self.config and self.config.arrival_rate > 0 and self.generator:
            arrivals = self.generator.generate_arrivals(
                self.config.arrival_rate, current_step=self.current_step
            )
            for t in arrivals:
                self.tickets[t.id] = t
                if t.duplicate_of is not None:
                    self._actual_duplicates.add(t.id)

        # Burst events
        if self.config and self.generator:
            for burst in self.config.bursts:
                if burst.step == self.current_step:
                    burst_tickets = self.generator.generate_burst(burst, current_step=self.current_step)
                    for t in burst_tickets:
                        self.tickets[t.id] = t
                        if t.duplicate_of is not None:
                            self._actual_duplicates.add(t.id)

    def _check_done(self) -> None:
        if self.config is None:
            return

        # Time limit
        if self.current_step >= self.config.max_steps:
            self._apply_end_bonuses()
            self.done = True
            return

        # Breach threshold
        breached_count = sum(
            1 for t in self.tickets.values()
            if t.status == TicketStatus.BREACHED
        )
        if breached_count >= self.config.breach_threshold:
            self.done = True
            return

        # All tickets handled
        active = [
            t for t in self.tickets.values()
            if t.status in (TicketStatus.OPEN, TicketStatus.IN_PROGRESS)
        ]
        if len(active) == 0 and len(self.tickets) == 0:
            self._apply_end_bonuses()
            self.done = True

    def _apply_end_bonuses(self) -> None:
        breached = sum(
            1 for t in list(self.tickets.values()) + self.resolved_tickets
            if t.status == TicketStatus.BREACHED
        )

        # Zero breach bonus
        if breached == 0:
            self.total_reward = round(self.total_reward + ZERO_BREACH_BONUS, 4)

        # Enterprise SLA bonus
        enterprise_tickets = [
            t for t in list(self.tickets.values()) + self.resolved_tickets
            if t.customer.tier.value == "enterprise"
        ]
        if enterprise_tickets:
            enterprise_breached = sum(1 for t in enterprise_tickets if t.status == TicketStatus.BREACHED)
            if enterprise_breached == 0:
                self.total_reward = round(self.total_reward + ENTERPRISE_SLA_BONUS, 4)

        # Resolution ratio bonus
        total = len(self.tickets) + len(self.resolved_tickets)
        if total > 0:
            ratio = len(self.resolved_tickets) / total
            bonus = round(MAX_RESOLUTION_BONUS * ratio, 4)
            self.total_reward = round(self.total_reward + bonus, 4)

    # ── Grading Sub-functions ──────────────────────────────────────────────

    def _grade_prioritization(self) -> float:
        """Kendall-tau correlation between resolution order and ideal order."""
        if len(self._resolution_order) < 2:
            return 0.5

        all_tickets = {t.id: t for t in self.resolved_tickets}
        for tid, t in self.tickets.items():
            all_tickets[tid] = t

        # Ideal order: by urgency (p0 first), then by tier (enterprise first)
        urgency_rank = {"p0": 0, "p1": 1, "p2": 2, "p3": 3}
        tier_rank = {"enterprise": 0, "pro": 1, "free": 2}

        def sort_key(tid: str):
            t = all_tickets.get(tid)
            if t is None:
                return (4, 3)
            return (urgency_rank.get(t.urgency.value, 4), tier_rank.get(t.customer.tier.value, 3))

        actual_order = self._resolution_order
        ideal_order = sorted(actual_order, key=sort_key)

        # Kendall tau: count concordant and discordant pairs
        n = len(actual_order)
        concordant = 0
        discordant = 0
        actual_positions = {tid: i for i, tid in enumerate(actual_order)}
        ideal_positions = {tid: i for i, tid in enumerate(ideal_order)}

        for i in range(n):
            for j in range(i + 1, n):
                a_i, a_j = actual_order[i], actual_order[j]
                actual_diff = actual_positions[a_i] - actual_positions[a_j]
                ideal_diff = ideal_positions[a_i] - ideal_positions[a_j]
                if actual_diff * ideal_diff > 0:
                    concordant += 1
                elif actual_diff * ideal_diff < 0:
                    discordant += 1

        total_pairs = n * (n - 1) / 2
        if total_pairs == 0:
            return 0.5

        tau = (concordant - discordant) / total_pairs
        # Normalize from [-1, 1] to [0, 1]
        return round((tau + 1) / 2, 4)

    def _grade_response_quality(self) -> float:
        if not self._response_qualities:
            return 0.0
        return round(sum(self._response_qualities) / len(self._response_qualities), 4)

    def _grade_duplicate_detection(self) -> float:
        """F1 score for duplicate detection."""
        if not self._actual_duplicates:
            # No duplicates in this scenario — give full marks if none incorrectly merged
            false_merges = len(self._detected_duplicates - self._actual_duplicates)
            return 1.0 if false_merges == 0 else 0.5

        true_positives = len(self._detected_duplicates & self._actual_duplicates)
        false_positives = len(self._detected_duplicates - self._actual_duplicates)
        false_negatives = len(self._actual_duplicates - self._detected_duplicates)

        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return round(f1, 4)

    # ── Observation Builder ────────────────────────────────────────────────

    def _build_observation(self) -> QueueObservation:
        ticket_views = [
            TicketView.from_ticket(t) for t in self.tickets.values()
            if t.status in (TicketStatus.OPEN, TicketStatus.IN_PROGRESS, TicketStatus.BREACHED)
        ]

        # Sort by SLA urgency
        ticket_views.sort(key=lambda tv: tv.sla_remaining)

        sla_warnings = [
            tv.id for tv in ticket_views if tv.sla_remaining <= 2
        ]

        dept_status = [
            DepartmentStatus(
                name=dept,
                queue_size=self.department_queues.get(dept, 0),
                available=True,
            )
            for dept in Department
        ]

        breached_count = sum(
            1 for t in self.tickets.values()
            if t.status == TicketStatus.BREACHED
        )
        escalated_count = sum(
            1 for t in self.resolved_tickets
            if t.status == TicketStatus.ESCALATED
        )

        return QueueObservation(
            tickets=ticket_views,
            current_step=self.current_step,
            max_steps=self.config.max_steps if self.config else 0,
            actions_this_step=self.actions_this_step,
            capacity_per_step=self.config.capacity_per_step if self.config else 0,
            department_status=dept_status,
            sla_warnings=sla_warnings,
            total_reward=round(self.total_reward, 4),
            tickets_resolved=len(self.resolved_tickets),
            tickets_breached=breached_count,
            tickets_escalated=escalated_count,
        )
