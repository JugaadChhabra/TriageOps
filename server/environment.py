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

# Tiered action rewards (classification < prioritization < resolution)
BASE_RESOLVE_REWARD = 1.0
ESCALATE_CORRECT_REWARD = 0.5
ESCALATE_WRONG_PENALTY = -0.3
MERGE_REWARD = 0.3
DEFER_REWARD = 0.1
INVALID_ACTION_PENALTY = -0.2
BREACH_PENALTY = -1.0

# Speed bonus: multiplier for responding quickly relative to SLA
# speed_ratio = sla_remaining / sla_total; bonus = SPEED_BONUS_MAX * speed_ratio
SPEED_BONUS_MAX = 0.5

# Penalty for ignoring critical (P0/P1) tickets: applied per-step if critical sits untouched
CRITICAL_IGNORE_PENALTY = -0.15

# Penalty for repeat action on same ticket (loop detection)
REPEAT_ACTION_PENALTY = -0.3

# Penalty for closing without meaningful response
EMPTY_CLOSE_PENALTY = -0.4

# Penalty when a ticket breaches SLA while agent had capacity (forced escalation)
BREACH_ESCALATION_PENALTY = -0.5

# Heavy penalty for ignoring a "compliance landmine" ticket (hard task).
# Applied at episode end for each landmine left in OPEN/IN_PROGRESS/BREACHED state.
LANDMINE_PENALTY = -5.0

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

# VIP ticket multiplier
VIP_MULT = 3.0

# Penalty when sentiment hits 0 (customer rage-quits / auto-escalates)
SENTIMENT_MELTDOWN_PENALTY = -1.5

# Department overload penalty (escalation rejected)
DEPT_OVERLOAD_PENALTY = -0.4

# Maximum theoretical raw reward per ticket (for normalization)
# P0 enterprise VIP perfect response: 1.0 * 1.0 * 3.0 * 2.5 * 3.0 + speed = 24
MAX_REWARD_PER_TICKET = 10.0

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

# Synonym map for keyword matching: allows near-miss credit
KEYWORD_SYNONYMS: dict[str, list[str]] = {
    "refund": ["credit", "reimbursement", "money back", "credited"],
    "credited": ["refund", "reimbursed", "credit applied", "adjusted"],
    "fix": ["resolve", "repair", "patch", "address", "correct"],
    "resolved": ["fixed", "addressed", "corrected", "handled", "taken care"],
    "fixed": ["resolved", "repaired", "patched", "corrected"],
    "investigated": ["looked into", "reviewed", "examined", "analyzed", "checked"],
    "restored": ["recovered", "brought back", "back online", "reactivated"],
    "reset": ["changed", "updated", "regenerated", "reissued"],
    "unlocked": ["reactivated", "enabled", "restored access", "unblocked"],
    "transfer": ["move", "reassign", "migrate", "hand over"],
    "cancelled": ["terminated", "ended", "stopped", "closed"],
    "provided": ["sent", "shared", "delivered", "attached"],
    "sent": ["emailed", "delivered", "shared", "provided"],
    "confirmed": ["verified", "validated", "acknowledged"],
    "noted": ["recorded", "logged", "acknowledged", "documented"],
    "revoked": ["invalidated", "disabled", "deactivated", "cancelled"],
    "patched": ["fixed", "updated", "resolved", "deployed"],
    "deployed": ["released", "pushed", "shipped", "rolled out"],
    "processed": ["completed", "handled", "executed", "done"],
    "blocked": ["banned", "restricted", "denied", "blacklisted"],
    "exported": ["downloaded", "extracted", "generated", "saved"],
    "deleted": ["removed", "erased", "purged", "wiped"],
    "merged": ["combined", "consolidated", "unified", "joined"],
    "signed": ["executed", "completed", "finalized", "agreed"],
    "apologize": ["sorry", "regret", "apologies"],
}


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
        # (ticket_subject, ticket_description, agent_response) for LLM grading
        self._response_records: list[tuple[str, str, str]] = []
        self._resolution_order: list[str] = []
        self._detected_duplicates: set[str] = set()
        self._actual_duplicates: set[str] = set()
        self._action_history: dict[str, list[str]] = {}  # ticket_id → [action_types]
        self._initial_sla: dict[str, int] = {}  # ticket_id → original SLA at creation
        self._normalized_reward: float = 0.0
        # LLM-based response quality grader (anti-exploit). Initialized lazily;
        # falls back to None if no API credentials are available.
        from .llm_grader import LLMGrader

        self._llm_grader: Optional[LLMGrader] = LLMGrader()
        if not self._llm_grader.is_available():
            self._llm_grader = None

    def reset(self, config: TaskConfig) -> StepResult:
        """Initialize a new episode."""
        self.config = config
        self.rng = random.Random(config.seed)
        self.generator = TicketGenerator(
            seed=config.seed,
            sla_steps=config.sla_steps,
            use_realistic_templates=getattr(config, "use_realistic_templates", False),
        )
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
        self._response_records = []
        # Reset LLM grader budget for the new episode
        if self._llm_grader is not None:
            self._llm_grader.reset_budget()
        self._resolution_order = []
        self._detected_duplicates = set()
        self._actual_duplicates = set()
        self._action_history = {}
        self._initial_sla = {}
        self._normalized_reward = 0.0
        self._dept_outages: dict[str, int] = config.department_outage or {}
        self._dept_available: dict[str, bool] = {d.value: True for d in Department}

        # Generate initial tickets
        initial = self.generator.generate_batch(
            config.initial_tickets, current_step=0, vip_ratio=config.vip_ratio
        )
        for t in initial:
            self.tickets[t.id] = t
            self._initial_sla[t.id] = t.sla_remaining
            if t.duplicate_of is not None:
                self._actual_duplicates.add(t.id)

        # Generate compliance landmine tickets (hard task only)
        landmine_count = getattr(config, "landmine_count", 0)
        for _ in range(landmine_count):
            mine = self.generator.generate_landmine(current_step=0)
            self.tickets[mine.id] = mine
            self._initial_sla[mine.id] = mine.sla_remaining

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
        self._update_normalized_reward()
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
        all_tickets = list(self.tickets.values()) + self.resolved_tickets
        breached = sum(1 for t in all_tickets if t.status == TicketStatus.BREACHED)
        sla_compliance = 1.0 - (breached / max(total_tickets, 1))

        # Prioritization (Kendall-tau on resolution order)
        prioritization = self._grade_prioritization()

        # Response quality
        response_quality = self._grade_response_quality()

        # Duplicate detection F1
        duplicate_f1 = self._grade_duplicate_detection()

        # Classification accuracy (correct dept routing + keyword quality)
        classification = self._grade_classification()

        # Critical coverage (P0/P1 handled before breach)
        critical_coverage = self._grade_critical_coverage()

        breakdown = {
            "resolution_rate": round(resolution_rate, 4),
            "prioritization": round(prioritization, 4),
            "sla_compliance": round(sla_compliance, 4),
            "response_quality": round(response_quality, 4),
            "duplicate_detection": round(duplicate_f1, 4),
            "classification_accuracy": round(classification, 4),
            "critical_coverage": round(critical_coverage, 4),
        }

        # Only include components that have non-zero weight
        score = sum(breakdown.get(k, 0.0) * w for k, w in weights.items())
        score = round(max(0.0, min(1.0, score)), 4)

        details = {
            "total_tickets": str(total_tickets),
            "resolved": str(resolved_count),
            "breached": str(breached),
            "total_reward": str(round(self.total_reward, 4)),
            "normalized_reward": str(self._normalized_reward),
        }

        return GradeResult(score=score, breakdown=breakdown, details=details)

    # ── Action Handlers ────────────────────────────────────────────────────

    def _process_action(self, action: SupportAction) -> tuple[float, dict]:
        ticket = self.tickets.get(action.ticket_id)
        if ticket is None:
            return INVALID_ACTION_PENALTY, {"error": f"Ticket {action.ticket_id} not found or already resolved"}

        if ticket.status in (TicketStatus.RESOLVED, TicketStatus.MERGED, TicketStatus.ESCALATED):
            return INVALID_ACTION_PENALTY, {"error": f"Ticket {action.ticket_id} already {ticket.status.value}"}

        # Loop detection: penalize repeat identical action on same ticket
        history = self._action_history.setdefault(action.ticket_id, [])
        if action.action_type.value in history:
            return REPEAT_ACTION_PENALTY, {
                "error": f"Repeat {action.action_type.value} on {action.ticket_id}",
                "penalty": "repeat_action",
                "reward": REPEAT_ACTION_PENALTY,
            }
        history.append(action.action_type.value)

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

        # Penalty for closing without meaningful response
        if len(response_text.strip()) < 10:
            ticket.status = TicketStatus.RESOLVED
            self._resolution_order.append(ticket.id)
            self.resolved_tickets.append(ticket)
            del self.tickets[ticket.id]
            self._response_qualities.append(0.0)
            return EMPTY_CLOSE_PENALTY, {
                "action": "respond",
                "ticket_id": ticket.id,
                "quality": 0.0,
                "penalty": "empty_close",
                "reward": EMPTY_CLOSE_PENALTY,
            }

        quality = self._evaluate_response_quality(response_text, ticket)
        self._response_qualities.append(quality)
        # Record (subject, description, response) for end-of-episode LLM grading
        self._response_records.append((ticket.subject, ticket.description, response_text))

        u_mult = URGENCY_MULT[ticket.urgency]
        t_mult = TIER_MULT[ticket.customer.tier.value]
        vip_mult = VIP_MULT if ticket.is_vip else 1.0
        base_reward = BASE_RESOLVE_REWARD * quality * u_mult * t_mult * vip_mult

        # Speed bonus: decaying reward based on how quickly ticket is addressed
        initial_sla = self._initial_sla.get(ticket.id, ticket.sla_remaining)
        if initial_sla > 0:
            speed_ratio = ticket.sla_remaining / initial_sla
        else:
            speed_ratio = 0.0
        speed_bonus = SPEED_BONUS_MAX * speed_ratio * u_mult

        reward = round(base_reward + speed_bonus, 4)

        ticket.status = TicketStatus.RESOLVED
        self._resolution_order.append(ticket.id)
        self.resolved_tickets.append(ticket)
        del self.tickets[ticket.id]

        return reward, {
            "action": "respond",
            "ticket_id": ticket.id,
            "quality": round(quality, 4),
            "speed_bonus": round(speed_bonus, 4),
            "is_vip": ticket.is_vip,
            "reward": reward,
        }

    def _handle_escalate(self, ticket: Ticket, action: SupportAction) -> tuple[float, dict]:
        target_dept = action.target_department
        if target_dept is None:
            return INVALID_ACTION_PENALTY, {"error": "Escalation requires target_department"}

        # Department outage check
        if not self._dept_available.get(target_dept.value, True):
            return DEPT_OVERLOAD_PENALTY, {
                "error": f"Department {target_dept.value} is currently unavailable (outage)",
                "penalty": "dept_outage",
                "reward": DEPT_OVERLOAD_PENALTY,
            }

        # Department capacity check
        dept_cap = self.config.department_capacity if self.config else 10
        current_load = self.department_queues.get(target_dept, 0)
        if current_load >= dept_cap:
            return DEPT_OVERLOAD_PENALTY, {
                "error": f"Department {target_dept.value} is overloaded ({current_load}/{dept_cap})",
                "penalty": "dept_overload",
                "reward": DEPT_OVERLOAD_PENALTY,
            }

        correct = target_dept == ticket.required_department
        u_mult = URGENCY_MULT[ticket.urgency]

        if correct:
            # Speed bonus for quick correct escalation
            initial_sla = self._initial_sla.get(ticket.id, ticket.sla_remaining)
            speed_ratio = ticket.sla_remaining / initial_sla if initial_sla > 0 else 0.0
            speed_bonus = round(SPEED_BONUS_MAX * 0.5 * speed_ratio * u_mult, 4)
            reward = ESCALATE_CORRECT_REWARD + speed_bonus
        else:
            reward = ESCALATE_WRONG_PENALTY
            speed_bonus = 0.0

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
            "speed_bonus": speed_bonus,
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

        # ── (1) Keyword coverage with synonyms (35%) ────────────────────────
        keywords = ticket.resolution_keywords
        if keywords:
            matched = 0
            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower in response_lower:
                    matched += 1
                elif any(syn in response_lower for syn in KEYWORD_SYNONYMS.get(kw_lower, [])):
                    matched += 0.7  # partial credit for synonym
            keyword_score = min(1.0, matched / len(keywords))
        else:
            keyword_score = 0.5

        # ── (2) Jaccard semantic overlap with the ticket text (15%) ─────────
        # Lightweight bag-of-words overlap between response and ticket subject+body.
        # Anti-templating: penalizes generic responses that don't reference the
        # actual ticket content. No embeddings needed — fits in 8GB easily.
        ticket_text = (ticket.subject + " " + ticket.description).lower()
        ticket_tokens = self._content_tokens(ticket_text)
        response_tokens = self._content_tokens(response_lower)
        if ticket_tokens and response_tokens:
            intersection = ticket_tokens & response_tokens
            union = ticket_tokens | response_tokens
            jaccard = len(intersection) / max(len(union), 1)
            # Scale: jaccard >0.15 is excellent for this domain
            semantic_score = min(1.0, jaccard / 0.15)
        else:
            semantic_score = 0.0

        # ── (3) Length score (10%) — sweet spot 50-300 chars ────────────────
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

        # ── (4) Empathy (15%) ───────────────────────────────────────────────
        empathy_count = sum(1 for w in EMPATHY_WORDS if w in response_lower)
        empathy_score = min(1.0, empathy_count / 2)

        # ── (5) Actionability (10%) ─────────────────────────────────────────
        action_count = sum(1 for p in ACTIONABILITY_PHRASES if p in response_lower)
        actionability_score = min(1.0, action_count / 2)

        # ── (6) Sentiment alignment (15%) ───────────────────────────────────
        if ticket.sentiment < 0.3:
            sentiment_score = min(1.0, empathy_count / 1.5)
        elif ticket.sentiment > 0.7:
            sentiment_score = min(1.0, action_count / 1.5)
        else:
            sentiment_score = min(1.0, (empathy_count + action_count) / 3)

        quality = (
            0.35 * keyword_score
            + 0.15 * semantic_score
            + 0.10 * length_score
            + 0.15 * empathy_score
            + 0.10 * actionability_score
            + 0.15 * sentiment_score
        )
        return round(max(0.0, min(1.0, quality)), 4)

    @staticmethod
    def _content_tokens(text: str) -> set[str]:
        """Tokenize text into lowercase content words for Jaccard similarity."""
        # Strip punctuation, drop short tokens, drop common stopwords
        import re

        stopwords = {
            "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "must", "shall",
            "can", "i", "you", "we", "they", "he", "she", "it", "this", "that",
            "these", "those", "my", "your", "our", "their", "his", "her", "its",
            "for", "of", "in", "on", "at", "to", "from", "with", "by", "as",
            "if", "so", "no", "not", "yes", "all", "any", "some", "each", "more",
            "most", "other", "such", "only", "own", "same", "than", "too", "very",
            "s", "t", "just", "now", "also", "about", "very", "out", "up", "down",
            "what", "which", "who", "when", "where", "why", "how", "im", "ive",
            "ill", "id", "youre", "weve", "well", "wont", "dont", "cant", "isnt",
            "please", "thank", "thanks", "hi", "hello", "regards",
        }
        tokens = {
            t for t in re.findall(r"[a-z][a-z]{2,}", text)
            if t not in stopwords
        }
        return tokens

    # ── Reward Normalization ───────────────────────────────────────────────

    def _update_normalized_reward(self) -> None:
        """Normalize cumulative reward to 0.0–1.0 based on theoretical max."""
        if self.config is None:
            self._normalized_reward = 0.0
            return
        total_possible = len(self.tickets) + len(self.resolved_tickets)
        if total_possible == 0:
            self._normalized_reward = 0.0
            return
        # Max possible = per-ticket max * count + end bonuses
        max_raw = (total_possible * MAX_REWARD_PER_TICKET
                   + ZERO_BREACH_BONUS + ENTERPRISE_SLA_BONUS + MAX_RESOLUTION_BONUS)
        # Clamp to [0, 1] — negative rewards map to 0
        self._normalized_reward = round(
            max(0.0, min(1.0, self.total_reward / max_raw)), 4
        )

    # ── Time & State Management ────────────────────────────────────────────

    def _advance_time(self) -> None:
        self.current_step += 1
        self.actions_this_step = 0

        # Activate department outages at configured steps
        for dept_name, outage_step in self._dept_outages.items():
            if self.current_step >= outage_step:
                self._dept_available[dept_name] = False

        # Tick SLAs and decay sentiment
        for ticket in list(self.tickets.values()):
            ticket.sla_remaining -= 1

            # Sentiment decays faster for VIP and abusive tickets
            decay = SENTIMENT_DECAY
            if ticket.is_vip:
                decay *= 1.5
            if ticket.is_abusive:
                decay *= 2.0
            ticket.sentiment = round(max(0.0, ticket.sentiment - decay), 2)

            # Sentiment meltdown: customer rage-quits when sentiment hits 0
            if ticket.sentiment <= 0.0 and ticket.status in (
                TicketStatus.OPEN, TicketStatus.IN_PROGRESS
            ):
                ticket.status = TicketStatus.BREACHED
                vip_mult = VIP_MULT if ticket.is_vip else 1.0
                meltdown = round(SENTIMENT_MELTDOWN_PENALTY * URGENCY_MULT[ticket.urgency] * vip_mult, 4)
                self.total_reward = round(self.total_reward + meltdown, 4)
                continue  # skip further checks for this ticket

            # Critical-ignore penalty: P0/P1 tickets sitting untouched (still OPEN) lose reward
            if (ticket.status == TicketStatus.OPEN
                    and ticket.urgency in (TicketUrgency.P0, TicketUrgency.P1)):
                ignore_penalty = round(CRITICAL_IGNORE_PENALTY * URGENCY_MULT[ticket.urgency], 4)
                self.total_reward = round(self.total_reward + ignore_penalty, 4)

            # SLA breach
            if ticket.sla_remaining <= 0 and ticket.status not in (
                TicketStatus.RESOLVED, TicketStatus.ESCALATED, TicketStatus.MERGED, TicketStatus.BREACHED
            ):
                ticket.status = TicketStatus.BREACHED
                # Base breach penalty
                penalty = round(BREACH_PENALTY * URGENCY_MULT[ticket.urgency], 4)
                # Extra penalty: breach-forced escalation (agent had capacity but ignored)
                penalty += BREACH_ESCALATION_PENALTY
                self.total_reward = round(self.total_reward + penalty, 4)

        self._update_normalized_reward()

        vip_ratio = self.config.vip_ratio if self.config else 0.0

        # New arrivals via Poisson process
        if self.config and self.config.arrival_rate > 0 and self.generator:
            arrivals = self.generator.generate_arrivals(
                self.config.arrival_rate, current_step=self.current_step,
                vip_ratio=vip_ratio,
            )
            for t in arrivals:
                self.tickets[t.id] = t
                self._initial_sla[t.id] = t.sla_remaining
                if t.duplicate_of is not None:
                    self._actual_duplicates.add(t.id)

        # Burst events
        if self.config and self.generator:
            for burst in self.config.bursts:
                if burst.step == self.current_step:
                    burst_tickets = self.generator.generate_burst(burst, current_step=self.current_step)
                    for t in burst_tickets:
                        self.tickets[t.id] = t
                        self._initial_sla[t.id] = t.sla_remaining
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

        # Compliance LANDMINE penalty: any landmine ticket that wasn't handled
        # (resolved/escalated/merged) by end of episode costs -5.0 each.
        # This punishes agents that ignore high-stakes compliance signals
        # buried in the noise.
        for t in list(self.tickets.values()) + self.resolved_tickets:
            if getattr(t, "is_landmine", False) and t.status not in (
                TicketStatus.RESOLVED,
                TicketStatus.ESCALATED,
                TicketStatus.MERGED,
            ):
                self.total_reward = round(self.total_reward + LANDMINE_PENALTY, 4)

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
        """
        Grade response quality. If an LLM grader is available, blend its
        judgement with the heuristic score (60% LLM + 40% heuristic). Otherwise
        use pure heuristic. The LLM grading is sampled and bounded by the
        grader's per-episode budget so eval time stays predictable.
        """
        if not self._response_qualities:
            return 0.0

        heuristic_score = sum(self._response_qualities) / len(self._response_qualities)

        # If no LLM grader, return pure heuristic
        if self._llm_grader is None or not self._response_records:
            return round(heuristic_score, 4)

        # Sample up to MAX_LLM_GRADE_SAMPLES responses for LLM grading.
        # Spread the sample across the trajectory (early, middle, late) so we
        # catch templating in different episode phases.
        from .llm_grader import MAX_LLM_GRADE_SAMPLES, LLM_BLEND_WEIGHT

        n_records = len(self._response_records)
        if n_records <= MAX_LLM_GRADE_SAMPLES:
            sample_indices = list(range(n_records))
        else:
            # Even spread across the trajectory
            step = n_records / MAX_LLM_GRADE_SAMPLES
            sample_indices = [int(i * step) for i in range(MAX_LLM_GRADE_SAMPLES)]

        llm_scores: list[float] = []
        for idx in sample_indices:
            subject, description, response = self._response_records[idx]
            score = self._llm_grader.score_response(subject, description, response)
            if score is not None:
                llm_scores.append(score)

        # If every LLM call failed, fall back to heuristic
        if not llm_scores:
            return round(heuristic_score, 4)

        llm_avg = sum(llm_scores) / len(llm_scores)
        blended = LLM_BLEND_WEIGHT * llm_avg + (1.0 - LLM_BLEND_WEIGHT) * heuristic_score
        return round(max(0.0, min(1.0, blended)), 4)

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

    def _grade_classification(self) -> float:
        """Classification accuracy: correct dept routing for escalations + keyword
        coverage for responses. Measures whether the agent understood ticket category."""
        if not self.resolved_tickets:
            return 0.0

        scores: list[float] = []
        for t in self.resolved_tickets:
            if t.status == TicketStatus.ESCALATED:
                # Check if this ticket was routed to the correct department
                # Look through action_log for this ticket's escalation
                correct = False
                for entry in self.action_log:
                    action_data = entry.get("action", {})
                    if (action_data.get("ticket_id") == t.id
                            and action_data.get("action_type") == "escalate"):
                        info = entry.get("info", {})
                        correct = info.get("correct_department", False)
                        break
                scores.append(1.0 if correct else 0.0)
            elif t.status == TicketStatus.RESOLVED:
                # For responses, use the stored quality as a proxy for classification
                # (keyword coverage implies the agent understood the ticket category)
                for entry in self.action_log:
                    action_data = entry.get("action", {})
                    if (action_data.get("ticket_id") == t.id
                            and action_data.get("action_type") == "respond"):
                        info = entry.get("info", {})
                        quality = info.get("quality", 0.0)
                        # Keyword coverage is 40% of quality; extract classification signal
                        # Score > 0.4 means good keyword match → correct classification
                        scores.append(min(1.0, quality / 0.6))
                        break
            elif t.status == TicketStatus.MERGED:
                # Merges: credit if it was an actual duplicate
                for entry in self.action_log:
                    action_data = entry.get("action", {})
                    if (action_data.get("ticket_id") == t.id
                            and action_data.get("action_type") == "merge"):
                        info = entry.get("info", {})
                        scores.append(1.0 if info.get("was_duplicate", False) else 0.3)
                        break

        if not scores:
            return 0.0
        return round(sum(scores) / len(scores), 4)

    def _grade_critical_coverage(self) -> float:
        """Fraction of P0/P1 tickets that were handled (resolved/escalated/merged)
        before or without breaching SLA."""
        all_tickets = list(self.tickets.values()) + self.resolved_tickets
        criticals = [t for t in all_tickets
                     if t.urgency in (TicketUrgency.P0, TicketUrgency.P1)]
        if not criticals:
            return 1.0  # No critical tickets → perfect coverage

        handled = sum(
            1 for t in criticals
            if t.status in (TicketStatus.RESOLVED, TicketStatus.ESCALATED, TicketStatus.MERGED)
        )
        breached_criticals = sum(
            1 for t in criticals
            if t.status == TicketStatus.BREACHED
        )

        # Score: handled / total, with extra penalty for breached criticals
        coverage = handled / len(criticals)
        breach_penalty = breached_criticals / len(criticals) * 0.5
        return round(max(0.0, coverage - breach_penalty), 4)

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

        dept_cap = self.config.department_capacity if self.config else 10
        dept_status = [
            DepartmentStatus(
                name=dept,
                queue_size=self.department_queues.get(dept, 0),
                available=(
                    self._dept_available.get(dept.value, True)
                    and self.department_queues.get(dept, 0) < dept_cap
                ),
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
            normalized_reward=self._normalized_reward,
            tickets_resolved=len(self.resolved_tickets),
            tickets_breached=breached_count,
            tickets_escalated=escalated_count,
        )
