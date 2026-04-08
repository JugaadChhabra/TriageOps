"""Ticket generation: templates, customer profiles, Poisson arrivals, bursts."""
import random
from typing import Optional

from .models import (
    BurstConfig,
    Customer,
    CustomerTier,
    Department,
    Ticket,
    TicketCategory,
    TicketStatus,
    TicketUrgency,
)

# ── Templates ──────────────────────────────────────────────────────────────
# Each entry: (subject, description, resolution_keywords, subcategory)

TEMPLATES: dict[TicketCategory, list[tuple[str, str, list[str], str]]] = {
    TicketCategory.BILLING: [
        (
            "Incorrect charge on my account",
            "I was charged ${amount} on {date} but my plan is only ${plan_price}/mo. "
            "Please investigate and refund the overcharge. Account: {email}.",
            ["refund", "credited", "adjusted", "corrected"],
            "overcharge",
        ),
        (
            "Cannot update payment method",
            "I'm trying to update my credit card but the payment page keeps throwing an error. "
            "I need to update before my next billing cycle on {date}.",
            ["payment", "updated", "card", "billing"],
            "payment_update",
        ),
        (
            "Duplicate subscription charge",
            "I see two charges for the same subscription this month — ${amount} each. "
            "Please cancel the duplicate and refund. Ref: {ref_id}.",
            ["refund", "duplicate", "cancelled", "reversed"],
            "duplicate_charge",
        ),
        (
            "Requesting plan downgrade",
            "I'd like to downgrade from {current_plan} to {target_plan}. "
            "Please confirm the prorated amount and effective date.",
            ["downgrade", "prorated", "effective", "confirmed"],
            "plan_change",
        ),
        (
            "Invoice discrepancy",
            "Invoice #{ref_id} shows a line item for '{feature}' at ${amount}, "
            "but we never enabled that feature. Please correct.",
            ["invoice", "corrected", "removed", "credit"],
            "invoice_error",
        ),
        (
            "Tax exemption not applied",
            "Our organization is tax-exempt (EIN: {ref_id}) but we are still being charged tax. "
            "Please apply our exemption retroactively.",
            ["tax", "exempt", "applied", "retroactive"],
            "tax_issue",
        ),
    ],
    TicketCategory.TECHNICAL: [
        (
            "API returning 500 errors",
            "Our integration started getting HTTP 500 on the /v2/data endpoint around {time}. "
            "Request ID: {ref_id}. This is blocking our {feature} pipeline.",
            ["investigated", "fix", "deployed", "resolved", "endpoint"],
            "api_error",
        ),
        (
            "Dashboard loading extremely slowly",
            "The analytics dashboard takes 30+ seconds to load since the last update. "
            "We're on {browser} {version}. Cleared cache, same issue.",
            ["performance", "optimized", "fix", "loading"],
            "performance",
        ),
        (
            "Data export failing",
            "CSV export from the reports section fails with 'timeout' for any date range "
            "longer than 7 days. We need monthly exports for compliance. Account: {email}.",
            ["export", "fix", "timeout", "resolved"],
            "export_failure",
        ),
        (
            "Webhook deliveries failing",
            "Our webhook endpoint at {url} stopped receiving events 2 hours ago. "
            "The endpoint is healthy — confirmed via direct curl. Event ID: {ref_id}.",
            ["webhook", "delivery", "restored", "events"],
            "webhook_issue",
        ),
        (
            "SSO login broken after update",
            "After your platform update on {date}, our SAML SSO integration is rejecting "
            "all login attempts. 200+ employees affected. IdP: {feature}.",
            ["SSO", "SAML", "login", "fix", "restored"],
            "sso_issue",
        ),
        (
            "Mobile app crashes on launch",
            "The iOS app (v{version}) crashes immediately after the splash screen. "
            "iPhone 15, iOS 17. Reinstalled — same issue. Crash ID: {ref_id}.",
            ["crash", "fix", "update", "resolved"],
            "mobile_crash",
        ),
        (
            "Search function returns no results",
            "Full-text search across our workspace has been returning zero results for "
            "any query since {date}. Index might be corrupted.",
            ["search", "index", "rebuilt", "resolved"],
            "search_broken",
        ),
    ],
    TicketCategory.ACCOUNT: [
        (
            "Cannot reset password",
            "I've requested a password reset 3 times but never received the email. "
            "Checked spam. Email: {email}. Need access urgently for a client demo.",
            ["password", "reset", "sent", "access"],
            "password_reset",
        ),
        (
            "Account locked after failed attempts",
            "My account got locked after too many login attempts — I was using an old "
            "password. Username: {email}. Please unlock.",
            ["unlocked", "account", "access", "restored"],
            "account_locked",
        ),
        (
            "Need to transfer account ownership",
            "Our previous admin {name} has left the company. I need to transfer ownership "
            "to {email}. We have the company domain verification.",
            ["transfer", "ownership", "admin", "verified"],
            "ownership_transfer",
        ),
        (
            "Two-factor authentication device lost",
            "I lost my phone and can't complete 2FA. I have my backup codes but they're "
            "not working either. Account: {email}.",
            ["2FA", "reset", "authentication", "access"],
            "2fa_issue",
        ),
        (
            "Delete my account and data",
            "Per GDPR Article 17, I request complete deletion of my account and all "
            "associated data. Email: {email}.",
            ["deleted", "GDPR", "confirmed", "data", "removal"],
            "account_deletion",
        ),
        (
            "Merge duplicate accounts",
            "I accidentally created two accounts — {email} and a second one. "
            "Please merge all data into the primary account.",
            ["merged", "accounts", "consolidated", "primary"],
            "account_merge",
        ),
    ],
    TicketCategory.FEATURE_REQUEST: [
        (
            "Request: Bulk export to PDF",
            "We need the ability to export multiple reports as a single PDF bundle. "
            "Currently we have to export one at a time. This would save our team hours weekly.",
            ["noted", "roadmap", "feature", "request", "consider"],
            "export_feature",
        ),
        (
            "Request: Dark mode support",
            "Please add dark mode. Our team works late shifts and the bright UI causes "
            "eye strain. Several competitors already support this.",
            ["noted", "dark mode", "roadmap", "planned"],
            "ui_feature",
        ),
        (
            "Request: API rate limit dashboard",
            "We need a self-service dashboard showing our current API rate limit usage "
            "and remaining quota. Right now we have to contact support every time.",
            ["noted", "dashboard", "rate limit", "roadmap"],
            "api_feature",
        ),
        (
            "Request: Custom webhook filters",
            "We receive thousands of webhook events daily but only care about a subset. "
            "Please add filtering/subscription options by event type.",
            ["noted", "webhook", "filter", "roadmap"],
            "webhook_feature",
        ),
        (
            "Request: Team-level permissions",
            "We need granular permissions at the team level, not just org-wide. "
            "Our compliance team should not see engineering dashboards.",
            ["noted", "permissions", "team", "roadmap", "consider"],
            "permissions_feature",
        ),
        (
            "Request: Scheduled reports",
            "Would love the ability to schedule automated reports to be emailed to "
            "stakeholders weekly. Currently this is a manual process.",
            ["noted", "scheduled", "reports", "roadmap"],
            "reports_feature",
        ),
        (
            "Request: Multi-language support",
            "Our customer base is 40% non-English speakers. We need localization for "
            "at least Spanish, French, and German in the dashboard and email templates.",
            ["noted", "localization", "language", "roadmap", "i18n"],
            "localization_feature",
        ),
    ],
    TicketCategory.OUTAGE: [
        (
            "URGENT: Complete service outage",
            "Your entire platform is down. We're getting 503 on all endpoints. "
            "This is impacting our production systems. Started at {time}. "
            "We have {count} customers unable to access our product.",
            ["outage", "restored", "incident", "resolved", "RCA"],
            "full_outage",
        ),
        (
            "CRITICAL: Database connection failures",
            "We're seeing intermittent database connection timeouts across all our "
            "API calls. Error rate jumped from 0.1% to 45% at {time}.",
            ["database", "connection", "resolved", "restored"],
            "db_outage",
        ),
        (
            "URGENT: Payment processing down",
            "Credit card processing is completely broken. All transactions failing with "
            "'gateway timeout'. We're losing ${amount}/hr in revenue.",
            ["payment", "processing", "restored", "resolved"],
            "payment_outage",
        ),
        (
            "CRITICAL: Data sync stopped",
            "Real-time data sync between our systems stopped at {time}. "
            "Over {count} records are now out of sync. This is causing customer-visible errors.",
            ["sync", "restored", "data", "resolved", "backfill"],
            "sync_outage",
        ),
        (
            "URGENT: Authentication service down",
            "No one at our company can log in. The auth service returns 502 for all "
            "requests. {count} employees locked out.",
            ["auth", "login", "restored", "resolved"],
            "auth_outage",
        ),
        (
            "CRITICAL: CDN returning stale content",
            "Your CDN is serving content from 2 days ago. All our latest deployments "
            "are invisible to end users. Cache purge requests also failing.",
            ["CDN", "cache", "purged", "resolved", "fresh"],
            "cdn_outage",
        ),
    ],
    TicketCategory.SECURITY: [
        (
            "Suspicious login from unknown location",
            "I received an alert about a login from {location} at {time}. "
            "I was not traveling. Please verify if my account is compromised. Email: {email}.",
            ["security", "reviewed", "password", "compromised", "secured"],
            "suspicious_login",
        ),
        (
            "Potential data breach notification",
            "We found our internal API key exposed in a public GitHub repository. "
            "Key prefix: {ref_id}. Please revoke immediately and audit access logs.",
            ["revoked", "key", "audit", "rotated", "secured"],
            "key_exposure",
        ),
        (
            "Vulnerability in your API",
            "Our security team discovered an IDOR vulnerability on your /users/{id} endpoint. "
            "We can access other customers' data by changing the ID. This is critical.",
            ["vulnerability", "patched", "fix", "security", "acknowledged"],
            "vulnerability_report",
        ),
        (
            "Request for SOC 2 compliance report",
            "Our compliance team needs your latest SOC 2 Type II report for our "
            "vendor risk assessment. Due by {date}.",
            ["SOC 2", "report", "compliance", "provided", "sent"],
            "compliance_request",
        ),
        (
            "Unauthorized API access detected",
            "Our logs show API calls from IPs we don't recognize making requests "
            "to our workspace. IPs: {ref_id}. Please investigate.",
            ["investigated", "blocked", "IP", "secured", "audit"],
            "unauthorized_access",
        ),
        (
            "Data encryption at rest question",
            "For our compliance audit, we need confirmation that all customer data "
            "is encrypted at rest with AES-256 or equivalent. Can you provide documentation?",
            ["encryption", "AES-256", "confirmed", "documentation", "compliance"],
            "encryption_inquiry",
        ),
    ],
    TicketCategory.COMPLIANCE: [
        (
            "GDPR data subject access request",
            "Under GDPR Article 15, I request a copy of all personal data you hold "
            "about me. Email: {email}. Please respond within 30 days.",
            ["GDPR", "data", "export", "provided", "request"],
            "dsar",
        ),
        (
            "CCPA opt-out request",
            "Per CCPA, I am opting out of the sale of my personal information. "
            "Please process immediately. Account: {email}.",
            ["CCPA", "opt-out", "processed", "confirmed"],
            "ccpa_optout",
        ),
        (
            "Data processing agreement needed",
            "We need a signed DPA before we can proceed with onboarding. "
            "Our legal team requires GDPR-compliant data processing terms. Contact: {email}.",
            ["DPA", "signed", "agreement", "processing", "sent"],
            "dpa_request",
        ),
        (
            "Data residency requirements",
            "Our company policy requires all data to be stored within the EU. "
            "Can you confirm your data center locations and provide a data residency guarantee?",
            ["data", "residency", "EU", "confirmed", "location"],
            "data_residency",
        ),
        (
            "Audit log access request",
            "For our internal audit, we need complete access logs for our organization "
            "for the past 90 days. Account: {email}.",
            ["audit", "logs", "provided", "access", "exported"],
            "audit_logs",
        ),
        (
            "HIPAA BAA request",
            "We're a healthcare company and need a signed Business Associate Agreement "
            "before storing any PHI on your platform. Contact: {email}.",
            ["HIPAA", "BAA", "signed", "agreement", "compliance"],
            "hipaa_baa",
        ),
    ],
    TicketCategory.GENERAL: [
        (
            "How to integrate with Slack?",
            "We'd like to set up the Slack integration but can't find documentation. "
            "Specifically, we need to pipe alerts from your platform into our #ops channel.",
            ["Slack", "integration", "documentation", "guide", "setup"],
            "integration_help",
        ),
        (
            "Onboarding help for new team",
            "We just added 15 new team members. What's the best way to onboard them? "
            "Do you have training materials or a getting-started guide?",
            ["onboarding", "guide", "training", "setup", "welcome"],
            "onboarding",
        ),
        (
            "Feedback on recent UI changes",
            "The new navigation is confusing. The settings menu moved and several of "
            "my team members can't find the export function anymore.",
            ["feedback", "noted", "navigation", "UI", "thank"],
            "ui_feedback",
        ),
        (
            "Question about pricing tiers",
            "Can you explain the difference between Pro and Enterprise plans? "
            "Specifically around SSO, audit logs, and SLA guarantees.",
            ["pricing", "plan", "features", "comparison", "explained"],
            "pricing_question",
        ),
        (
            "Partnership inquiry",
            "We're interested in a technology partnership. We build {feature} tools "
            "and think there's a great integration opportunity. Who should we contact?",
            ["partnership", "forwarded", "contact", "team"],
            "partnership",
        ),
        (
            "Cancellation request",
            "We've decided to move to a different solution. Please cancel our account "
            "effective end of the current billing period. Account: {email}.",
            ["cancelled", "confirmation", "effective", "billing"],
            "cancellation",
        ),
        (
            "THIS IS UNACCEPTABLE",
            "I have been waiting FOUR DAYS for a response to my original ticket. "
            "This is the worst support I have ever experienced. I am filing a complaint "
            "with the BBB if this is not resolved TODAY. Account: {email}.",
            ["apologize", "understand", "resolve", "escalate", "priority"],
            "abusive_complaint",
        ),
        (
            "Multiple issues need resolution",
            "I have three problems: (1) My invoice #{ref_id} is wrong — charged ${amount} "
            "instead of ${plan_price}. (2) The {feature} dashboard has been broken since {date}. "
            "(3) My teammate {name} still can't log in after your last update. Please help.",
            ["invoice", "dashboard", "login", "resolved", "fixed"],
            "multi_issue",
        ),
        (
            "Third time contacting support",
            "This is my THIRD ticket about the same issue. Previous refs: {ref_id}. "
            "Nobody has followed up. I was promised a callback on {date} that never happened. "
            "I need a supervisor or I'm cancelling our {current_plan} plan.",
            ["apologize", "supervisor", "escalated", "priority", "follow up"],
            "repeat_caller",
        ),
    ],
}

# Category → Department mapping
CATEGORY_DEPARTMENT: dict[TicketCategory, Department] = {
    TicketCategory.BILLING: Department.BILLING,
    TicketCategory.TECHNICAL: Department.ENGINEERING,
    TicketCategory.ACCOUNT: Department.ACCOUNT_MANAGEMENT,
    TicketCategory.FEATURE_REQUEST: Department.ENGINEERING,
    TicketCategory.OUTAGE: Department.ENGINEERING,
    TicketCategory.SECURITY: Department.SECURITY,
    TicketCategory.COMPLIANCE: Department.SECURITY,
    TicketCategory.GENERAL: Department.GENERAL_SUPPORT,
}

# Category → urgency weight distributions
URGENCY_WEIGHTS: dict[TicketCategory, dict[TicketUrgency, float]] = {
    TicketCategory.BILLING: {TicketUrgency.P0: 0.05, TicketUrgency.P1: 0.25, TicketUrgency.P2: 0.45, TicketUrgency.P3: 0.25},
    TicketCategory.TECHNICAL: {TicketUrgency.P0: 0.10, TicketUrgency.P1: 0.35, TicketUrgency.P2: 0.35, TicketUrgency.P3: 0.20},
    TicketCategory.ACCOUNT: {TicketUrgency.P0: 0.05, TicketUrgency.P1: 0.20, TicketUrgency.P2: 0.45, TicketUrgency.P3: 0.30},
    TicketCategory.FEATURE_REQUEST: {TicketUrgency.P0: 0.0, TicketUrgency.P1: 0.05, TicketUrgency.P2: 0.25, TicketUrgency.P3: 0.70},
    TicketCategory.OUTAGE: {TicketUrgency.P0: 0.60, TicketUrgency.P1: 0.30, TicketUrgency.P2: 0.08, TicketUrgency.P3: 0.02},
    TicketCategory.SECURITY: {TicketUrgency.P0: 0.30, TicketUrgency.P1: 0.40, TicketUrgency.P2: 0.20, TicketUrgency.P3: 0.10},
    TicketCategory.COMPLIANCE: {TicketUrgency.P0: 0.10, TicketUrgency.P1: 0.30, TicketUrgency.P2: 0.40, TicketUrgency.P3: 0.20},
    TicketCategory.GENERAL: {TicketUrgency.P0: 0.0, TicketUrgency.P1: 0.10, TicketUrgency.P2: 0.40, TicketUrgency.P3: 0.50},
}

# Customer name pools
FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Quinn", "Avery",
    "Cameron", "Reese", "Dakota", "Skyler", "Finley", "Rowan", "Sage", "Blake",
    "Hayden", "Parker", "Drew", "Emery", "Jamie", "Robin", "Shannon", "Kerry",
    "Priya", "Wei", "Yuki", "Carlos", "Fatima", "Olga", "Marco", "Aisha",
]
LAST_NAMES = [
    "Chen", "Patel", "Kim", "Müller", "Silva", "Johnson", "Tanaka", "Garcia",
    "Nguyen", "Anderson", "Singh", "Williams", "Okafor", "Petrov", "Martinez",
    "Brown", "Yamamoto", "Ali", "Schmidt", "Kowalski", "Lee", "Wilson", "Clark",
    "Lopez", "Walker", "Hall", "Young", "King", "Wright", "Adams", "Baker", "Hill",
]


class TicketGenerator:
    """Generates tickets with realistic profiles and stochastic properties."""

    def __init__(self, seed: int = 42, sla_steps: Optional[dict[str, int]] = None):
        self.rng = random.Random(seed)
        self.ticket_counter = 0
        self.sla_steps = sla_steps or {"p0": 3, "p1": 5, "p2": 8, "p3": 12}

    def _next_id(self) -> str:
        self.ticket_counter += 1
        return f"TKT-{self.ticket_counter:04d}"

    def _random_customer(self) -> Customer:
        tier_roll = self.rng.random()
        if tier_roll < 0.5:
            tier = CustomerTier.FREE
            ltv = round(self.rng.uniform(0, 100), 2)
        elif tier_roll < 0.85:
            tier = CustomerTier.PRO
            ltv = round(self.rng.uniform(500, 5000), 2)
        else:
            tier = CustomerTier.ENTERPRISE
            ltv = round(self.rng.uniform(10000, 200000), 2)

        return Customer(
            name=f"{self.rng.choice(FIRST_NAMES)} {self.rng.choice(LAST_NAMES)}",
            tier=tier,
            ltv=ltv,
            churn_risk=round(self.rng.uniform(0.05, 0.80), 2),
            satisfaction=round(self.rng.uniform(0.3, 0.9), 2),
            prior_interactions=self.rng.choices([0, 1, 2, 3, 5, 8], weights=[30, 25, 20, 12, 8, 5], k=1)[0],
        )

    def _pick_urgency(self, category: TicketCategory) -> TicketUrgency:
        weights = URGENCY_WEIGHTS[category]
        return self.rng.choices(
            list(weights.keys()), weights=list(weights.values()), k=1
        )[0]

    def _fill_template(self, text: str) -> str:
        placeholders = {
            "{amount}": str(self.rng.choice([9.99, 19.99, 29.99, 49.99, 99.99, 149.99, 499.99])),
            "{date}": f"2026-0{self.rng.randint(1,4)}-{self.rng.randint(1,28):02d}",
            "{time}": f"{self.rng.randint(0,23):02d}:{self.rng.randint(0,59):02d} UTC",
            "{email}": f"user{self.rng.randint(100,999)}@company.com",
            "{ref_id}": f"REF-{self.rng.randint(10000,99999)}",
            "{feature}": self.rng.choice(["analytics", "reporting", "dashboard", "API", "notifications", "SSO"]),
            "{browser}": self.rng.choice(["Chrome", "Firefox", "Safari", "Edge"]),
            "{version}": f"{self.rng.randint(1,5)}.{self.rng.randint(0,9)}.{self.rng.randint(0,20)}",
            "{url}": f"https://hooks.company{self.rng.randint(1,50)}.com/webhook",
            "{location}": self.rng.choice(["Lagos, Nigeria", "Shenzhen, China", "São Paulo, Brazil", "Moscow, Russia", "Bucharest, Romania"]),
            "{name}": f"{self.rng.choice(FIRST_NAMES)} {self.rng.choice(LAST_NAMES)}",
            "{count}": str(self.rng.choice([50, 100, 200, 500, 1000, 5000])),
            "{current_plan}": self.rng.choice(["Enterprise", "Pro", "Business"]),
            "{target_plan}": self.rng.choice(["Pro", "Starter", "Free"]),
            "{plan_price}": str(self.rng.choice([29, 49, 99, 199])),
            "{id}": str(self.rng.randint(1000, 9999)),
        }
        for key, val in placeholders.items():
            text = text.replace(key, val)
        return text

    def generate_ticket(
        self,
        category: Optional[TicketCategory] = None,
        urgency: Optional[TicketUrgency] = None,
        current_step: int = 0,
        duplicate_of: Optional[str] = None,
        force_customer_tier: Optional[CustomerTier] = None,
        vip_ratio: float = 0.0,
    ) -> Ticket:
        if category is None:
            category = self.rng.choice(list(TicketCategory))

        if urgency is None:
            urgency = self._pick_urgency(category)

        templates = TEMPLATES[category]
        subject_t, desc_t, keywords, subcategory = self.rng.choice(templates)

        subject = self._fill_template(subject_t)
        description = self._fill_template(desc_t)

        customer = self._random_customer()
        if force_customer_tier is not None:
            customer.tier = force_customer_tier
            if force_customer_tier == CustomerTier.ENTERPRISE:
                customer.ltv = round(self.rng.uniform(10000, 200000), 2)

        sla = self.sla_steps.get(urgency.value, 8)
        sentiment = round(max(0.1, min(0.9, 0.6 - 0.15 * (["p3", "p2", "p1", "p0"].index(urgency.value)) + self.rng.gauss(0, 0.1))), 2)

        # VIP flag: carries 3x reward weight
        is_vip = self.rng.random() < vip_ratio

        # Abusive flag: based on subcategory or very low sentiment
        is_abusive = subcategory in ("abusive_complaint", "repeat_caller") or sentiment <= 0.15

        return Ticket(
            id=self._next_id(),
            subject=subject,
            description=description,
            customer=customer,
            category=category,
            urgency=urgency,
            required_department=CATEGORY_DEPARTMENT[category],
            resolution_keywords=keywords,
            status=TicketStatus.OPEN,
            sla_remaining=sla,
            created_step=current_step,
            sentiment=sentiment,
            duplicate_of=duplicate_of,
            subcategory=subcategory,
            is_vip=is_vip,
            is_abusive=is_abusive,
        )

    def generate_batch(self, count: int, current_step: int = 0, vip_ratio: float = 0.0) -> list[Ticket]:
        return [self.generate_ticket(current_step=current_step, vip_ratio=vip_ratio) for _ in range(count)]

    def generate_arrivals(self, rate: float, current_step: int = 0, vip_ratio: float = 0.0) -> list[Ticket]:
        """Poisson-distributed new ticket arrivals."""
        n = self.rng.poisson(rate) if hasattr(self.rng, "poisson") else self._poisson(rate)
        return [self.generate_ticket(current_step=current_step, vip_ratio=vip_ratio) for _ in range(n)]

    def _poisson(self, lam: float) -> int:
        """Simple Poisson sample using inverse transform."""
        import math
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while True:
            k += 1
            p *= self.rng.random()
            if p < L:
                return k - 1

    def generate_burst(
        self, config: BurstConfig, current_step: int = 0
    ) -> list[Ticket]:
        """Generate a burst of tickets, with configurable duplicate ratio."""
        tickets: list[Ticket] = []

        # Primary outage ticket
        primary = self.generate_ticket(
            category=TicketCategory.OUTAGE,
            urgency=TicketUrgency.P0,
            current_step=current_step,
        )
        tickets.append(primary)

        dup_count = int((config.count - 1) * config.duplicate_ratio)
        normal_count = config.count - 1 - dup_count

        # Duplicate tickets referencing primary
        for _ in range(dup_count):
            dup = self.generate_ticket(
                category=TicketCategory.OUTAGE,
                urgency=self.rng.choice([TicketUrgency.P0, TicketUrgency.P1]),
                current_step=current_step,
                duplicate_of=primary.id,
            )
            tickets.append(dup)

        # Non-duplicate tickets (noise, including one compliance buried in)
        for i in range(normal_count):
            if i == 0:
                # Bury a compliance ticket in the noise
                t = self.generate_ticket(
                    category=TicketCategory.COMPLIANCE,
                    urgency=TicketUrgency.P1,
                    current_step=current_step,
                )
            else:
                t = self.generate_ticket(current_step=current_step)
            tickets.append(t)

        return tickets
