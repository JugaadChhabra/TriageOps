"""Unit tests for the SupportBench environment."""
import json
import importlib
from pathlib import Path

import server.app
importlib.reload(server.app)

from server.environment import CustomerSupportEnv
from server.models import (
    ActionType, CustomerTier, Department, GradeResult, QueueObservation,
    StepResult, SupportAction, TicketStatus, TicketUrgency, TicketView,
)
from server.tickets import TicketGenerator, TEMPLATES
from server.app import TASK_CONFIGS


# ── Helpers ────────────────────────────────────────────────────────────────

def _good_response(ticket):
    kw = " ".join(ticket.resolution_keywords)
    return f"I sincerely apologize for the inconvenience. I have {kw}. Fixed immediately."


# ── Reset / Step / State Cycle ─────────────────────────────────────────────

class TestResetStepState:
    def test_reset_returns_step_result(self):
        env = CustomerSupportEnv()
        r = env.reset(TASK_CONFIGS["ticket_classification"])
        assert isinstance(r.observation, QueueObservation)
        assert r.reward == 0.0
        assert r.done is False
        assert len(r.observation.tickets) == 10

    def test_reset_clears_state(self):
        env = CustomerSupportEnv()
        env.reset(TASK_CONFIGS["ticket_classification"])
        tid = list(env.tickets.keys())[0]
        env.step(SupportAction(action_type=ActionType.RESPOND, ticket_id=tid, response_text="Fixed."))
        assert env.total_reward != 0.0
        env.reset(TASK_CONFIGS["ticket_classification"])
        assert env.total_reward == 0.0
        assert env.current_step == 0

    def test_step_returns_reward_and_updates(self):
        env = CustomerSupportEnv()
        env.reset(TASK_CONFIGS["ticket_classification"])
        tid = list(env.tickets.keys())[0]
        t = env.tickets[tid]
        r = env.step(SupportAction(
            action_type=ActionType.RESPOND, ticket_id=tid,
            response_text=_good_response(t),
        ))
        assert isinstance(r.reward, float)
        assert r.reward > 0
        assert r.observation.tickets_resolved == 1

    def test_state_matches_internal(self):
        env = CustomerSupportEnv()
        env.reset(TASK_CONFIGS["ticket_classification"])
        obs = env.state()
        assert obs.total_reward == env.total_reward
        assert obs.current_step == env.current_step

    def test_step_on_done_episode(self):
        env = CustomerSupportEnv()
        env.reset(TASK_CONFIGS["ticket_classification"])
        env.done = True
        r = env.step(SupportAction(action_type=ActionType.RESPOND, ticket_id="TKT-0001", response_text="hi"))
        assert r.reward == 0.0
        assert r.done is True

    def test_reproducibility(self):
        scores = []
        for _ in range(3):
            env = CustomerSupportEnv()
            env.reset(TASK_CONFIGS["ticket_classification"])
            for tid in list(env.tickets.keys())[:3]:
                t = env.tickets[tid]
                env.step(SupportAction(
                    action_type=ActionType.RESPOND, ticket_id=tid,
                    response_text=_good_response(t),
                ))
            scores.append(env.grade().score)
        assert scores[0] == scores[1] == scores[2]


# ── Graders ────────────────────────────────────────────────────────────────

class TestGraders:
    def test_grade_range(self):
        for name, cfg in TASK_CONFIGS.items():
            for strategy in ["nothing", "respond"]:
                env = CustomerSupportEnv()
                env.reset(cfg)
                if strategy == "respond":
                    while not env.done:
                        active = [t for t in env.tickets.values()
                                  if t.status in (TicketStatus.OPEN, TicketStatus.IN_PROGRESS)]
                        if not active:
                            env.advance_step()
                            continue
                        t = active[0]
                        env.step(SupportAction(
                            action_type=ActionType.RESPOND, ticket_id=t.id,
                            response_text=_good_response(t),
                        ))
                else:
                    while not env.done:
                        env.advance_step()
                g = env.grade()
                assert 0.0 <= g.score <= 1.0, f"{name}/{strategy}: {g.score}"
                for k, v in g.breakdown.items():
                    assert 0.0 <= v <= 1.0, f"{name}/{strategy}/{k}: {v}"

    def test_grade_deterministic(self):
        for name, cfg in TASK_CONFIGS.items():
            scores = []
            for _ in range(3):
                env = CustomerSupportEnv()
                env.reset(cfg)
                ids = list(env.tickets.keys())[:5]
                for tid in ids:
                    if env.done:
                        break
                    t = env.tickets.get(tid)
                    if t is None:
                        continue
                    env.step(SupportAction(
                        action_type=ActionType.RESPOND, ticket_id=tid,
                        response_text=_good_response(t),
                    ))
                scores.append(env.grade().score)
            assert scores[0] == scores[1] == scores[2]

    def test_grade_has_all_components(self):
        env = CustomerSupportEnv()
        env.reset(TASK_CONFIGS["ticket_classification"])
        g = env.grade()
        expected = [
            "resolution_rate", "prioritization", "sla_compliance",
            "response_quality", "duplicate_detection",
            "classification_accuracy", "critical_coverage",
        ]
        for k in expected:
            assert k in g.breakdown

    def test_no_episode_grade(self):
        env = CustomerSupportEnv()
        g = env.grade()
        assert g.score == 0.0


# ── Edge Cases ─────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_invalid_ticket_id(self):
        env = CustomerSupportEnv()
        env.reset(TASK_CONFIGS["ticket_classification"])
        r = env.step(SupportAction(
            action_type=ActionType.RESPOND, ticket_id="NONEXISTENT",
            response_text="hi",
        ))
        assert r.reward < 0

    def test_already_resolved_ticket(self):
        env = CustomerSupportEnv()
        env.reset(TASK_CONFIGS["ticket_classification"])
        tid = list(env.tickets.keys())[0]
        env.step(SupportAction(
            action_type=ActionType.RESPOND, ticket_id=tid,
            response_text=_good_response(env.tickets[tid]) if tid in env.tickets else "hi",
        ))
        r = env.step(SupportAction(
            action_type=ActionType.RESPOND, ticket_id=tid, response_text="again",
        ))
        assert r.reward < 0

    def test_escalate_missing_department(self):
        env = CustomerSupportEnv()
        env.reset(TASK_CONFIGS["ticket_classification"])
        tid = list(env.tickets.keys())[0]
        r = env.step(SupportAction(action_type=ActionType.ESCALATE, ticket_id=tid))
        assert r.reward < 0

    def test_merge_missing_target(self):
        env = CustomerSupportEnv()
        env.reset(TASK_CONFIGS["ticket_classification"])
        tid = list(env.tickets.keys())[0]
        r = env.step(SupportAction(action_type=ActionType.MERGE, ticket_id=tid))
        assert r.reward < 0

    def test_empty_close_penalty(self):
        env = CustomerSupportEnv()
        env.reset(TASK_CONFIGS["ticket_classification"])
        tid = list(env.tickets.keys())[0]
        r = env.step(SupportAction(
            action_type=ActionType.RESPOND, ticket_id=tid, response_text="ok",
        ))
        assert r.reward < 0

    def test_repeat_action_penalty(self):
        env = CustomerSupportEnv()
        env.reset(TASK_CONFIGS["ticket_classification"])
        tid = list(env.tickets.keys())[0]
        env.step(SupportAction(action_type=ActionType.DEFER, ticket_id=tid))
        r = env.step(SupportAction(action_type=ActionType.DEFER, ticket_id=tid))
        assert r.reward < 0
        assert r.info.get("penalty") == "repeat_action"


# ── Tickets ────────────────────────────────────────────────────────────────

class TestTickets:
    def test_template_count(self):
        total = sum(len(v) for v in TEMPLATES.values())
        assert total >= 53

    def test_prior_interactions_generated(self):
        gen = TicketGenerator(seed=42)
        interactions = [gen.generate_ticket().customer.prior_interactions for _ in range(100)]
        assert any(i > 0 for i in interactions)
        assert any(i == 0 for i in interactions)

    def test_vip_generation(self):
        gen = TicketGenerator(seed=42)
        vips = [gen.generate_ticket(vip_ratio=0.5).is_vip for _ in range(100)]
        assert any(vips) and not all(vips)

    def test_abusive_generation(self):
        gen = TicketGenerator(seed=42)
        found = any(gen.generate_ticket().is_abusive for _ in range(200))
        assert found

    def test_poisson_arrivals(self):
        gen = TicketGenerator(seed=42)
        counts = [len(gen.generate_arrivals(rate=2.0)) for _ in range(50)]
        avg = sum(counts) / len(counts)
        assert 1.0 < avg < 3.5


# ── FastAPI Endpoints ──────────────────────────────────────────────────────

class TestEndpoints:
    def _client(self):
        from fastapi.testclient import TestClient
        importlib.reload(server.app)
        from server.app import app
        return TestClient(app)

    def test_health(self):
        assert self._client().get("/health").json()["status"] == "healthy"

    def test_metadata(self):
        r = self._client().get("/metadata")
        assert r.json()["name"] == "SupportBench"

    def test_schema_endpoint(self):
        d = self._client().get("/schema").json()
        assert "action" in d and "observation" in d and "state" in d

    def test_mcp(self):
        assert self._client().post("/mcp", json={}).json()["jsonrpc"] == "2.0"

    def test_reset_empty_body(self):
        r = self._client().post("/reset", json={})
        assert r.status_code == 200
        assert len(r.json()["observation"]["tickets"]) == 10

    def test_reset_all_tasks(self):
        c = self._client()
        for name in ["ticket_classification", "triage_prioritize", "full_resolution"]:
            assert c.post("/reset", json={"task": name}).status_code == 200

    def test_reset_invalid_task(self):
        assert self._client().post("/reset", json={"task": "nope"}).status_code == 404

    def test_step_respond(self):
        c = self._client()
        c.post("/reset", json={})
        tid = c.get("/state").json()["tickets"][0]["id"]
        r = c.post("/step", json={
            "action_type": "respond", "ticket_id": tid,
            "response_text": "I apologize. Fixed immediately.",
        })
        assert r.status_code == 200
        assert "reward" in r.json()

    def test_step_malformed(self):
        c = self._client()
        c.post("/reset", json={})
        assert c.post("/step", json={}).status_code == 422
        assert c.post("/step", json={"action_type": "respond"}).status_code == 422

    def test_grade_returns_all_components(self):
        c = self._client()
        c.post("/reset", json={})
        g = c.get("/grade").json()
        assert 0.0 <= g["score"] <= 1.0
        assert len(g["breakdown"]) == 7
