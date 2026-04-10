"""Unit tests for the TriageOps environment."""
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
from server.triage_environment import get_task_configs

# Tasks are now loaded by the OpenEnv-spec wrapper (triage_environment.py)
TASK_CONFIGS = get_task_configs()


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


# ── FastAPI Endpoints (HTTP, stateless per-request via create_app) ────────
#
# Note: HTTP /reset and /step are STATELESS in the OpenEnv create_app() factory.
# Each request gets a fresh Environment instance via the factory, so multi-call
# episodes must use the WebSocket /ws endpoint via the EnvClient subclass.
# These tests just verify the HTTP surface is up and shape-correct.

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
        # OpenEnv create_app() returns EnvironmentMetadata; name follows env_name="triageops"
        assert r.json()["name"] == "triageops"
        assert "description" in r.json()

    def test_schema_endpoint(self):
        d = self._client().get("/schema").json()
        assert "action" in d and "observation" in d and "state" in d

    def test_mcp(self):
        assert self._client().post("/mcp", json={}).json()["jsonrpc"] == "2.0"

    def test_reset_empty_body(self):
        r = self._client().post("/reset", json={})
        assert r.status_code == 200
        # OpenEnv create_app wraps observation under "observation" key
        body = r.json()
        assert "observation" in body
        assert len(body["observation"]["tickets"]) == 10

    def test_reset_all_tasks(self):
        c = self._client()
        for name in ["ticket_classification", "triage_prioritize", "full_resolution"]:
            assert c.post("/reset", json={"task": name}).status_code == 200

    def test_reset_invalid_task(self):
        """Invalid task name → ValueError raised inside the env.
        OpenEnv create_app surfaces this as a 500 / lets it propagate from the
        thread-pool — we just verify the env *itself* rejects bad task names."""
        import pytest
        from server.triage_environment import TriageEnvironment

        env = TriageEnvironment()
        with pytest.raises(ValueError):
            env.reset(task="definitely_not_a_task")

    def test_step_via_websocket_client(self):
        """End-to-end /reset → /step → /state via the typed WebSocket EnvClient.
        This is the canonical test path for the OpenEnv contract."""
        import asyncio
        import threading
        import time as _time

        import uvicorn

        from client import TriageOpsEnv
        from models import TriageAction, ActionType

        importlib.reload(server.app)
        from server.app import app

        class _Server(uvicorn.Server):
            def install_signal_handlers(self): pass

        config = uvicorn.Config(app, host="127.0.0.1", port=8801, log_level="error")
        srv = _Server(config)
        thread = threading.Thread(target=srv.run, daemon=True)
        thread.start()
        _time.sleep(1.0)

        async def go():
            async with TriageOpsEnv(base_url="http://127.0.0.1:8801") as env:
                result = await env.reset()
                assert len(result.observation.tickets) > 0
                tid = result.observation.tickets[0].id
                step_result = await env.step(TriageAction(
                    action_type=ActionType.RESPOND,
                    ticket_id=tid,
                    response_text="I sincerely apologize. I have resolved this immediately.",
                ))
                assert step_result.reward is not None
                state = await env.state()
                assert state.step_count >= 1

        asyncio.run(go())
        srv.should_exit = True

    def test_step_malformed(self):
        c = self._client()
        c.post("/reset", json={})
        # OpenEnv create_app wraps action under "action" key — bare body fails validation
        r = c.post("/step", json={})
        assert r.status_code == 422

    def test_state_endpoint(self):
        c = self._client()
        c.post("/reset", json={})
        r = c.get("/state")
        assert r.status_code == 200
        assert "step_count" in r.json()
