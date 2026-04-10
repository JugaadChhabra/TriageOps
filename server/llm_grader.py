"""
LLM-based response quality grader.

This is the anti-exploit layer. The keyword/Jaccard grader can be gamed by an
agent that crafts responses with the right keywords without actually addressing
the customer's issue. This module adds a real LLM judge that scores how well
each response addresses its ticket on a 0.0–1.0 scale.

Design constraints:
  - **Bounded cost**: only invoked at episode end (`grade()`), and sampled to
    at most MAX_LLM_GRADE_SAMPLES responses per episode. Worst case = 10 calls
    per episode regardless of trajectory length.
  - **Graceful fallback**: if no API key is available OR the LLM call fails,
    returns None and the env falls back to pure heuristic scoring.
  - **Same credentials as inference.py**: reads HF_TOKEN, API_BASE_URL,
    MODEL_NAME from environment variables. Works with any OpenAI-compatible
    endpoint (OpenAI, HF Inference Router, Groq, vLLM, etc).
  - **Reproducible**: temperature=0, prompt is deterministic.
  - **Cached**: same (ticket, response) pair is graded only once.

The final response_quality score blends:
    final = 0.6 * llm_score + 0.4 * heuristic_score   (if LLM available)
    final = 1.0 * heuristic_score                      (fallback)
"""

from __future__ import annotations

import os
import re
from typing import Optional

# Maximum number of LLM grading calls per episode. Caps cost and runtime.
MAX_LLM_GRADE_SAMPLES = 10
LLM_BLEND_WEIGHT = 0.6
GRADER_TIMEOUT_SECONDS = 8.0
GRADER_MAX_TOKENS = 16

# Optional dependency — if openai isn't importable for some reason, we
# silently fall back to heuristic-only scoring.
try:
    from openai import OpenAI

    _OPENAI_AVAILABLE = True
except Exception:  # pragma: no cover
    _OPENAI_AVAILABLE = False
    OpenAI = None  # type: ignore[assignment]


GRADER_SYSTEM_PROMPT = """You are an expert customer support quality reviewer.
You will be shown a customer support ticket and the agent's response.
Score the response on a 0-10 integer scale based on:

  - Does it actually address the customer's specific issue (not generic boilerplate)?
  - Does it state concrete actions taken (not just "we'll look into it")?
  - Does it match the customer's emotional tone (empathy for angry, action for neutral)?
  - Is the routing/judgement correct for the ticket type?

Scoring guide:
  10 = perfect: addresses the exact issue with concrete actions and right tone
   8 = strong: addresses the issue with reasonable actions
   6 = adequate: relevant but generic
   4 = weak: tangentially related, mostly boilerplate
   2 = poor: doesn't address the issue
   0 = wrong: addresses a different issue or harmful

Reply with ONLY the integer score (0-10), nothing else. No explanation.
"""


class LLMGrader:
    """
    Lightweight LLM judge for response quality.

    Usage:
        grader = LLMGrader()
        if grader.is_available():
            score = grader.score_response(ticket_subject, ticket_desc, agent_response)
            # score is in [0.0, 1.0] or None on failure
    """

    def __init__(
        self,
        api_base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.api_base_url = api_base_url or os.getenv(
            "API_BASE_URL", "https://api.openai.com/v1"
        )
        self.model_name = model_name or os.getenv("MODEL_NAME", "gpt-4o-mini")
        # Same credential resolution order as inference.py
        self.api_key = api_key or os.getenv("HF_TOKEN") or os.getenv("API_KEY")
        # Cache: (subject, response_prefix) → score in [0, 1]
        self._cache: dict[tuple[str, str], float] = {}
        # Per-episode call counter (reset by reset_budget)
        self._calls_this_episode: int = 0

        if _OPENAI_AVAILABLE and self.api_key:
            try:
                self._client = OpenAI(  # type: ignore[misc]
                    base_url=self.api_base_url,
                    api_key=self.api_key,
                    timeout=GRADER_TIMEOUT_SECONDS,
                )
                self._available = True
            except Exception:
                self._client = None
                self._available = False
        else:
            self._client = None
            self._available = False

    def is_available(self) -> bool:
        """True if the grader can make LLM calls."""
        return self._available

    def reset_budget(self) -> None:
        """Reset the per-episode call counter (call at start of each episode)."""
        self._calls_this_episode = 0

    def score_response(
        self,
        ticket_subject: str,
        ticket_description: str,
        agent_response: str,
    ) -> Optional[float]:
        """
        Score an agent response on a 0.0–1.0 scale via LLM judgment.

        Returns:
            float in [0.0, 1.0] on success
            None if grader unavailable, budget exhausted, or LLM call failed
        """
        if not self._available or self._client is None:
            return None
        if self._calls_this_episode >= MAX_LLM_GRADE_SAMPLES:
            return None
        if not agent_response or not agent_response.strip():
            return 0.0

        # Cache key: subject + first 80 chars of response (handles repeated
        # template responses without burning calls)
        cache_key = (ticket_subject[:120], agent_response[:120])
        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt = (
            f"TICKET SUBJECT: {ticket_subject}\n"
            f"TICKET DESCRIPTION: {ticket_description[:600]}\n\n"
            f"AGENT RESPONSE: {agent_response[:600]}\n\n"
            f"Score (0-10):"
        )

        try:
            self._calls_this_episode += 1
            completion = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": GRADER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=GRADER_MAX_TOKENS,
            )
            text = (completion.choices[0].message.content or "").strip()
            score = self._parse_score(text)
            if score is not None:
                normalized = max(0.0, min(1.0, score / 10.0))
                self._cache[cache_key] = normalized
                return normalized
            return None
        except Exception:
            # Network error, rate limit, malformed response — fall back silently
            return None

    @staticmethod
    def _parse_score(text: str) -> Optional[float]:
        """Extract the first integer 0-10 from the LLM response."""
        match = re.search(r"\b([0-9]|10)\b", text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None
