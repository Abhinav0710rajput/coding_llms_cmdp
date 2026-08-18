"""
User simulator: GPT-4o-mini that holds the full problem spec and answers
the agent's clarifying questions.

Multi-question handling (controlled by multi_question_mode in config):
  "count"    → simulator counts atomic questions in the agent's message
               and returns QUESTION_COUNT: N at the end of its response.
               The environment uses this N as cost_q for that turn.
  "truncate" → only the first question (up to the first '?') is passed
               to the simulator. cost_q is always 1 per [ASK] turn.
"""

import asyncio
import os
import re
from typing import Tuple

from openai import AsyncOpenAI, RateLimitError, APITimeoutError, APIConnectionError


# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT_COUNT = """\
You are a helpful assistant who holds the complete specification for a coding problem.

Full specification:
{original_prompt}

Note: The agent may refer to the function by a different name (e.g., "candidate"). \
Treat any function name the agent uses as referring to the function described above.

Rules:
1. Answer ONLY the specific question(s) the agent asks. Do not volunteer extra information.
2. Do not reveal test cases or the full solution.
3. Keep your answer brief and factual.
4. After your answer, on a new line write EXACTLY:
   QUESTION_COUNT: N
   where N is the number of distinct atomic questions you identified in the agent's message.

Counting rules for N:
- Each distinct piece of information being requested = 1.
- "What is X and what is Y?" = 2.
- "What is X?" = 1.
- Conjunctions like "and", "also", "additionally" between separate requests = separate questions.
- "Describe the full behavior of X" where X is one concept = 1.
"""

_SYSTEM_PROMPT_TRUNCATE = """\
You are a helpful assistant who holds the complete specification for a coding problem.

Full specification:
{original_prompt}

Note: The agent may refer to the function by a different name (e.g., "candidate"). \
Treat any function name the agent uses as referring to the function described above.

Rules:
1. Answer ONLY the single question below. Do not volunteer extra information.
2. Do not reveal test cases or the full solution.
3. Keep your answer brief and factual.
"""


# ── Simulator class ───────────────────────────────────────────────────────────

class UserSimulator:
    """
    Async wrapper around GPT-4o-mini for answering agent clarifying questions.
    """

    def __init__(self, cfg):
        self.model = cfg.user_simulator.model
        self.temperature = cfg.user_simulator.temperature
        self.max_tokens = cfg.user_simulator.max_tokens
        self.multi_question_mode = cfg.environment.multi_question_mode
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set. Cannot initialize UserSimulator.")
        self.client = AsyncOpenAI()
        self._max_concurrent = cfg.user_simulator.max_concurrent_api
        self._semaphore = None

    def _get_semaphore(self):
        """Lazily create semaphore bound to the current event loop."""
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._semaphore._loop is not loop:
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
        return self._semaphore

    async def answer(
        self,
        question: str,
        original_prompt: str,
    ) -> Tuple[str, int]:
        """
        Answer the agent's question given the full original spec.

        Args:
            question:        The agent's raw [ASK] text (after stripping the [ASK] tag).
            original_prompt: The full, un-degraded problem spec.

        Returns:
            (answer_text, question_count)
            - answer_text:    the simulator's response shown to the agent
            - question_count: number of atomic questions detected (for cost_q)
        """
        if self.multi_question_mode == "truncate":
            return await self._answer_truncate(question, original_prompt)
        else:
            return await self._answer_count(question, original_prompt)

    async def _answer_count(self, question: str, original_prompt: str) -> Tuple[str, int]:
        system = _SYSTEM_PROMPT_COUNT.format(original_prompt=original_prompt)
        last_exc = None
        for attempt in range(5):
            try:
                async with self._get_semaphore():
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user",   "content": question},
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        timeout=30.0,
                    )
                raw = response.choices[0].message.content.strip()
                return _parse_count_response(raw)
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                last_exc = e
                wait = 2 ** attempt
                print(f"  [API] {type(e).__name__}, retry {attempt + 1}/5 in {wait}s")
                await asyncio.sleep(wait)
        raise last_exc

    async def _answer_truncate(self, question: str, original_prompt: str) -> Tuple[str, int]:
        truncated = _truncate_to_first_question(question)
        system = _SYSTEM_PROMPT_TRUNCATE.format(original_prompt=original_prompt)
        last_exc = None
        for attempt in range(5):
            try:
                async with self._get_semaphore():
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user",   "content": truncated},
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        timeout=30.0,
                    )
                answer_text = response.choices[0].message.content.strip()
                return answer_text, 1  # always cost 1 in truncate mode
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                last_exc = e
                wait = 2 ** attempt
                print(f"  [API] {type(e).__name__}, retry {attempt + 1}/5 in {wait}s")
                await asyncio.sleep(wait)
        raise last_exc


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_count_response(raw: str) -> Tuple[str, int]:
    """
    Extract answer text and QUESTION_COUNT: N from the simulator's response.
    Falls back to count=1 if the tag is missing or malformed.
    """
    pattern = re.compile(r"QUESTION_COUNT:\s*(\d+)", re.IGNORECASE)
    match = pattern.search(raw)

    if match:
        count = max(1, int(match.group(1)))
        answer_text = raw[:match.start()].strip()
    else:
        # Fallback: default to 1 question (counting ? in the response is unreliable
        # since it counts question marks in the answer, not the agent's question)
        count = 1
        answer_text = raw.strip()

    return answer_text, count


def _truncate_to_first_question(text: str) -> str:
    """
    Keep only the first question from the agent's [ASK] text.
    Cuts at the first '?' and trims trailing partial sentences.
    """
    idx = text.find("?")
    if idx == -1:
        return text.strip()
    return text[: idx + 1].strip()
