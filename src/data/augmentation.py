"""
MBPP augmentation: programmatic degradation of MBPP specs using GPT-4o.
This is a one-time preprocessing step, not run during training.

Usage:
    python -m src.data.augmentation --output data/processed/mbpp_augmented.json
"""

import argparse
import asyncio
import json
import os
from dataclasses import asdict
from typing import List

from openai import AsyncOpenAI

from src.data.dataset import Problem


# ── Degradation prompt templates ─────────────────────────────────────────────

_AMBIGUITY_PROMPT = """\
You are given a Python function docstring. Rewrite it to introduce ambiguity by:
- Replacing specific values (numbers, strings, exact conditions) with vague terms
  like "a number", "some value", "a certain condition".
- Keep the function signature unchanged.
- Return ONLY the rewritten docstring text (the content between the triple quotes).
  Do not include the triple quotes themselves.

Original docstring:
{docstring}
"""

_INCONSISTENCY_PROMPT = """\
You are given a Python function docstring that includes examples.
Rewrite it so the examples contradict the description:
- Keep the description text unchanged.
- Modify the example outputs so they are plausible but wrong
  (e.g., show increment by 2 when description says increment by 1).
- Return ONLY the rewritten docstring text.

Original docstring:
{docstring}
"""

_INCOMPLETENESS_PROMPT = """\
You are given a Python function docstring.
Rewrite it to be incomplete:
- Keep only the first sentence of the description (remove all examples and details).
- Replace specific parameter names with generic ones (e.g., "a value", "some input").
- Return ONLY the shortened docstring text.

Original docstring:
{docstring}
"""


async def _call_gpt4o(
    client: AsyncOpenAI,
    prompt: str,
    semaphore: asyncio.Semaphore,
    model: str = "gpt-4o",
) -> str:
    async with semaphore:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=512,
        )
    return response.choices[0].message.content.strip()


async def degrade_problem(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    task_id: str,
    entry_point: str,
    original_docstring: str,
    full_code: str,
    test_code: str,
) -> List[Problem]:
    """
    Generate 1a, 1c, 1p degraded variants for one MBPP problem.
    Returns a list of Problem objects (one per successful degradation).
    """
    tasks = {
        "1a": _call_gpt4o(client, _AMBIGUITY_PROMPT.format(docstring=original_docstring), semaphore),
        "1c": _call_gpt4o(client, _INCONSISTENCY_PROMPT.format(docstring=original_docstring), semaphore),
        "1p": _call_gpt4o(client, _INCOMPLETENESS_PROMPT.format(docstring=original_docstring), semaphore),
    }
    results = {}
    for key, coro in tasks.items():
        try:
            results[key] = await coro
        except Exception as e:
            print(f"[WARN] {task_id}/{key} failed: {e}")

    problems = []
    signature_line = full_code.split("\n")[0]  # def entry_point(...):

    for degradation_type, degraded_doc in results.items():
        degraded_prompt = f'{signature_line}\n    """{degraded_doc}\n    """'
        problems.append(Problem(
            task_id=f"{task_id}/{degradation_type}",
            entry_point=entry_point,
            degraded_prompt=degraded_prompt,
            original_prompt=full_code,
            test_cases=_parse_mbpp_tests(test_code, entry_point),
            solution=full_code,
            degradation_type=degradation_type,
            source="mbpp",
        ))

    return problems


def _parse_mbpp_tests(test_code: str, entry_point: str) -> list:
    """Convert MBPP assert strings to our test_case format."""
    tests = []
    for line in test_code.strip().split("\n"):
        line = line.strip()
        if line.startswith("assert "):
            tests.append({"input": "", "output": "", "relation": line.replace("assert ", "")})
    return tests


async def augment_mbpp(output_path: str, max_problems: int = 500):
    """
    Load MBPP from HuggingFace, degrade specs, save to JSON.
    Requires OPENAI_API_KEY in environment.
    """
    from datasets import load_dataset

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    semaphore = asyncio.Semaphore(20)  # max concurrent GPT-4o calls

    mbpp = load_dataset("mbpp", split="train")
    rows = list(mbpp)[:max_problems]

    all_problems = []
    batch_tasks = []

    for row in rows:
        task_id = f"MBPP/{row['task_id']}"
        code = row["code"]
        test_list = row.get("test_list", [])
        test_code = "\n".join(test_list)

        # Extract entry point from code
        entry_point = "solution"
        for line in code.split("\n"):
            if line.startswith("def "):
                entry_point = line.split("(")[0].replace("def ", "").strip()
                break

        # Extract docstring
        doc_start = code.find('"""')
        doc_end = code.find('"""', doc_start + 3)
        if doc_start == -1:
            continue
        original_docstring = code[doc_start + 3:doc_end].strip()

        batch_tasks.append(
            degrade_problem(client, semaphore, task_id, entry_point,
                            original_docstring, code, test_code)
        )

    results = await asyncio.gather(*batch_tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, list):
            all_problems.extend(r)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([vars(p) for p in all_problems], f, indent=2)

    print(f"Saved {len(all_problems)} augmented MBPP problems to {output_path}")


def load_mbpp_augmented(path: str) -> List[Problem]:
    """Load previously augmented MBPP problems from JSON."""
    with open(path) as f:
        data = json.load(f)
    return [Problem(**d) for d in data]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/processed/mbpp_augmented.json")
    parser.add_argument("--max_problems", type=int, default=500)
    args = parser.parse_args()
    asyncio.run(augment_mbpp(args.output, args.max_problems))
