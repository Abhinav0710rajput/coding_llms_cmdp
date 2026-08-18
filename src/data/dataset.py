"""
Dataset loading and problem representation for HumanEvalComm.
"""

import ast
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from datasets import load_dataset


DEGRADATION_FIELDS = ["prompt1a", "prompt1c", "prompt1p", "prompt2ac", "prompt2ap", "prompt2cp", "prompt3acp"]


@dataclass
class Problem:
    task_id: str                  # e.g. "HumanEval/42"
    entry_point: str              # function name
    degraded_prompt: str          # what the agent sees
    original_prompt: str          # what the user simulator holds
    test_cases: List[Dict]        # [{"input": ..., "output": ..., "relation": ...}]
    solution: str                 # reference solution (never shown to agent)
    degradation_type: str         # "1a", "1c", "1p", "2ac", "2ap", "2cp"
    source: str = "humaneval"


def _parse_test_cases(raw: str) -> List[Dict]:
    """Parse test_case field from Python literal string."""
    if not raw or raw.strip().lower() == "none":
        return []
    try:
        parsed = ast.literal_eval(raw)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def _is_valid(value: Optional[str]) -> bool:
    return value is not None and str(value).strip().lower() not in ("none", "")


def load_humaneval_comm(
    use_variants: List[str] = None,
    eval_size: int = 100,
    seed: int = 42,
) -> tuple:
    """
    Load HumanEvalComm from HuggingFace and return (train_problems, eval_problems).

    Args:
        use_variants: list of degradation field names to include, e.g. ["prompt1a", "prompt1p"].
                      Defaults to all single-type variants.
        eval_size:    number of base problems to hold out. Variants of a held-out base
                      problem are also held out (split is on base problem level).
        seed:         random seed for the split.

    Returns:
        train_problems, eval_problems — each a List[Problem]
    """
    if use_variants is None:
        use_variants = ["prompt1a", "prompt1c", "prompt1p", "prompt2ac"]

    ds = load_dataset("jie-jw-wu/HumanEvalComm", split="train")
    rows = list(ds)
    rng = random.Random(seed)

    # ── Stratified split on base problem level ───────────────────────────
    # Group base problems by their rarest available variant so that all 7
    # degradation types are guaranteed to appear in both train and eval.

    # 1. For each base problem, find which requested variants it has
    base_to_variants = {}
    for row in rows:
        base_id = row["name"]
        available = set()
        for field_name in use_variants:
            if _is_valid(row.get(field_name)):
                available.add(field_name)
        base_to_variants[base_id] = available

    # 2. Define rarity order (rarest first)
    rarity_order = ["prompt3acp", "prompt2cp", "prompt2ap", "prompt2ac",
                     "prompt1p", "prompt1c", "prompt1a"]

    # 3. Assign each base problem to a group based on its rarest variant
    groups: Dict[str, List[str]] = {}  # group_key -> list of base_ids
    for base_id, available in base_to_variants.items():
        group_key = "common"
        for rare_type in rarity_order:
            if rare_type in available:
                group_key = rare_type
                break
        groups.setdefault(group_key, []).append(base_id)

    # 4. Within each group, split proportionally (eval_size / total)
    total = len(base_to_variants)
    eval_ratio = eval_size / total

    eval_base_ids = set()
    train_base_ids = set()
    for group_key, ids in groups.items():
        rng.shuffle(ids)
        n_eval = max(1, round(len(ids) * eval_ratio))  # at least 1 per group
        eval_base_ids.update(ids[:n_eval])
        train_base_ids.update(ids[n_eval:])

    # Log the split
    print(f"  Stratified split: {len(train_base_ids)} train / {len(eval_base_ids)} eval base problems")
    for group_key, ids in sorted(groups.items()):
        n_train = sum(1 for x in ids if x in train_base_ids)
        n_eval = sum(1 for x in ids if x in eval_base_ids)
        print(f"    {group_key:12s}: {n_train} train, {n_eval} eval (of {len(ids)} total)")

    # Degradation type label from field name
    _type_map = {
        "prompt1a":  "1a",
        "prompt1c":  "1c",
        "prompt1p":  "1p",
        "prompt2ac": "2ac",
        "prompt2ap": "2ap",
        "prompt2cp": "2cp",
        "prompt3acp": "3acp",
    }

    train_problems: List[Problem] = []
    eval_problems: List[Problem] = []

    for row in rows:
        base_id = row["name"]
        original_prompt = row["prompt"]
        test_cases = _parse_test_cases(row["test_case"])
        solution = row["solution"] or ""

        if not _is_valid(original_prompt) or not test_cases:
            continue

        for field_name in use_variants:
            degraded = row.get(field_name)
            if not _is_valid(degraded):
                continue

            problem = Problem(
                task_id=f"{base_id}/{field_name}",
                entry_point=row["entry_point"],
                degraded_prompt=str(degraded),
                original_prompt=str(original_prompt),
                test_cases=test_cases,
                solution=solution,
                degradation_type=_type_map.get(field_name, field_name),
                source="humaneval",
            )

            if base_id in eval_base_ids:
                eval_problems.append(problem)
            else:
                train_problems.append(problem)

    return train_problems, eval_problems


def sample_problems(problems: List[Problem], n: int, rng: random.Random) -> List[Problem]:
    """Sample n problems with replacement (for rollout batches)."""
    return rng.choices(problems, k=n)
