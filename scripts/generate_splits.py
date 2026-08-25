"""Regenerate the reported HumanEvalComm train/validation/test task-ID files."""

import argparse
import ast
import csv
import random
from pathlib import Path

from omegaconf import OmegaConf


VAL_SEED = 99
VAL_N = 50  # Proportional per-variant rounding produces 52 task IDs.
RARITY_ORDER = [
    "prompt3acp", "prompt2cp", "prompt2ap", "prompt2ac",
    "prompt1p", "prompt1c", "prompt1a",
]
TYPE_MAP = {
    "prompt1a": "1a", "prompt1c": "1c", "prompt1p": "1p",
    "prompt2ac": "2ac", "prompt2ap": "2ap", "prompt2cp": "2cp",
    "prompt3acp": "3acp",
}


def is_valid(value):
    return value is not None and str(value).strip().lower() not in ("none", "")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/HumanEvalComm_v2.csv")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def write_ids(path, task_ids):
    path.write_text("".join(f"{task_id}\n" for task_id in task_ids))


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    variants = list(cfg.data.use_variants)

    with open(args.csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    base_to_variants = {
        row["name"]: {field for field in variants if is_valid(row.get(field))}
        for row in rows
    }
    groups = {}
    for base_id, available in base_to_variants.items():
        group = next((field for field in RARITY_ORDER if field in available), "common")
        groups.setdefault(group, []).append(base_id)

    rng = random.Random(int(cfg.data.seed))
    eval_base_ids = set()
    ratio = int(cfg.data.eval_size) / len(base_to_variants)
    for base_ids in groups.values():
        rng.shuffle(base_ids)
        n_eval = max(1, round(len(base_ids) * ratio))
        eval_base_ids.update(base_ids[:n_eval])

    train_ids = []
    held_out = []
    for row in rows:
        try:
            test_cases = ast.literal_eval(row["test_case"])
        except (ValueError, SyntaxError):
            test_cases = []
        if not is_valid(row["prompt"]) or not test_cases:
            continue
        destination = held_out if row["name"] in eval_base_ids else train_ids
        for field in variants:
            if is_valid(row.get(field)):
                destination.append((f"{row['name']}/{field}", TYPE_MAP[field]))

    by_type = {}
    for task_id, degradation_type in held_out:
        by_type.setdefault(degradation_type, []).append(task_id)
    val_rng = random.Random(VAL_SEED)
    validation_ids = []
    for degradation_type, bucket in sorted(by_type.items()):
        shuffled = list(bucket)
        val_rng.shuffle(shuffled)
        n = max(1, round(len(shuffled) / len(held_out) * VAL_N))
        validation_ids.extend(shuffled[:n])

    validation_set = set(validation_ids)
    test_ids = [task_id for task_id, _ in held_out if task_id not in validation_set]
    train_task_ids = [task_id for task_id, _ in train_ids]

    expected = (302, 52, 417)
    actual = (len(train_task_ids), len(validation_ids), len(test_ids))
    if actual != expected:
        raise RuntimeError(f"Unexpected split sizes {actual}; expected {expected}")

    output_dir = Path(args.output_dir or cfg.paths.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_ids(output_dir / "train_task_ids.txt", train_task_ids)
    write_ids(output_dir / "validation_task_ids.txt", validation_ids)
    write_ids(output_dir / "test_task_ids.txt", test_ids)
    print(f"Wrote {actual[0]} train, {actual[1]} validation, and {actual[2]} test IDs to {output_dir}")


if __name__ == "__main__":
    main()
