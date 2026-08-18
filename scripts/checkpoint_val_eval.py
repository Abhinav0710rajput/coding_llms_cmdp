"""
Re-run mid-training evals at every iter_* checkpoint with a fixed validation split.

Loads the base model once, then hot-swaps LoRA weights per checkpoint.
Uses the same 50 problems for all checkpoints (seed=VAL_SEED) so results
are directly comparable. Saves which task_ids were used so the final eval
can exclude them.

Usage:
    python scripts/checkpoint_val_eval.py --policy d1_1.0 --budget 1.0
    python scripts/checkpoint_val_eval.py --policy d1_0.5 --budget 0.5

Output:
    logs/checkpoint_val_<policy>.json  — per-checkpoint results + best selection
"""

import argparse
import asyncio
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf

VAL_SEED = 99
VAL_N    = 50   # target; stratified rounding yields 52 problems in practice


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policy",  required=True, help="Checkpoint subdir, e.g. d1_1.0 or d1_0.5")
    p.add_argument("--budget",  type=float, required=True, help="Question budget (d1 value)")
    p.add_argument("--config",  default="configs/default.yaml")
    p.add_argument("--ckpt-base", default="checkpoints/orig_prompt",
                   help="Base dir containing policy subdirs (default: checkpoints/orig_prompt)")
    p.add_argument("--out", default=None,
                   help="Output JSON path (default: logs/checkpoint_val_<policy>.json)")
    p.add_argument("--extra-ckpt-dirs", default=None,
                   help="Comma-separated extra checkpoint dirs to merge (for split training runs)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.model.train_device  = "cuda:0"
    cfg.model.rollout_device = "cuda:0"

    from src.data.dataset import load_humaneval_comm
    from src.models.agent import Agent
    from src.environment.env import ClarificationEnv
    from src.evaluation.evaluator import evaluate_policy

    _, eval_problems = load_humaneval_comm(
        use_variants=list(cfg.data.use_variants),
        eval_size=cfg.data.eval_size,
        seed=cfg.data.seed,
    )
    print(f"Eval pool: {len(eval_problems)} problems")

    # Fixed validation split — stratified by degradation type, same problems every checkpoint
    val_rng = random.Random(VAL_SEED)
    by_type: dict = {}
    for p in eval_problems:
        by_type.setdefault(p.degradation_type, []).append(p)
    val_problems = []
    for dtype, bucket in sorted(by_type.items()):
        shuffled = list(bucket)
        val_rng.shuffle(shuffled)
        n = max(1, round(len(shuffled) / len(eval_problems) * VAL_N))
        val_problems.extend(shuffled[:n])
    val_ids = [p.task_id for p in val_problems]
    print(f"Validation split: {len(val_problems)} problems (seed={VAL_SEED}, stratified by degradation type)")
    for dtype, bucket in sorted(by_type.items()):
        n = sum(1 for p in val_problems if p.degradation_type == dtype)
        print(f"  {dtype:8s}: {n} / {len(bucket)} eval pool")

    ckpt_dir  = os.path.join(args.ckpt_base, args.policy)
    eval_interval = cfg.training.eval_interval

    # Collect checkpoints from primary dir and any extra dirs, deduplicated by iter name
    search_dirs = [ckpt_dir]
    if args.extra_ckpt_dirs:
        search_dirs += [d.strip() for d in args.extra_ckpt_dirs.split(",")]

    ckpt_map = {}  # iter_name -> path
    for d in search_dirs:
        if not os.path.isdir(d):
            print(f"WARNING: checkpoint dir not found: {d}")
            continue
        for name in os.listdir(d):
            if (name.startswith("iter_") and os.path.isdir(os.path.join(d, name))
                    and name[5:].isdigit()):
                if name not in ckpt_map:
                    ckpt_map[name] = os.path.join(d, name)

    all_ckpts = sorted(ckpt_map.keys())
    ckpt_names = [
        d for d in all_ckpts
        if (int(d.split("_")[1]) + 1) % eval_interval == 0
    ]
    print(f"Found {len(all_ckpts)} checkpoints across {len(search_dirs)} dir(s), "
          f"keeping {len(ckpt_names)} at eval_interval={eval_interval}: "
          f"{ckpt_names[0]} → {ckpt_names[-1]}")

    # Load base model once
    print(f"\nLoading base model: {cfg.model.name}")
    agent = Agent(cfg)
    agent.sync_rollout_model()
    agent.rollout_temperature = 0.0

    env = ClarificationEnv(cfg, tokenizer=agent.tokenizer)

    # Resume: load already-completed results from a prior interrupted run
    out_path = args.out or f"logs/checkpoint_val_{args.policy}.json"
    results = []
    done_ckpts = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            prior = json.load(f)
        if prior.get("val_seed") == VAL_SEED and prior.get("val_task_ids") == val_ids:
            results = prior.get("results", [])
            # Backfill checkpoint_path for results written by older code
            for r in results:
                if not r.get("checkpoint_path") and r["checkpoint"] in ckpt_map:
                    r["checkpoint_path"] = ckpt_map[r["checkpoint"]]
            done_ckpts = {r["checkpoint"] for r in results}
            print(f"Resuming: {len(done_ckpts)} checkpoints already evaluated — skipping")
        else:
            print("WARNING: existing output has different val split — starting fresh")

    for ckpt_name in ckpt_names:
        if ckpt_name in done_ckpts:
            print(f"  [skip] {ckpt_name} (already evaluated)")
            continue
        ckpt_path = ckpt_map[ckpt_name]
        print(f"\n{'='*50}")
        print(f"Checkpoint: {ckpt_name}")

        agent.load_lora(ckpt_path)
        agent.sync_rollout_model()

        rng     = random.Random(VAL_SEED)
        metrics = asyncio.run(evaluate_policy(agent, env, val_problems, rng))

        row = {
            "checkpoint":      ckpt_name,
            "checkpoint_path": ckpt_path,
            "pass_at_1":       round(metrics["pass_at_1"],     4),
            "avg_questions":   round(metrics["avg_questions"],  4),
            "satisfies_budget": metrics["avg_questions"] <= args.budget,
        }
        results.append(row)
        print(f"  pass@1={row['pass_at_1']:.4f}  "
              f"avg_qs={row['avg_questions']:.3f}  "
              f"budget_ok={row['satisfies_budget']}")
        # Write incrementally so a killed job can resume
        with open(out_path, "w") as f:
            json.dump({
                "policy": args.policy, "budget": args.budget,
                "val_seed": VAL_SEED, "val_n": len(val_problems),
                "val_task_ids": val_ids, "results": results, "best": None,
            }, f, indent=2)

    # Select best budget-satisfying checkpoint; fall back to lowest avg_qs if none satisfy
    satisfying = [r for r in results if r["satisfies_budget"]]
    if satisfying:
        best = max(satisfying, key=lambda r: r["pass_at_1"])
        best["budget_feasible"] = True
        print(f"\nBest budget-satisfying checkpoint: {best['checkpoint']} "
              f"(pass@1={best['pass_at_1']:.4f}, avg_qs={best['avg_questions']:.3f})")
    else:
        # Budget infeasible — pick checkpoint with lowest avg_qs, break ties by pass@1
        best = min(results, key=lambda r: (r["avg_questions"], -r["pass_at_1"]))
        best["budget_feasible"] = False
        print(f"\nNo checkpoints satisfy the budget constraint.")
        print(f"Fallback (lowest avg_qs): {best['checkpoint']} "
              f"(pass@1={best['pass_at_1']:.4f}, avg_qs={best['avg_questions']:.3f})")

    with open(out_path, "w") as f:
        json.dump({
            "policy":      args.policy,
            "budget":      args.budget,
            "val_seed":    VAL_SEED,
            "val_n":       len(val_problems),
            "val_task_ids": val_ids,
            "results":     results,
            "best":        best,
        }, f, indent=2)
    print(f"\nResults saved to: {out_path}")
    print(f"Run final eval on remaining {len(eval_problems) - VAL_N} problems:")
    print(f"  python scripts/baseline_eval.py --multi-turn --temperature 0.0 --all \\")
    print(f"    --checkpoint {ckpt_dir}/{best['checkpoint'] if best else '<checkpoint>'} \\")
    print(f"    --exclude-file {out_path}")


if __name__ == "__main__":
    main()
