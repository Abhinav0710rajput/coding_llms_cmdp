"""
Evaluation entry point.

Loads a trained checkpoint and evaluates on the held-out test set.
Can evaluate a single checkpoint or sweep over all d₁ settings to
build the Pareto frontier.

Examples:
    # Evaluate one checkpoint
    python scripts/evaluate.py --checkpoint checkpoints/d1_0/final

    # Pareto sweep over best-eval checkpoints (default)
    python scripts/evaluate.py --sweep --output_dir outputs/pareto

    # Pareto sweep over final checkpoints
    python scripts/evaluate.py --sweep --sweep_mode final --output_dir outputs/pareto_final
"""

import argparse
import asyncio
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf

from src.data.dataset import load_humaneval_comm
from src.environment.env import ClarificationEnv
from src.models.agent import Agent
from src.models.value_heads import ThreeHeads
from src.evaluation.evaluator import (
    evaluate_policy,
    compute_pareto_frontier,
    plot_pareto_frontier,
    print_eval_table,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained clarification agent")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to a single checkpoint directory (LoRA weights + value heads)",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Sweep all d₁ checkpoints in checkpoints/ and build Pareto frontier",
    )
    parser.add_argument(
        "--sweep_mode", choices=["best", "final"], default="best",
        help="Which checkpoint to use per d₁: 'best' (best eval reward) or 'final' (end of training)",
    )
    parser.add_argument("--output_dir", default="outputs/eval")
    parser.add_argument("--d1", type=int, default=None, help="d₁ value for labelling")
    return parser.parse_args()


def eval_one(cfg, checkpoint_dir: str, eval_problems, d1: int = None) -> dict:
    """Evaluate a single checkpoint."""
    agent = Agent(cfg)
    agent.load_lora(checkpoint_dir)

    vh_path = os.path.join(checkpoint_dir, "value_heads.pt")
    value_heads = ThreeHeads(
        input_dim=agent.policy.config.hidden_size,
        hidden_dim=1024,
    )
    if os.path.exists(vh_path):
        import torch
        value_heads.load_state_dict(
            torch.load(vh_path, map_location=cfg.model.train_device)
        )

    env = ClarificationEnv(cfg, tokenizer=agent.tokenizer)
    rng = random.Random(cfg.data.seed)

    results = asyncio.run(
        evaluate_policy(agent=agent, env=env, problems=eval_problems, rng=rng)
    )
    if d1 is not None:
        results["d1"] = d1
    return results


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading eval dataset...")
    _, eval_problems = load_humaneval_comm(
        use_variants=list(cfg.data.use_variants),
        eval_size=cfg.data.eval_size,
        seed=cfg.data.seed,
    )
    print(f"  Eval problems: {len(eval_problems)}")

    all_results = []

    if args.sweep:
        ckpt_root = cfg.paths.checkpoint_dir
        for entry in sorted(os.listdir(ckpt_root)):
            if not entry.startswith("d1_"):
                continue
            d1_val = int(entry.split("_")[1])
            d1_root = os.path.join(ckpt_root, entry)

            if args.sweep_mode == "best":
                priority = ["best", "final"]
            else:
                priority = ["final", "best"]

            ckpt_dir = None
            for tag in priority:
                candidate = os.path.join(d1_root, tag)
                if os.path.isdir(candidate):
                    ckpt_dir = candidate
                    break

            if ckpt_dir is None:
                # Fall back to latest iter checkpoint
                iters = sorted([
                    d for d in os.listdir(d1_root) if d.startswith("iter_")
                ])
                if not iters:
                    print(f"  [Skip] d₁={d1_val}: no checkpoint found")
                    continue
                ckpt_dir = os.path.join(d1_root, iters[-1])

            print(f"\n[Eval] d₁={d1_val} ({args.sweep_mode}) from {ckpt_dir}")
            result = eval_one(cfg, ckpt_dir, eval_problems, d1=d1_val)
            all_results.append(result)

            out_file = os.path.join(args.output_dir, f"d1_{d1_val}.json")
            with open(out_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Saved to {out_file}")

    elif args.checkpoint:
        result = eval_one(cfg, args.checkpoint, eval_problems, d1=args.d1)
        all_results.append(result)
        out_file = os.path.join(args.output_dir, f"results_d1_{args.d1}.json")
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {out_file}")

    else:
        print("Provide --checkpoint or --sweep.")
        return

    # Print table and plot if multiple results
    if all_results:
        print("\n=== Results ===")
        print_eval_table(all_results)

    if len(all_results) >= 2:
        pareto = compute_pareto_frontier(all_results)
        mode_tag = f"_{args.sweep_mode}" if args.sweep else ""
        plot_pareto_frontier(
            pareto,
            output_path=os.path.join(args.output_dir, f"pareto_frontier{mode_tag}.png"),
        )
        with open(os.path.join(args.output_dir, f"pareto{mode_tag}.json"), "w") as f:
            json.dump(pareto, f, indent=2)


if __name__ == "__main__":
    main()
