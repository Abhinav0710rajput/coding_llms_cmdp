"""
Training entry point.

Examples:
    # Train d₁=0 (never-ask policy)
    python scripts/train.py --d1 0

    # Train d₁=1 (at most 1 question on average)
    python scripts/train.py --d1 1

    # Resume from checkpoint
    python scripts/train.py --d1 1 --resume checkpoints/d1_1/iter_0019

    # Override any config value
    python scripts/train.py --d1 1 training.n_iterations=40
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf

from src.data.dataset import load_humaneval_comm
from src.training.trainer import PPOLagrangianTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO-Lagrangian clarification agent")
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--d1", type=float, default=None,
        help="Question budget constraint d₁ (overrides config). E.g. 0, 0.5, 1.",
    )
    parser.add_argument(
        "--resume", default=None,
        help="Path to checkpoint directory to resume training from",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print episode details for first 3 episodes per iteration",
    )
    parser.add_argument(
        "overrides", nargs="*",
        help="OmegaConf-style overrides, e.g. training.n_iterations=40",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load base config
    cfg = OmegaConf.load(args.config)

    # Apply CLI overrides
    if args.overrides:
        overrides = OmegaConf.from_dotlist(args.overrides)
        cfg = OmegaConf.merge(cfg, overrides)

    # d1 override
    if args.d1 is not None:
        cfg.constraint.d1 = args.d1

    print(f"\nConfig:\n{OmegaConf.to_yaml(cfg)}")

    # Load dataset
    print("Loading dataset...")
    train_problems, eval_problems = load_humaneval_comm(
        use_variants=list(cfg.data.use_variants),
        eval_size=cfg.data.eval_size,
        seed=cfg.data.seed,
    )
    print(f"  Train problems: {len(train_problems)}")
    print(f"  Eval problems:  {len(eval_problems)}")

    # Build trainer
    trainer = PPOLagrangianTrainer(cfg, train_problems, eval_problems, verbose=args.verbose)

    # Resume if requested
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
