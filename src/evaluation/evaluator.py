"""
Evaluation utilities: run a trained policy on the held-out test set
and compute pass@1, average questions, and Pareto frontier metrics.
"""

import asyncio
import json
import os
import random
from typing import List, Dict

from src.data.dataset import Problem
from src.environment.env import ClarificationEnv


async def evaluate_policy(
    agent,
    env: ClarificationEnv,
    problems: List[Problem],
    rng: random.Random,
) -> Dict:
    """
    Run the agent on all problems in the eval set (greedy decoding, temp=0).
    Returns aggregate metrics.
    """
    from src.training.rollout import _run_episode

    # Switch agent to greedy for eval
    original_temp = agent.rollout_temperature
    agent.rollout_temperature = 0.0

    agent.sync_rollout_model()
    tasks = [
        _run_episode(i, problem, agent, env)
        for i, problem in enumerate(problems)
    ]
    episodes = await asyncio.gather(*tasks)

    agent.rollout_temperature = original_temp

    rewards   = [e.total_reward  for e in episodes]
    questions = [e.total_cost_q  for e in episodes]
    turns     = [e.total_cost_t  for e in episodes]

    return {
        "n_problems":    len(episodes),
        "avg_reward":    sum(rewards)   / len(rewards),
        "avg_questions": sum(questions) / len(questions),
        "avg_turns":     sum(turns)     / len(turns),
        "pass_at_1":     sum(1 for r in rewards if r == 1.0) / len(rewards),
        "per_episode": [
            {
                "problem_id":   e.problem_id,
                "reward":       e.total_reward,
                "questions":    e.total_cost_q,
                "turns":        e.total_cost_t,
            }
            for e in episodes
        ],
    }


def load_eval_results(results_dir: str) -> List[Dict]:
    """Load all eval result JSONs from a directory (one per d₁ setting)."""
    results = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".json"):
            with open(os.path.join(results_dir, fname)) as f:
                results.append(json.load(f))
    return results


def compute_pareto_frontier(results: List[Dict]) -> List[Dict]:
    """
    Given eval results for multiple policies (different d₁ values),
    compute the Pareto frontier points: (avg_questions, avg_reward).

    Returns list of dicts sorted by avg_questions ascending.
    """
    points = [
        {
            "d1":            r.get("d1"),
            "avg_questions": r["avg_questions"],
            "avg_reward":    r["avg_reward"],
        }
        for r in results
    ]
    points.sort(key=lambda p: p["avg_questions"])

    # Keep only Pareto-optimal points (reward non-decreasing as questions increase)
    pareto = []
    best_reward = -1.0
    for p in points:
        if p["avg_reward"] >= best_reward:
            pareto.append(p)
            best_reward = p["avg_reward"]

    return pareto


def plot_pareto_frontier(pareto_points: List[Dict], output_path: str = None):
    """
    Plot the Pareto frontier: avg_questions (x) vs avg_reward (y).
    Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed — skipping plot.")
        return

    xs = [p["avg_questions"] for p in pareto_points]
    ys = [p["avg_reward"]    for p in pareto_points]
    labels = [f"d₁={p['d1']}" for p in pareto_points]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, ys, "o-", color="steelblue", linewidth=2, markersize=8)
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 4))

    ax.set_xlabel("Average questions asked per problem")
    ax.set_ylabel("pass@1")
    ax.set_title("Pareto Frontier: Question Budget vs Code Correctness")
    ax.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Pareto plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def print_eval_table(results: List[Dict]):
    """Print a formatted table of eval results across d₁ settings."""
    header = f"{'d1':>4}  {'pass@1':>8}  {'avg_q':>7}  {'avg_turns':>9}  {'n':>6}"
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda x: x.get("d1", 0)):
        print(
            f"{r.get('d1', '?'):>4}  "
            f"{r['avg_reward']:>8.4f}  "
            f"{r['avg_questions']:>7.2f}  "
            f"{r['avg_turns']:>9.2f}  "
            f"{r['n_problems']:>6}"
        )
