"""
Baseline evaluation: test the base model (no training) on many problems.

Runs the agent on each problem with NO conversation (single turn, straight to [ANSWER]).
This measures the base model's raw coding ability on the degraded specs.

Also runs a multi-turn version where the agent can ask questions (with the simulator).

Usage:
    # Single-turn only (no API key needed), 30 problems
    python scripts/baseline_eval.py --n 30

    # Include multi-turn episodes (needs OPENAI_API_KEY)
    python scripts/baseline_eval.py --n 30 --multi-turn

    # Two-GPU mode
    python scripts/baseline_eval.py --n 30 --two-gpu --multi-turn
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline evaluation")
    parser.add_argument("--n", type=int, default=30, help="Number of problems to test")
    parser.add_argument("--multi-turn", action="store_true", help="Also run multi-turn episodes with simulator")
    parser.add_argument("--two-gpu", action="store_true", help="Use 2 GPUs")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: random)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint dir to load trained LoRA weights")
    parser.add_argument("--all", dest="run_all", action="store_true", help="Run ALL eval problems (469 problems)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Generation temperature (default: 0.8; use 0.0 for greedy)")
    parser.add_argument("--progress-file", default=None, help="Path to JSONL file for saving/resuming eval progress")
    parser.add_argument("--exclude-file", default=None,
                        help="Path to checkpoint_val JSON; task_ids in val_task_ids will be excluded")
    parser.add_argument("--mt-only", action="store_true",
                        help="Skip single-turn eval; run multi-turn only")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    if not args.two_gpu:
        cfg.model.train_device = "cuda:0"
        cfg.model.rollout_device = "cuda:0"


    import torch
    from src.data.dataset import load_humaneval_comm
    from src.models.agent import Agent
    from src.environment.env import ClarificationEnv, _build_prompt
    from src.environment.code_executor import CodeExecutor, extract_code, _extract_helper_context

    n_label = "all" if args.run_all else str(args.n)
    print("=" * 60)
    print(f"BASELINE EVAL — {n_label} problems")
    print("=" * 60)

    # Load data
    train_problems, eval_problems = load_humaneval_comm(
        use_variants=list(cfg.data.use_variants),
        eval_size=cfg.data.eval_size,
        seed=cfg.data.seed,
    )

    # Sample problems
    seed = args.seed if args.seed is not None else random.randint(0, 999999)
    print(f"  Seed: {seed}")
    rng = random.Random(seed)
    diverse_problems = []

    # Load exclusion list from checkpoint_val output if provided
    exclude_ids = set()
    if args.exclude_file:
        with open(args.exclude_file) as f:
            val_data = json.load(f)
        exclude_ids = set(val_data["val_task_ids"])
        print(f"  Excluding {len(exclude_ids)} validation problems (from {args.exclude_file})")

    if args.run_all:
        diverse_problems = [p for p in eval_problems if p.task_id not in exclude_ids]
        rng.shuffle(diverse_problems)
        print(f"  Selected {len(diverse_problems)} problems (all eval problems minus exclusions)")
    else:
        seen_base = set()
        shuffled = list(eval_problems)
        rng.shuffle(shuffled)
        for p in shuffled:
            base_id = p.task_id.rsplit("/", 1)[0]
            if base_id not in seen_base:
                seen_base.add(base_id)
                diverse_problems.append(p)
            if len(diverse_problems) >= args.n:
                break
        print(f"  Selected {len(diverse_problems)} diverse problems")

    # Load progress file for resume
    progress_file = args.progress_file
    st_done = {}  # task_id -> result
    mt_done = {}  # task_id -> result
    if progress_file and os.path.exists(progress_file):
        with open(progress_file) as f:
            for line in f:
                r = json.loads(line.strip())
                if r.get("phase") == "st":
                    st_done[r["task_id"]] = r
                elif r.get("phase") == "mt":
                    mt_done[r["task_id"]] = r
        print(f"  Resuming: {len(st_done)} ST, {len(mt_done)} MT episodes already completed")

    def _append_progress(result):
        if progress_file:
            with open(progress_file, "a") as f:
                f.write(json.dumps(result) + "\n")

    # Load model
    print(f"\nLoading model: {cfg.model.name}")
    agent = Agent(cfg)
    if args.checkpoint:
        print(f"  Loading checkpoint: {args.checkpoint}")
        agent.load_lora(args.checkpoint)
    agent.sync_rollout_model()
    agent.rollout_temperature = args.temperature
    print(f"  Generation temperature: {args.temperature}")

    # Code executor
    executor = CodeExecutor(cfg)

    # ── Single-turn eval (no questions, just code) ───────────────────────
    if args.mt_only:
        print("\n[Skipping single-turn eval (--mt-only)]")
        single_results = list(st_done.values())
    else:
      print("\n" + "=" * 60)
      print("SINGLE-TURN EVAL (agent submits code immediately)")
      print("=" * 60)

      _DIRECT_SYSTEM = (
          "You are a coding assistant. Write a Python solution for the task below. "
          "Respond with ONLY the Python code. No explanations, no questions."
      )

      def _build_direct_prompt(degraded_prompt, tokenizer):
          messages = [
              {"role": "system", "content": _DIRECT_SYSTEM},
              {"role": "user", "content": f"Task:\n{degraded_prompt}"},
          ]
          return tokenizer.apply_chat_template(
              messages, tokenize=False, add_generation_prompt=True,
          )

      single_results = list(st_done.values())
      for i, p in enumerate(diverse_problems):
          if p.task_id in st_done:
              print(f"\n  [{i+1}/{len(diverse_problems)}] {p.task_id} — skipped (already done)")
              continue
          prompt = _build_direct_prompt(p.degraded_prompt, agent.tokenizer)
          action_text, _, _, _, _ = agent.generate(prompt, constrain_prefix=False)

          upper = action_text.strip().upper()
          if upper.startswith("[ANSWER]"):
              code = extract_code(action_text)
              action_type = "answer"
          elif upper.startswith("[ASK]"):
              action_type = "ask"
          else:
              action_type = "malformed"

          # Always try to extract and run code, even from malformed/ask outputs
          # This tells us if the model COULD have scored, separate from formatting
          code = extract_code(action_text)
          if not code.strip() and "def " in action_text:
              # Fallback: grab everything from first 'def' onward
              idx = action_text.find("def ")
              code = action_text[idx:].strip()
          context = _extract_helper_context(p.degraded_prompt, p.entry_point)
          score = executor.run(code, p.test_cases, p.entry_point, context=context) if code.strip() else 0.0

          result = {
              "phase": "st",
              "task_id": p.task_id,
              "degradation": p.degradation_type,
              "action_type": action_type,
              "score": score,
          }
          single_results.append(result)
          _append_progress(result)

          status = f"pass@1={score:.2f}" if action_type == "answer" else f"{action_type} (score={score:.2f})"
          print(f"\n  [{i+1}/{len(diverse_problems)}] {p.task_id} ({p.degradation_type}): {status}")
          print(f"  Original spec (hidden from agent):")
          for line in p.original_prompt.strip().splitlines():
              print(f"    {line}")
          print(f"  Test cases: {p.test_cases}")
          print(f"  Degraded spec:")
          for line in p.degraded_prompt.strip().splitlines():
              print(f"    {line}")
          print(f"  Raw output:")
          for line in action_text.strip().splitlines():
              print(f"    {line}")
          print(f"  Extracted code:")
          for line in (code or "(none)").strip().splitlines():
              print(f"    {line}")

      # Single-turn summary
      n_answer = sum(1 for r in single_results if r["action_type"] == "answer")
      n_ask = sum(1 for r in single_results if r["action_type"] == "ask")
      n_malformed = sum(1 for r in single_results if r["action_type"] == "malformed")
      all_scores = [r["score"] for r in single_results]
      avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
      n_perfect = sum(1 for s in all_scores if s == 1.0)
      n_partial = sum(1 for s in all_scores if 0.0 < s < 1.0)
      n_zero = sum(1 for s in all_scores if s == 0.0)

      print(f"\n  Single-turn summary:")
      print(f"    Format: [ANSWER]={n_answer}, [ASK]={n_ask}, malformed={n_malformed}")
      print(f"    Avg pass@1 (all, code extracted regardless of format): {avg_score:.3f}")
      print(f"    Perfect (1.0):   {n_perfect}/{len(diverse_problems)}")
      print(f"    Partial (0<x<1): {n_partial}/{len(diverse_problems)}")
      print(f"    Failed (0.0):    {n_zero}/{len(diverse_problems)}")

    # ── Multi-turn eval (with simulator) ─────────────────────────────────
    if args.multi_turn:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("\n  [SKIP] Multi-turn eval: OPENAI_API_KEY not set")
        else:
            print("\n" + "=" * 60)
            print("MULTI-TURN EVAL (agent can ask questions)")
            print("=" * 60)

            env = ClarificationEnv(cfg, tokenizer=agent.tokenizer)
            multi_results = list(mt_done.values())

            for i, p in enumerate(diverse_problems):
                if p.task_id in mt_done:
                    print(f"\n  [{i+1}/{len(diverse_problems)}] {p.task_id} — skipped (already done)")
                    continue
                state = env.reset(p)
                total_q = 0
                total_t = 0

                print(f"\n  [{i+1}/{len(diverse_problems)}] {p.task_id} ({p.degradation_type})")
                print(f"  Original spec (hidden from agent):")
                for line in p.original_prompt.strip().splitlines():
                    print(f"    {line}")
                print(f"  Test cases: {p.test_cases}")
                print(f"  Degraded spec:")
                for line in p.degraded_prompt.strip().splitlines():
                    print(f"    {line}")

                for turn in range(cfg.environment.max_turns):
                    action_text, _, _, _, _ = agent.generate(state.prompt)
                    result = asyncio.run(env.step(state, action_text))
                    total_q += result.cost_q
                    total_t += result.cost_t

                    action_type = result.info.get("action_type", "?")
                    print(f"  Turn {turn+1}:")
                    print(f"    Action: {action_text}")
                    if action_type == "ask":
                        print(f"    Simulator answer: {result.info['answer']}")
                        print(f"    Atomic questions: {result.info['atomic_count']}")
                    elif action_type == "answer":
                        print(f"    pass@1: {result.info['pass_rate']:.2f}")
                    elif action_type == "malformed":
                        print(f"    [MALFORMED OUTPUT]")

                    state = result.next_state
                    if result.done:
                        break

                score = result.reward
                mt_result = {
                    "phase": "mt",
                    "task_id": p.task_id,
                    "degradation": p.degradation_type,
                    "score": score,
                    "questions": total_q,
                    "turns": total_t,
                }
                multi_results.append(mt_result)
                _append_progress(mt_result)

                print(f"  Summary: pass@1={score:.2f}, questions={total_q:.0f}, turns={total_t:.0f}")

            # Multi-turn summary
            m_scores = [r["score"] for r in multi_results]
            m_avg = sum(m_scores) / len(m_scores) if m_scores else 0.0
            m_perfect = sum(1 for s in m_scores if s == 1.0)
            m_partial = sum(1 for s in m_scores if 0.0 < s < 1.0)
            m_avg_q = sum(r["questions"] for r in multi_results) / len(multi_results)

            print(f"\n  Multi-turn summary:")
            print(f"    Avg pass@1:      {m_avg:.3f}")
            print(f"    Perfect (1.0):   {m_perfect}/{len(diverse_problems)}")
            print(f"    Partial (0<x<1): {m_partial}/{len(diverse_problems)}")
            print(f"    Avg questions:   {m_avg_q:.1f}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
