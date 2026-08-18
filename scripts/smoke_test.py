"""
Smoke test: validate the full pipeline end-to-end WITHOUT training.

Runs a handful of episodes with the base model (no LoRA training) and prints
results. Use this to verify that all components work on your HPC setup before
committing to a real training run.

Requirements:
  - OPENAI_API_KEY env var (for user simulator)
  - HF_TOKEN env var (not required for Qwen2.5-Coder-7B-Instruct)
  - At least 1 GPU (runs everything on cuda:0 in single-GPU mode)

Usage:
    # Quick test: 3 episodes, single GPU
    python scripts/smoke_test.py

    # Custom episode count
    python scripts/smoke_test.py --episodes 5

    # Skip model loading (test dataset + code executor only)
    python scripts/smoke_test.py --no-model

    # Two-GPU mode (matches training setup)
    python scripts/smoke_test.py --two-gpu
"""

import argparse
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test for the RL pipeline")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--no-model", action="store_true", help="Skip model loading (test data + executor only)")
    parser.add_argument("--two-gpu", action="store_true", help="Use 2 GPUs (cuda:0 + cuda:1) like real training")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file")
    return parser.parse_args()


def test_dataset():
    """Step 1: Verify dataset loading."""
    print("\n" + "=" * 60)
    print("STEP 1: Dataset loading")
    print("=" * 60)

    from src.data.dataset import load_humaneval_comm

    t0 = time.time()
    train_problems, eval_problems = load_humaneval_comm(
        use_variants=["prompt1a", "prompt1c", "prompt1p", "prompt2ac"],
        eval_size=100,
        seed=42,
    )
    elapsed = time.time() - t0

    print(f"  Loaded in {elapsed:.1f}s")
    print(f"  Train problems: {len(train_problems)}")
    print(f"  Eval problems:  {len(eval_problems)}")

    # Spot-check one problem
    p = train_problems[0]
    print(f"\n  Sample problem:")
    print(f"    task_id:      {p.task_id}")
    print(f"    entry_point:  {p.entry_point}")
    print(f"    degradation:  {p.degradation_type}")
    print(f"    test_cases:   {len(p.test_cases)}")
    print(f"    degraded len: {len(p.degraded_prompt)} chars")
    print(f"    original len: {len(p.original_prompt)} chars")

    assert len(train_problems) > 0, "No training problems loaded!"
    assert len(eval_problems) > 0, "No eval problems loaded!"
    assert p.test_cases, "First problem has no test cases!"
    print("\n  [PASS] Dataset loading works.")
    return train_problems, eval_problems


def test_code_executor(problems):
    """Step 2: Verify code execution with the reference solution."""
    print("\n" + "=" * 60)
    print("STEP 2: Code executor")
    print("=" * 60)

    from src.environment.code_executor import CodeExecutor, build_test_program

    # Minimal config stub for the executor
    cfg = OmegaConf.create({
        "code_executor": {"timeout": 10.0, "partial_credit": True}
    })
    executor = CodeExecutor(cfg)

    passed = 0
    tested = 0
    for p in problems[:10]:
        if not p.solution.strip():
            continue
        score = executor.run(p.solution, p.test_cases, p.entry_point)
        tested += 1
        if score == 1.0:
            passed += 1
        else:
            print(f"    WARN: {p.task_id} reference solution scored {score:.2f} (not 1.0)")

    print(f"  Tested {tested} reference solutions: {passed}/{tested} scored 1.0")

    # Test with bad code
    bad_score = executor.run("def foo(): return None", problems[0].test_cases, problems[0].entry_point)
    print(f"  Bad code score: {bad_score:.2f} (expected ~0.0)")

    # Test with empty code
    empty_score = executor.run("", problems[0].test_cases, problems[0].entry_point)
    print(f"  Empty code score: {empty_score:.2f} (expected 0.0)")
    assert empty_score == 0.0

    print("\n  [PASS] Code executor works.")


async def test_user_simulator(problems):
    """Step 3: Verify GPT-4o-mini user simulator."""
    print("\n" + "=" * 60)
    print("STEP 3: User simulator (GPT-4o-mini API)")
    print("=" * 60)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("  [SKIP] OPENAI_API_KEY not set. Skipping simulator test.")
        return False

    cfg = OmegaConf.create({
        "user_simulator": {
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "max_tokens": 300,
            "max_concurrent_api": 5,
        },
        "environment": {"multi_question_mode": "count"},
    })

    from src.environment.user_simulator import UserSimulator
    sim = UserSimulator(cfg)

    p = problems[0]
    print(f"  Testing with: {p.task_id}")
    print(f"  Question: 'What should the function return?'")

    t0 = time.time()
    answer, q_count = await sim.answer(
        question="What should the function return?",
        original_prompt=p.original_prompt,
    )
    elapsed = time.time() - t0

    print(f"  Answer ({elapsed:.1f}s): {answer[:200]}")
    print(f"  Question count: {q_count}")

    # Test multi-question counting
    answer2, q_count2 = await sim.answer(
        question="What are the input types and what are the edge cases?",
        original_prompt=p.original_prompt,
    )
    print(f"\n  Multi-question test:")
    print(f"  Question: 'What are the input types and what are the edge cases?'")
    print(f"  Answer: {answer2[:200]}")
    print(f"  Question count: {q_count2} (expected >= 2)")

    print("\n  [PASS] User simulator works.")
    return True


def test_model_loading(cfg):
    """Step 4: Verify model + LoRA loading."""
    print("\n" + "=" * 60)
    print("STEP 4: Model loading (Qwen2.5-Coder-7B-Instruct + LoRA)")
    print("=" * 60)

    import torch
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({mem:.1f} GB)")

    if not torch.cuda.is_available():
        print("  [SKIP] No GPU available. Skipping model load.")
        return None

    from src.models.agent import Agent

    t0 = time.time()
    agent = Agent(cfg)
    elapsed = time.time() - t0

    # Memory report
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f"  GPU {i}: {alloc:.1f} GB allocated, {reserved:.1f} GB reserved")

    print(f"  Model loaded in {elapsed:.1f}s")
    print("\n  [PASS] Model loading works.")
    return agent


def test_generation(agent, problems):
    """Step 5: Generate actions with the base model."""
    print("\n" + "=" * 60)
    print("STEP 5: Action generation (base model, no training)")
    print("=" * 60)

    import torch
    from src.environment.env import _build_prompt

    agent.sync_rollout_model()

    for i, p in enumerate(problems[:3]):
        prompt = _build_prompt(p, conversation=[], tokenizer=agent.tokenizer)
        print(f"\n  --- Episode {i+1}: {p.task_id} ({p.degradation_type}) ---")
        print(f"\n  Original spec (hidden from agent):")
        for line in p.original_prompt.strip().splitlines():
            print(f"    {line}")
        print(f"\n  Degraded spec (what the agent sees):")
        for line in p.degraded_prompt.strip().splitlines():
            print(f"    {line}")
        print(f"\n  Full prompt sent to model:")
        for line in prompt.strip().splitlines():
            print(f"    {line}")
        print(f"\n  Prompt length: {len(prompt)} chars")

        t0 = time.time()
        action_text, action_ids, action_logp, hidden, _ = agent.generate(prompt)
        elapsed = time.time() - t0

        print(f"  Generated in {elapsed:.1f}s ({len(action_ids)} tokens)")
        print(f"  Log-prob: {action_logp:.2f}")
        print(f"  Hidden shape: {hidden.shape}")
        print(f"  Action: {action_text[:300]}")

        # Check action format
        upper = action_text.strip().upper()
        if upper.startswith("[ASK]"):
            print(f"  -> ASK action detected")
        elif upper.startswith("[ANSWER]"):
            print(f"  -> ANSWER action detected")
        else:
            print(f"  -> WARNING: malformed action (no [ASK] or [ANSWER] prefix)")

    # Memory after generation
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        print(f"  GPU {i}: {alloc:.1f} GB allocated")

    print("\n  [PASS] Generation works.")


async def test_full_episode(agent, problems, cfg):
    """Step 6: Run complete episodes through the environment."""
    print("\n" + "=" * 60)
    print("STEP 6: Full episodes (env + simulator + executor)")
    print("=" * 60)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("  [SKIP] OPENAI_API_KEY not set.")
        return

    from src.environment.env import ClarificationEnv

    env = ClarificationEnv(cfg, tokenizer=agent.tokenizer)

    for i, p in enumerate(problems[:3]):
        print(f"\n  --- Episode {i+1}: {p.task_id} ({p.degradation_type}) ---")
        state = env.reset(p)
        total_reward = 0.0
        total_q_cost = 0.0
        total_t_cost = 0.0

        for turn in range(cfg.environment.max_turns):
            # Generate action
            action_text, action_ids, logp, hidden, _ = agent.generate(state.prompt)
            print(f"    Turn {turn+1}: {action_text[:120]}...")

            # Step environment
            result = await env.step(state, action_text)
            total_reward += result.reward
            total_q_cost += result.cost_q
            total_t_cost += result.cost_t

            action_type = result.info.get("action_type", "?")
            if action_type == "ask":
                print(f"      -> Simulator: {result.info['answer'][:100]}...")
                print(f"      -> Atomic questions: {result.info['atomic_count']}")
            elif action_type == "answer":
                print(f"      -> pass@1: {result.info['pass_rate']:.2f}")
            elif action_type == "malformed":
                print(f"      -> Malformed output")

            state = result.next_state
            if result.done:
                break

        print(f"    Summary: reward={total_reward:.2f}, questions={total_q_cost:.0f}, turns={total_t_cost:.0f}")

    print("\n  [PASS] Full episodes work.")


def test_value_heads(agent):
    """Step 7: Verify value heads forward pass."""
    print("\n" + "=" * 60)
    print("STEP 7: Value heads")
    print("=" * 60)

    import torch
    from src.models.value_heads import ThreeHeads

    hidden_dim = agent.policy.config.hidden_size
    device = agent.train_device

    heads = ThreeHeads(input_dim=hidden_dim, hidden_dim=1024).to(device)

    # Fake batch of hidden states
    fake_hidden = torch.randn(8, hidden_dim, device=device)
    v_r, v_q, v_t = heads(fake_hidden)

    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Input shape: {fake_hidden.shape}")
    print(f"  v_reward shape: {v_r.shape}, values: {v_r[:3].tolist()}")
    print(f"  v_q_cost shape: {v_q.shape}, values: {v_q[:3].tolist()}")
    print(f"  v_t_cost shape: {v_t.shape}, values: {v_t[:3].tolist()}")

    # Verify near-zero init
    assert v_r.abs().max() < 1.0, "Value head outputs should be near zero at init"
    print("\n  [PASS] Value heads work.")


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    # In single-GPU mode, put everything on cuda:0
    if not args.two_gpu:
        cfg.model.train_device = "cuda:0"
        cfg.model.rollout_device = "cuda:0"

    print("=" * 60)
    print("SMOKE TEST — RL Clarification Pipeline")
    print("=" * 60)
    print(f"  Mode: {'2-GPU' if args.two_gpu else 'single-GPU'}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Model loading: {'yes' if not args.no_model else 'SKIPPED'}")
    print(f"  OPENAI_API_KEY: {'set' if os.environ.get('OPENAI_API_KEY') else 'NOT SET'}")
    print(f"  HF_TOKEN: {'set' if os.environ.get('HF_TOKEN') else 'NOT SET'}")

    # --- Step 1: Dataset ---
    train_problems, eval_problems = test_dataset()

    # --- Step 2: Code executor ---
    test_code_executor(train_problems)

    # --- Step 3: User simulator ---
    sim_ok = asyncio.run(test_user_simulator(train_problems))

    if args.no_model:
        print("\n" + "=" * 60)
        print("DONE (--no-model: skipped steps 4-7)")
        print("=" * 60)
        return

    # --- Step 4: Model loading ---
    agent = test_model_loading(cfg)
    if agent is None:
        print("\nDONE (no GPU: skipped steps 5-7)")
        return

    # --- Step 5: Generation ---
    test_generation(agent, train_problems)

    # --- Step 6: Full episodes ---
    if sim_ok:
        asyncio.run(test_full_episode(agent, train_problems[:args.episodes], cfg))
    else:
        print("\n  [SKIP] Step 6: Full episodes (no API key)")

    # --- Step 7: Value heads ---
    test_value_heads(agent)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
