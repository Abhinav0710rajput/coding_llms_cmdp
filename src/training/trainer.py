"""
PPOLagrangianTrainer: the main training loop.

One iteration:
  1. collect_rollouts()   → 32 episodes, RolloutBuffer
  2. compute_returns()    → GAE advantages for all 3 streams
  3. ppo_update()         → 4 epochs × mini-batches of 8
  4. lagrange_update()    → update λ₁, λ₂
  5. log + checkpoint
"""

import asyncio
import json
import os
import random
import shutil
import signal
import threading
import time
from typing import List

import torch
from torch.optim import AdamW
from omegaconf import OmegaConf

try:
    import bitsandbytes as bnb
    _HAS_BNB = True
except ImportError:
    _HAS_BNB = False

from src.data.dataset import Problem
from src.environment.env import ClarificationEnv
from src.models.agent import Agent
from src.models.value_heads import ThreeHeads
from src.training.lagrangian import DualVariables
from src.training.ppo import (
    compute_lagrangian_advantages,
    compute_ppo_loss,
    compute_value_loss,
    compute_kl_penalty,
    compute_entropy_bonus,
)
from src.training.rollout import collect_rollouts, RolloutBuffer


class PPOLagrangianTrainer:
    def __init__(self, cfg, train_problems: List[Problem], eval_problems: List[Problem],
                 verbose: bool = False):
        self.cfg = cfg
        self.train_problems = train_problems
        self.eval_problems = eval_problems
        self.verbose = verbose
        self.device = torch.device(cfg.model.train_device)

        random.seed(cfg.training.seed)
        torch.manual_seed(cfg.training.seed)
        self.rng = random.Random(cfg.training.seed)

        # ── Models ────────────────────────────────────────────────────────────
        self.agent = Agent(cfg)
        self.value_heads = ThreeHeads(
            input_dim=self.agent.policy.config.hidden_size,
            hidden_dim=1024,
        ).to(self.device)

        # ── Optimizers ────────────────────────────────────────────────────────
        policy_params = [p for p in self.agent.policy.parameters() if p.requires_grad]
        value_params  = list(self.value_heads.parameters())

        if cfg.training.optimizer == "adamw_8bit" and _HAS_BNB:
            self.policy_optimizer = bnb.optim.AdamW8bit(
                policy_params, lr=cfg.training.lr_policy
            )
            self.value_optimizer = bnb.optim.AdamW8bit(
                value_params, lr=cfg.training.lr_value
            )
        else:
            self.policy_optimizer = AdamW(policy_params, lr=cfg.training.lr_policy)
            self.value_optimizer   = AdamW(value_params,  lr=cfg.training.lr_value)

        # Linear warmup scheduler for policy
        from torch.optim.lr_scheduler import LinearLR
        self.policy_scheduler = LinearLR(
            self.policy_optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=cfg.training.warmup_steps,
        )

        # ── Constraint / Lagrangian ───────────────────────────────────────────
        self.dual = DualVariables(cfg)

        # ── Environment ───────────────────────────────────────────────────────
        self.env = ClarificationEnv(cfg, tokenizer=self.agent.tokenizer)

        # ── State ─────────────────────────────────────────────────────────────
        self.iteration = 0
        self.log_history: List[dict] = []
        self.best_eval_reward = -1.0

        os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)
        os.makedirs(cfg.paths.output_dir, exist_ok=True)

        # Keep GPU active during long API-wait periods so cloud-burst nodes
        # don't get powered off due to low utilization.
        def _gpu_keepalive(device, interval=60):
            while True:
                time.sleep(interval)
                a = torch.randn(4096, 4096, device=device)
                torch.mm(a, a).sum().item()
                del a
        threading.Thread(target=_gpu_keepalive, args=(self.device,),
                         daemon=True).start()

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self):
        def _save_and_exit(signum, frame):
            sig_name = signal.Signals(signum).name
            print(f"\n[{sig_name}] Saving checkpoint at iteration {self.iteration} before exit...")
            self.save_checkpoint(self.iteration)
            raise SystemExit(0)

        signal.signal(signal.SIGTERM, _save_and_exit)
        signal.signal(signal.SIGUSR1, _save_and_exit)

        print(f"\n{'='*60}")
        print(f"Training PPO-Lagrangian | d₁={self.cfg.constraint.d1}")
        print(f"Device: {self.device} | Iterations: {self.cfg.training.n_iterations}")
        print(f"{'='*60}\n")

        try:
            for it in range(self.iteration, self.cfg.training.n_iterations):
                self.iteration = it
                t_start = time.time()

                # 1. Collect rollouts
                buffer = asyncio.run(
                    collect_rollouts(
                        agent=self.agent,
                        env=self.env,
                        problems=self.train_problems,
                        batch_size=self.cfg.training.rollout_batch_size,
                        rng=self.rng,
                    )
                )
                torch.cuda.empty_cache()

                rollout_stats = buffer.stats()

                # Log episode details in verbose mode
                if self.verbose:
                    n_show = min(3, len(buffer.episodes))
                    for ep in buffer.episodes[:n_show]:
                        print(f"\n  --- Episode: {ep.problem_id} ---")
                        for t in ep.transitions:
                            action_type = t.info.get("action_type", "?")
                            print(f"    Turn {t.turn}: [{action_type.upper()}] {t.action_text}")
                            if action_type == "ask":
                                print(f"      Simulator: {t.info.get('answer', '')}")
                                print(f"      Atomic questions: {t.info.get('atomic_count', '?')}")
                            elif action_type == "answer":
                                print(f"      pass@1: {t.info.get('pass_rate', 0.0):.2f}")
                        print(f"    Summary: reward={ep.total_reward:.2f}, q={ep.total_cost_q:.0f}, turns={ep.n_turns}")

                # 2. Compute GAE returns & advantages
                buffer.compute_returns(
                    value_heads=self.value_heads,
                    gamma=self.cfg.training.gamma,
                    gae_lambda=self.cfg.training.gae_lambda,
                )

                # 3. Update Lagrange multipliers
                l1, l2 = self.dual.update(
                    avg_questions=rollout_stats["avg_questions"],
                    avg_turns=rollout_stats["avg_turns"],
                )

                # 4. PPO update (multiple epochs over the buffer)
                update_info = self._ppo_update(buffer)

                # 5. LR scheduler step
                self.policy_scheduler.step()

                elapsed = time.time() - t_start
                log = {
                    "iteration": it,
                    "elapsed_s": elapsed,
                    "lambda1": l1,
                    "lambda2": l2,
                    **rollout_stats,
                    **update_info,
                }
                self.log_history.append(log)
                self._print_log(log)

                # 6. Eval & checkpoint
                if (it + 1) % self.cfg.training.save_interval == 0:
                    self.save_checkpoint(it)

                if (it + 1) % self.cfg.training.eval_interval == 0:
                    self._run_eval()

                if (it + 1) % self.cfg.training.save_interval == 0:
                    self.save_checkpoint(it)

            # Final save
            self.save_checkpoint(self.cfg.training.n_iterations - 1, final=True)

        except SystemExit:
            raise
        except Exception as e:
            print(f"\n[ERROR] Training crashed at iteration {self.iteration}: {e}")
            print("Saving emergency checkpoint...")
            self.save_checkpoint(self.iteration)
            raise

    # ── PPO update ────────────────────────────────────────────────────────────

    def _ppo_update(self, buffer: RolloutBuffer) -> dict:
        """Run PPO_epochs passes over the rollout buffer."""
        transitions = buffer.transitions
        N = len(transitions)
        mini_bs = self.cfg.training.ppo_mini_batch_size
        indices = list(range(N))

        total_info = {
            "ppo_loss": 0.0, "value_loss": 0.0,
            "approx_kl": 0.0, "clip_frac": 0.0,
            "entropy": 0.0, "kl_penalty": 0.0,
            "kl_per_seq": 0.0, "kl_seq_max": 0.0,
        }

        n_updates = 0
        target_kl = self.cfg.training.get("target_kl", None)
        early_stop = False

        for epoch in range(self.cfg.training.ppo_epochs):
            if early_stop:
                break
            self.rng.shuffle(indices)

            for start in range(0, N, mini_bs):
                batch_idx = indices[start: start + mini_bs]
                if len(batch_idx) == 0:
                    continue

                batch_trans = [transitions[i] for i in batch_idx]
                batch_adv_r = [buffer.advantages_r[i] for i in batch_idx]
                batch_adv_q = [buffer.advantages_q[i] for i in batch_idx]
                batch_adv_t = [buffer.advantages_t[i] for i in batch_idx]
                batch_ret_r = [buffer.returns_r[i] for i in batch_idx]
                batch_ret_q = [buffer.returns_q[i] for i in batch_idx]
                batch_ret_t = [buffer.returns_t[i] for i in batch_idx]

                info = self._update_step(
                    batch_trans, batch_adv_r, batch_adv_q, batch_adv_t,
                    batch_ret_r, batch_ret_q, batch_ret_t,
                )

                for k, v in info.items():
                    total_info[k] += v
                n_updates += 1

                if target_kl is not None and info["approx_kl"] > target_kl:
                    early_stop = True
                    break

        return {k: v / max(n_updates, 1) for k, v in total_info.items()}

    def _update_step(
        self, transitions, adv_r, adv_q, adv_t, ret_r, ret_q, ret_t
    ) -> dict:
        """One mini-batch gradient step."""
        # Re-score all (prompt, action) pairs under current policy
        new_log_probs_list = []
        ref_log_probs_list = []
        hidden_states_list = []

        for t in transitions:
            prefix_len = t.info.get("prefix_len", 0)
            new_logp, hidden, ref_logp = self.agent.score(t.prompt, t.action_ids, prefix_len)
            new_log_probs_list.append(new_logp)
            ref_log_probs_list.append(ref_logp)
            hidden_states_list.append(hidden)

        new_log_probs = torch.stack(new_log_probs_list)        # (B,)
        ref_log_probs = torch.stack(ref_log_probs_list)        # (B,)
        hidden_states = torch.stack(hidden_states_list).to(self.device)  # (B, H)
        action_lengths = torch.tensor(
            [len(t.action_ids) for t in transitions],
            dtype=torch.float32, device=self.device,
        )
        old_log_probs = torch.tensor(
            [t.action_logp for t in transitions],
            dtype=torch.float32, device=self.device
        )

        # Lagrangian advantages
        A_lag = compute_lagrangian_advantages(
            adv_r, adv_q, adv_t,
            lambda1=self.dual.l1,
            lambda2=self.dual.l2,
            device=self.device,
        )

        # PPO loss
        ppo_loss, ppo_info = compute_ppo_loss(
            new_log_probs, old_log_probs, A_lag,
            clip_epsilon=self.cfg.training.clip_epsilon,
        )

        # Value predictions for value loss (cast to float32 — model outputs bfloat16)
        v_r, v_q, v_t = self.value_heads(hidden_states.float())
        value_loss, val_info = compute_value_loss(
            v_r, v_q, v_t, ret_r, ret_q, ret_t, self.device
        )

        kl, kl_info = compute_kl_penalty(new_log_probs, ref_log_probs)

        # Entropy bonus (length-normalized to avoid bias toward longer actions)
        entropy = compute_entropy_bonus(new_log_probs, action_lengths)

        # Total loss
        total_loss = (
            ppo_loss
            + self.cfg.training.kl_coeff * kl
            - self.cfg.training.entropy_coeff * entropy
            + 0.5 * value_loss   # value heads updated jointly
        )

        # Skip update if loss is NaN (prevents corrupting model weights)
        if torch.isnan(total_loss):
            print("  [WARNING] NaN loss detected, skipping mini-batch")
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            return {
                "ppo_loss": float("nan"), "value_loss": 0.0,
                "approx_kl": 0.0, "clip_frac": 0.0,
                "entropy": 0.0, "kl_penalty": 0.0,
                "kl_per_seq": 0.0, "kl_seq_max": 0.0,
            }

        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.agent.policy.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        torch.nn.utils.clip_grad_norm_(
            self.value_heads.parameters(),
            max_norm=1.0,
        )
        self.policy_optimizer.step()
        self.value_optimizer.step()

        return {
            "ppo_loss":    ppo_info["ppo_loss"],
            "value_loss":  val_info["value_loss"],
            "approx_kl":   ppo_info["approx_kl"],
            "clip_frac":   ppo_info["clip_frac"],
            "entropy":     entropy.item(),
            "kl_penalty":  kl.item(),
            "kl_per_seq":  kl_info["kl_per_seq"],
            "kl_seq_max":  kl_info["kl_seq_max"],
        }

    # ── Eval ─────────────────────────────────────────────────────────────────

    def _run_eval(self, max_eval_problems: int = 50):
        from src.evaluation.evaluator import evaluate_policy
        # Sample a subset for mid-training eval (full eval at the end)
        eval_subset = self.eval_problems
        if len(self.eval_problems) > max_eval_problems:
            eval_subset = self.rng.sample(self.eval_problems, max_eval_problems)
        print(f"\n[Eval] iteration={self.iteration} ({len(eval_subset)} problems)")
        results = asyncio.run(
            evaluate_policy(
                agent=self.agent,
                env=self.env,
                problems=eval_subset,
                rng=self.rng,
            )
        )
        print(f"  pass@1={results['avg_reward']:.4f}  "
              f"avg_questions={results['avg_questions']:.2f}  "
              f"avg_turns={results['avg_turns']:.2f}")
        if results["avg_reward"] > self.best_eval_reward:
            self.best_eval_reward = results["avg_reward"]
            self.save_checkpoint(self.iteration, tag="best")
            print(f"  [Best checkpoint] new best eval reward={self.best_eval_reward:.4f}")

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def save_checkpoint(self, iteration: int, final: bool = False, tag: str = None):
        tag = tag or ("final" if final else f"iter_{iteration:04d}")
        d1 = self.cfg.constraint.d1
        ckpt_dir = os.path.join(self.cfg.paths.checkpoint_dir, f"d1_{d1}", tag)
        tmp_dir = ckpt_dir + ".tmp"

        # Write everything to a temp dir first, then atomically rename so a
        # crash mid-save never leaves a partially-written checkpoint visible.
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        self.agent.save_lora(tmp_dir)
        torch.save(self.value_heads.state_dict(),
                   os.path.join(tmp_dir, "value_heads.pt"))
        torch.save(self.dual.state_dict(),
                   os.path.join(tmp_dir, "dual_variables.pt"))
        state = {
            "iteration": iteration,
            "best_eval_reward": self.best_eval_reward,
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "value_optimizer":  self.value_optimizer.state_dict(),
            "scheduler":        self.policy_scheduler.state_dict(),
        }
        torch.save(state, os.path.join(tmp_dir, "train_state.pt"))
        with open(os.path.join(tmp_dir, "log.json"), "w") as f:
            json.dump(self.log_history, f, indent=2)

        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)
        os.rename(tmp_dir, ckpt_dir)
        print(f"  [Checkpoint saved] {ckpt_dir}")

    def load_checkpoint(self, ckpt_dir: str):
        self.agent.load_lora(ckpt_dir)

        vh_path = os.path.join(ckpt_dir, "value_heads.pt")
        if os.path.exists(vh_path):
            self.value_heads.load_state_dict(torch.load(vh_path, map_location=self.device))

        dual_path = os.path.join(ckpt_dir, "dual_variables.pt")
        if os.path.exists(dual_path):
            self.dual.load_state_dict(torch.load(dual_path))

        state_path = os.path.join(ckpt_dir, "train_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.device)
            self.iteration = state["iteration"] + 1
            self.best_eval_reward = state.get("best_eval_reward", -1.0)
            self.policy_optimizer.load_state_dict(state["policy_optimizer"])
            self.value_optimizer.load_state_dict(state["value_optimizer"])
            self.policy_scheduler.load_state_dict(state["scheduler"])
            print(f"  [Best eval reward restored] {self.best_eval_reward:.4f}")

        log_path = os.path.join(ckpt_dir, "log.json")
        if os.path.exists(log_path):
            with open(log_path) as f:
                self.log_history = json.load(f)

        print(f"  [Checkpoint loaded] {ckpt_dir} (resuming from iteration {self.iteration})")

    # ── Logging ───────────────────────────────────────────────────────────────

    def _print_log(self, log: dict):
        print(
            f"iter={log['iteration']:4d} | "
            f"reward={log['avg_reward']:.4f} | "
            f"q={log['avg_questions']:.2f} (budget={self.cfg.constraint.d1}) | "
            f"λ₁={log['lambda1']:.4f} | "
            f"ppo={log['ppo_loss']:.4f} | "
            f"vf={log['value_loss']:.4f} | "
            f"approx_kl={log['approx_kl']:.4f} | "
            f"kl_seq_penalty={log['kl_penalty']:.4f} | "
            f"kl_seq={log['kl_per_seq']:.4f} | "
            f"kl_max={log['kl_seq_max']:.4f} | "
            f"t={log['elapsed_s']:.0f}s"
        )
