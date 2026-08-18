"""
Agent: LLM with LoRA adapters for the clarification task.

Responsibilities:
  generate()  → sample an action during rollout (returns text + log_prob + hidden_state)
  score()     → re-score a stored (state, action) pair during PPO update
  kl_penalty()→ KL divergence between current policy and frozen reference model
  save/load LoRA checkpoints
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType


class Agent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.train_device = torch.device(cfg.model.train_device)
        self.rollout_device = torch.device(cfg.model.rollout_device)
        self.max_new_tokens = cfg.environment.max_new_tokens
        self.max_seq_len = cfg.environment.max_seq_len
        self.rollout_temperature = cfg.environment.rollout_temperature
        self.dtype = torch.bfloat16 if cfg.model.dtype == "bfloat16" else torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.name,
            padding_side="left",
            truncation_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── Policy model (trainable, LoRA) on GPU 0 ──────────────────────────
        base = AutoModelForCausalLM.from_pretrained(
            cfg.model.name,
            torch_dtype=self.dtype,
            device_map=cfg.model.train_device,
        )
        if cfg.model.gradient_checkpointing:
            base.gradient_checkpointing_enable()

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.model.lora_rank,
            lora_alpha=cfg.model.lora_alpha,
            lora_dropout=cfg.model.lora_dropout,
            target_modules=list(cfg.model.lora_target_modules),
            bias="none",
        )
        self.policy = get_peft_model(base, lora_cfg)
        self.policy.print_trainable_parameters()

        # ── Reference model (frozen base, no LoRA) on GPU 1 ─────────────────
        # Skipped in single-GPU mode: only needed for kl_penalty() during training.
        if self.train_device != self.rollout_device:
            self.reference = AutoModelForCausalLM.from_pretrained(
                cfg.model.name,
                torch_dtype=self.dtype,
                device_map=cfg.model.rollout_device,
            )
            for p in self.reference.parameters():
                p.requires_grad = False
            self.reference.eval()
        else:
            self.reference = None

        # ── Rollout copy (policy weights synced before each rollout) ─────────
        # We re-use the reference model device for rollout inference to avoid
        # having three full models in memory. The rollout copy is just the policy
        # with LoRA weights copied over, running in eval mode on GPU 1.
        # Updated via sync_rollout_model() before each rollout batch.
        self._rollout_model = None

        # ── Prefix token IDs for constrained decoding ────────────────────────
        # Force every generation to start with either [ASK] or [ANSWER].
        # The model chooses which prefix by comparing their log-probs given the
        # prompt, then generates freely after the prefix.
        self._ask_ids = self.tokenizer.encode("[ASK] ", add_special_tokens=False)
        self._answer_ids = self.tokenizer.encode("[ANSWER] ", add_special_tokens=False)

    def _sample_kwargs(self):
        """Return generate() kwargs for the current rollout temperature."""
        if self.rollout_temperature == 0.0:
            return {"do_sample": False}
        return {"do_sample": True, "temperature": self.rollout_temperature}

    def sync_rollout_model(self):
        """
        Copy current LoRA weights to GPU 1 for rollout inference.
        Called once per training iteration before collect_rollouts().
        In single-GPU mode (train_device == rollout_device), reuse the policy
        directly to avoid loading a third model copy that would OOM.
        """
        if self.train_device == self.rollout_device:
            # Single-GPU mode: policy IS the rollout model — no second copy needed
            if self._rollout_model is None:
                if hasattr(self.policy, 'gradient_checkpointing_disable'):
                    self.policy.gradient_checkpointing_disable()
                self.policy.config.use_cache = True
                self._rollout_model = self.policy
            self._rollout_model.eval()
            return

        if self._rollout_model is None:
            # First sync: create the rollout model on GPU 1
            base1 = AutoModelForCausalLM.from_pretrained(
                self.cfg.model.name,
                torch_dtype=self.dtype,
                device_map=self.cfg.model.rollout_device,
            )
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.cfg.model.lora_rank,
                lora_alpha=self.cfg.model.lora_alpha,
                lora_dropout=self.cfg.model.lora_dropout,
                target_modules=list(self.cfg.model.lora_target_modules),
                bias="none",
            )
            self._rollout_model = get_peft_model(base1, lora_cfg)
            # Ensure no gradient checkpointing on rollout model (critical for KV cache speed)
            if hasattr(self._rollout_model, 'gradient_checkpointing_disable'):
                self._rollout_model.gradient_checkpointing_disable()
            self._rollout_model.config.use_cache = True
            print(f"  Rollout model: gradient_checkpointing={getattr(self._rollout_model, 'is_gradient_checkpointing', False)}, use_cache={self._rollout_model.config.use_cache}")

        # Copy LoRA weights from policy (GPU 0) to rollout model (GPU 1)
        policy_state = {
            k: v.to(self.rollout_device)
            for k, v in self.policy.state_dict().items()
            if "lora_" in k
        }
        self._rollout_model.load_state_dict(policy_state, strict=False)
        self._rollout_model.eval()

    @torch.no_grad()
    def generate(
        self, prompt: str, constrain_prefix: bool = True,
    ) -> Tuple[str, List[int], float, torch.Tensor, int]:
        """
        Generate one action during rollout (runs on GPU 1).

        Args:
            prompt:            full text prompt string
            constrain_prefix:  if True, force output to start with [ASK] or [ANSWER].
                               Set False for non-env prompts (e.g. direct code generation).

        Returns:
            action_text:  raw generated text (e.g. "[ASK] What is X?")
            action_ids:   list of token IDs for the generated action
            action_logp:  sum of log-probs of continuation tokens only (scalar float)
            state_hidden: hidden state of last prompt token (for value heads)
                          shape: (hidden_dim,) on CPU
            prefix_len:   number of prefix tokens (0 for unconstrained generation)
        """
        model = self._rollout_model or self.policy
        model.eval()
        device = self.rollout_device if self._rollout_model else self.train_device

        # Ensure KV cache is enabled and gradient checkpointing is off
        # (critical for generation speed — without cache, generation is O(n²))
        if hasattr(model, 'gradient_checkpointing_disable'):
            model.gradient_checkpointing_disable()
        model.config.use_cache = True

        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_seq_len,
            truncation=True,
            padding=False,
            add_special_tokens=False,  # chat template already includes special tokens
        ).to(device)

        prompt_len = enc.input_ids.shape[1]

        # Forward pass on prompt to get hidden state
        with torch.no_grad():
            outputs = model(
                **enc,
                output_hidden_states=True,
                use_cache=True,
            )
        last_hidden = outputs.hidden_states[-1][0, -1, :].cpu().float()

        if not constrain_prefix:
            # ── Unconstrained generation (e.g. direct code prompt) ────────
            gen_out = model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                **self._sample_kwargs(),
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
            )
            action_ids = gen_out.sequences[0, prompt_len:].tolist()
            action_text = self.tokenizer.decode(action_ids, skip_special_tokens=True).strip()

            # Compute log-probs via forward pass (not generate() scores, which
            # are filtered by logit processors and don't match score())
            action_logp = self._forward_logp(model, enc.input_ids, action_ids, 0, device)
            return action_text, action_ids, action_logp, last_hidden, 0

        # ── Constrained prefix selection ─────────────────────────────────
        # Use the prompt's logits to cheaply score the first token of each
        # prefix, then sample which prefix to use. This avoids expensive
        # full forward passes just to score the prefix.
        prompt_logits = outputs.logits[0, -1, :]
        log_probs = F.log_softmax(prompt_logits, dim=-1)

        # Score first token of each prefix (cheap — just a lookup)
        ask_score = log_probs[self._ask_ids[0]].item()
        ans_score = log_probs[self._answer_ids[0]].item()

        # If first tokens are the same (e.g., both start with '['),
        # use the KV cache to score the second token too
        if self._ask_ids[0] == self._answer_ids[0] and len(self._ask_ids) > 1 and len(self._answer_ids) > 1:
            first_tok = torch.tensor([[self._ask_ids[0]]], device=device)
            with torch.no_grad():
                out2 = model(input_ids=first_tok, past_key_values=outputs.past_key_values, use_cache=True)
            lp2 = F.log_softmax(out2.logits[0, -1, :], dim=-1)
            ask_score += lp2[self._ask_ids[1]].item()
            ans_score += lp2[self._answer_ids[1]].item()

        if self.rollout_temperature == 0.0:
            choice = 0 if ask_score >= ans_score else 1
        else:
            probs = torch.softmax(
                torch.tensor([ask_score, ans_score]) / self.rollout_temperature, dim=0
            )
            choice = torch.multinomial(probs, 1).item()

        if choice == 0:
            prefix_ids = self._ask_ids
        else:
            prefix_ids = self._answer_ids

        # ── Generate continuation after the prefix ───────────────────────
        prefix_tensor = torch.tensor([prefix_ids], device=device)
        full_input = torch.cat([enc.input_ids, prefix_tensor], dim=1)
        full_mask = torch.ones_like(full_input)

        gen_out = model.generate(
            input_ids=full_input,
            attention_mask=full_mask,
            max_new_tokens=self.max_new_tokens - len(prefix_ids),
            **self._sample_kwargs(),
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
        )

        # Combine prefix + generated continuation
        continuation_ids = gen_out.sequences[0, prompt_len + len(prefix_ids):].tolist()
        action_ids = prefix_ids + continuation_ids
        action_text = self.tokenizer.decode(action_ids, skip_special_tokens=True).strip()

        # Compute log-probs via forward pass over continuation tokens only.
        # Using generate() scores would cause a mismatch with score() because
        # generate() applies logit processors (top-k, temperature) that alter
        # the score distribution. A clean forward pass matches exactly.
        action_logp = self._forward_logp(model, enc.input_ids, action_ids, len(prefix_ids), device)

        return action_text, action_ids, action_logp, last_hidden, len(prefix_ids)

    @torch.no_grad()
    def _forward_logp(
        self, model, prompt_ids: torch.Tensor, action_ids: List[int],
        prefix_len: int, device: torch.device,
    ) -> float:
        """
        Compute old_log_probs via a clean forward pass (no generate() scores).

        This ensures old_log_probs are computed identically to new_log_probs
        in score(), avoiding mismatches from logit processors in generate().
        Only scores continuation tokens (skips prefix_len tokens).
        """
        action_tensor = torch.tensor([action_ids], device=device)
        full_ids = torch.cat([prompt_ids, action_tensor], dim=1)
        attention_mask = torch.ones_like(full_ids)
        prompt_len = prompt_ids.shape[1]

        out = model(input_ids=full_ids, attention_mask=attention_mask)

        score_start = prompt_len - 1 + prefix_len
        cont_tokens = action_tensor[0][prefix_len:]
        logits = out.logits[0, score_start: -1, :]
        logp = _token_logp_sum(logits, cont_tokens)
        return logp.item()

    def score(
        self,
        prompt: str,
        action_ids: List[int],
        prefix_len: int = 0,
    ) -> Tuple[float, torch.Tensor, float]:
        """
        Re-score a (prompt, action) pair during PPO update (runs on GPU 0).

        Args:
            prompt:     full text prompt string
            action_ids: token IDs of the action to score
            prefix_len: number of prefix tokens to skip when computing log-probs
                        (must match the prefix_len from generate() so old and new
                        log-probs cover the same continuation tokens)

        Returns:
            new_logp:     sum of log-probs of continuation tokens under current policy
            state_hidden: hidden state of last prompt token (for value heads)
                          shape: (hidden_dim,) on train_device
            ref_logp:     sum of log-probs of continuation tokens under frozen reference
        """
        self.policy.train()

        enc_prompt = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_seq_len,
            truncation=True,
            padding=False,
            add_special_tokens=False,  # chat template already includes special tokens
        ).to(self.train_device)

        prompt_ids = enc_prompt.input_ids  # (1, prompt_len)
        action_tensor = torch.tensor(action_ids, device=self.train_device).unsqueeze(0)
        full_ids = torch.cat([prompt_ids, action_tensor], dim=1)
        attention_mask = torch.ones_like(full_ids)
        prompt_len = prompt_ids.shape[1]

        # Offset into action tokens: skip prefix, score only continuation
        score_start = prompt_len - 1 + prefix_len   # logit index for first continuation token
        cont_tokens = action_tensor[0][prefix_len:]  # continuation token IDs

        # Policy forward pass
        policy_out = self.policy(
            input_ids=full_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Hidden state at last prompt token (float32 to match value heads)
        state_hidden = policy_out.hidden_states[-1][0, prompt_len - 1, :].float()  # (hidden_dim,)

        # Log-probs of continuation tokens only (skip prefix)
        logits = policy_out.logits[0, score_start: -1, :]  # (cont_len, vocab)
        new_logp = _token_logp_sum(logits, cont_tokens)

        # Reference model log-probs (no grad, GPU 1)
        with torch.no_grad():
            ref_enc_prompt = enc_prompt.to(self.rollout_device)
            ref_action = action_tensor.to(self.rollout_device)
            ref_full = torch.cat([ref_enc_prompt.input_ids, ref_action], dim=1)
            ref_mask = torch.ones_like(ref_full)
            ref_out = self.reference(input_ids=ref_full, attention_mask=ref_mask)
            ref_logits = ref_out.logits[0, score_start: -1, :].to(self.train_device)
            ref_logp = _token_logp_sum(ref_logits, cont_tokens.to(self.train_device)).detach()

        return new_logp, state_hidden, ref_logp

    def save_lora(self, path: str):
        """Save LoRA adapter weights to disk."""
        self.policy.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_lora(self, path: str):
        """Load LoRA adapter weights in-place (preserves object identity for optimizer)."""
        import os
        from peft import set_peft_model_state_dict

        safetensors_path = os.path.join(path, "adapter_model.safetensors")
        bin_path = os.path.join(path, "adapter_model.bin")
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path)
        else:
            state_dict = torch.load(bin_path, map_location="cpu")
        set_peft_model_state_dict(self.policy, state_dict)


# ── Helper functions ──────────────────────────────────────────────────────────



def _compute_action_logp(scores, action_ids: List[int], temperature: float = 1.0) -> float:
    """
    Compute sum of log-probs from model.generate() scores output.

    scores: tuple of (vocab_size,) tensors, one per generated token.
    NOTE: model.generate() returns scores that have already been divided by
    temperature (and passed through other logit processors). We must undo the
    temperature scaling to recover the true model log-probabilities, otherwise
    old_log_probs and new_log_probs (from score()) will be on different scales,
    inflating the apparent policy ratio and approx_kl.
    """
    total = 0.0
    for score, tok_id in zip(scores, action_ids):
        # Undo temperature scaling: score = logit / T  →  logit = score * T
        raw_logits = score[0] * temperature
        log_probs = F.log_softmax(raw_logits, dim=-1)
        total += log_probs[tok_id].item()
    return total


def _token_logp_sum(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """
    Compute sum of log-probs for a sequence of token IDs.

    Args:
        logits:    (seq_len, vocab_size) — shifted so logits[i] predicts token_ids[i]
        token_ids: (seq_len,)

    Returns:
        scalar tensor (gradient-connected)
    """
    log_probs = F.log_softmax(logits, dim=-1)  # (seq_len, vocab_size)
    selected = log_probs[torch.arange(len(token_ids)), token_ids]  # (seq_len,)
    return selected.sum()
