# Pareto-Dominant Clarification: Post-Training Coding LLMs via PPO-Lagrangian Budget Constraints

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch 2.3+](https://img.shields.io/badge/PyTorch-2.3%2B-ee4c2c.svg)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-orange)](https://huggingface.co/acv1229)

## Overview

This repository implements PPO-Lagrangian post-training for budget-constrained ask-vs-answer routing in coding LLMs, evaluated on HumanEvalComm. The agent learns when to ask clarifying questions on degraded coding specifications, subject to an explicit per-episode question-count budget enforced via a Lagrange multiplier updated by dual ascent.

**Authors:** [Abhinav Rajput](https://github.com/Abhinav0710rajput) and [Acey Vogelstein](https://github.com/acv1229)<br>
*[NYU Center for Data Science](https://cds.nyu.edu/)*

## Repository structure

```
src/
  data/           # HumanEvalComm dataset loader and degradation handling
    dataset.py    # Problem dataclass, load_humaneval_comm()
    augmentation.py
  environment/    # ClarificationEnv RL state machine, user simulator, code executor
    env.py        # Episode state machine (ask / answer / done transitions)
    user_simulator.py  # Async GPT-4o-mini wrapper
    code_executor.py   # Sandboxed Python executor → pass@1 score
  models/         # Qwen2.5-Coder-7B + LoRA policy and value heads
    agent.py      # Agent class: generate, score, forward
    value_heads.py  # ThreeHeads MLP (reward, question-cost, turn-cost)
  training/       # PPO loss, GAE, Lagrangian dual update, rollout buffer
    trainer.py    # PPOLagrangianTrainer main loop
    ppo.py        # PPO surrogate loss, GAE, KL penalty, entropy bonus
    lagrangian.py # Dual variable update (λ₁, λ₂)
    rollout.py    # Async episode collection, RolloutBuffer
  evaluation/
    evaluator.py  # pass@1 metrics, Pareto frontier utilities

scripts/
  train.py              # Training entry point
  evaluate.py           # Evaluation entry point
  baseline_eval.py      # Untrained baseline evaluation
  checkpoint_val_eval.py  # Val-set checkpoint selection
  generate_splits.py    # Reproduce train/validation/test task-ID files
  smoke_test.py           # End-to-end pipeline validation

configs/
  default.yaml          # All hyperparameters from the paper

data/
  HumanEvalComm_v2.csv  # Local cache of the HumanEvalComm dataset
  processed/             # Explicit deterministic split task IDs

results/
  figures/              # Pre-generated PNG figures

logs/
  checkpoint_val_*.json  # Val-set scores per policy (used for checkpoint selection)
  final_eval_*.jsonl     # Per-episode evaluation records for all policies
```

## Requirements

- Python 3.10+
- CUDA-capable GPU(s). Training was conducted on 2×A100-40GB.
- An OpenAI API key (`OPENAI_API_KEY` environment variable) is required to run the GPT-4o-mini user simulator during training and evaluation.

## Hardware requirements

| Requirement | Minimum | Recommended |
| --- | --- | --- |
| GPUs | 2× A100-40GB | 2× A100-40GB |
| CPU RAM | 64 GB | 128 GB |
| Disk | 50 GB free | 100 GB free |
| Internet | Required (Hugging Face and OpenAI API) | Not applicable |

**GPU layout:**

- `cuda:0`: Policy training (Qwen2.5-Coder-7B + LoRA + value heads + optimizer, ~19 GB)
- `cuda:1`: Rollout inference + frozen reference model (~17 GB)

To change GPU assignment, edit `model.train_device` and `model.rollout_device` in `configs/default.yaml`.

## Setup

```bash
git clone https://github.com/Abhinav0710rajput/coding_llms_cmdp.git
cd coding_llms_cmdp

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

export OPENAI_API_KEY=<your-key>
```

**Install CUDA-enabled PyTorch separately if needed** (the version in `requirements.txt` is CPU-safe but may not match your CUDA driver):

```bash
# Example for CUDA 12.1
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
```

**Download the base model** (Qwen2.5-Coder-7B-Instruct, publicly available, no token required):

```bash
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-7B-Instruct')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-7B-Instruct', torch_dtype='bfloat16')
"
```

## Trained model checkpoints

The trained policies are available on Hugging Face:

| Budget | Model | Description |
| --- | --- | --- |
| `d1=0` | [![](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/acv1229/rl-clarify-orig-prompt-d1-0) | Never asks; guesses from the degraded specification alone |
| `d1=0.5` | [![](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/acv1229/rl-clarify-orig-prompt-d1-0p5) | Asks sparingly; averages at most 0.5 questions |
| `d1=1` | [![](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/acv1229/rl-clarify-orig-prompt-d1-1) | Asks when useful; averages at most 1 question |

### Local checkpoint structure

All checkpoints are saved under `checkpoints/` (gitignored):

```
checkpoints/
├── d1_0/
│   ├── iter_0019/        
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors   ← LoRA weights (~80 MB)
│   │   ├── tokenizer.json
│   │   ├── value_heads.pt              ← Three MLP heads
│   │   ├── dual_variables.pt           ← λ₁, λ₂ values
│   │   ├── train_state.pt              ← Optimizer + scheduler state
│   │   └── log.json                    ← Training log up to this point
│   ├── iter_0039/
│   ├── best/               ← Overwritten whenever mid-training eval reward improves
│   └── final/              ← Checkpoint after the last training iteration
└── d1_1/
    ├── iter_0019/
    ├── best/
    └── final/
```

Pre-computed evaluation logs for all 10 trained policies and the baseline are available in `logs/final_eval_*.jsonl`.

## Training

Train a single policy at question-count budget `d1`:

```bash
python scripts/train.py --d1 <budget>
```

The paper trains 10 policies across seven budget levels (`d1 ∈ {0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0}`), with two independent runs at `d1 ∈ {0.5, 0.75, 1.0}`. Each run takes approximately 11 hours on 2×A100-40GB. Any config value can be overridden via OmegaConf dot-notation:

```bash
python scripts/train.py --d1 0.75 training.n_iterations=80 training.rollout_batch_size=32
```

To resume a run from a checkpoint:

```bash
python scripts/train.py --d1 1.0 --resume checkpoints/orig_prompt/d1_1.0/iter_0039
```

## Evaluation

Evaluate a trained checkpoint on the 417-problem held-out test set:

```bash
python scripts/evaluate.py --checkpoint <path-to-checkpoint>
```

Full sweep over all checkpoints (reproduces the Pareto frontier):

```bash
python scripts/evaluate.py --sweep --output_dir outputs/pareto
```

### Checkpoint selection

Select the best budget-feasible checkpoint per policy using the 52-problem val set:

```bash
python scripts/checkpoint_val_eval.py --policy d1_0.75 --budget 0.75
```

Pre-computed val-set scores are in `logs/checkpoint_val_*.json`.

### Baseline evaluation

Evaluate the untrained Qwen2.5-Coder-7B-Instruct baseline (single-turn and multi-turn):

```bash
python scripts/baseline_eval.py
```

### Pipeline validation

Validate the end-to-end pipeline with a single episode before a full training run:

```bash
python scripts/smoke_test.py
```

## Dataset

The dataset is **HumanEvalComm**: 164 Python coding problems from HumanEval, each with multiple degraded versions of the problem specification.

Training and evaluation automatically download the dataset from Hugging Face. The repository also includes the exact CSV used for the reported experiments at `data/HumanEvalComm_v2.csv`.

**What the degradations look like:**

| Type | Field | What changes |
| --- | --- | --- |
| Ambiguity | `prompt1a` | Specific values replaced with vague terms ("by 1" → "by a number") |
| Inconsistency | `prompt1c` | Examples contradict the description |
| Incompleteness | `prompt1p` | All examples and details stripped; only a stub remains |
| Ambiguity + Inconsistency | `prompt2ac` | Both combined |
| Ambiguity + Incompleteness | `prompt2ap` | Both combined |
| Inconsistency + Incompleteness | `prompt2cp` | Both combined |
| All three | `prompt3acp` | Ambiguity + Inconsistency + Incompleteness |

**Train/test split:** The split is **stratified** at the base problem level. Problems are grouped by their rarest available variant, then each group is split proportionally (~60% eval, ~40% train). This guarantees all 7 degradation types appear in both train and eval sets. All variants of a base problem go to the same set (no leakage). Enforced in `src/data/dataset.py`.

The explicit task-ID files are committed under `data/processed/`. Regenerate them deterministically from the saved CSV with:

```bash
python scripts/generate_splits.py
```

This uses the defaults in `configs/default.yaml` (`seed: 42`, `eval_size: 100`, and all seven degradation variants), followed by the checkpoint-selection validation seed 99. It produces 302 training IDs, 52 validation IDs, and 417 test IDs. Validation and test are disjoint subsets of the 469 held-out degraded instances; the train/held-out division remains at base-problem level (64/100 base problems).

**Which variants are used for training** is controlled by `data.use_variants` in the config:

```yaml
data:
  use_variants:
    - prompt1a
    - prompt1c
    - prompt1p
    - prompt2ac
    - prompt2ap
    - prompt2cp
    - prompt3acp
```

## Configuration

All hyperparameters are in `configs/default.yaml`. Values correspond to the hyperparameter table in the paper. Key parameters:

| Parameter | Key in YAML | Default |
| --- | --- | --- |
| Question budget (hard constraint) | `constraint.d1` | `1.0` |
| Turn budget (soft constraint) | `constraint.d2` | `4` |
| Training iterations | `training.n_iterations` | `80` |
| Rollout batch size | `training.rollout_batch_size` | `32` |
| PPO clip ε | `training.clip_epsilon` | `0.2` |
| KL penalty coefficient | `training.kl_coeff` | `0.25` |
| Base model | `model.name` | `Qwen/Qwen2.5-Coder-7B-Instruct` |

## License

Apache License 2.0. See [LICENSE](LICENSE).
