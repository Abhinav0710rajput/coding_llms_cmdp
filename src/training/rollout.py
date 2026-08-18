"""
Rollout collection: runs episodes in parallel (async API calls, sequential GPU)
and stores transitions in a RolloutBuffer for PPO updates.
"""

import asyncio
import random
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from src.data.dataset import Problem, sample_problems
from src.environment.env import ClarificationEnv, EnvState


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Transition:
    """One (state, action) step within an episode."""
    episode_id:   int
    turn:         int
    prompt:       str          # full text prompt at this step
    action_text:  str          # raw agent output
    action_ids:   List[int]    # token IDs of the action
    action_logp:  float        # log-prob under the policy that generated it
    state_hidden: torch.Tensor # last prompt token hidden state (hidden_dim,) on CPU
    reward:       float        # pass@1 (non-zero only on [ANSWER] turns)
    cost_q:       float        # atomic questions this turn
    cost_t:       float        # turn cost (1.0 per turn)
    done:         bool
    info:         dict = field(default_factory=dict)


@dataclass
class Episode:
    """All transitions from one complete episode."""
    episode_id: int
    problem_id: str
    transitions: List[Transition]
    total_reward: float          # sum of rewards (= final pass@1)
    total_cost_q: float          # total questions asked
    total_cost_t: float          # total turns used
    n_turns: int


class RolloutBuffer:
    """
    Stores all transitions from a batch of episodes.
    After collection, compute_returns() populates advantage arrays for PPO.
    """

    def __init__(self):
        self.episodes: List[Episode] = []
        self.transitions: List[Transition] = []

        # Populated by compute_returns()
        self.returns_r:    List[float] = []
        self.returns_q:    List[float] = []
        self.returns_t:    List[float] = []
        self.advantages_r: List[float] = []
        self.advantages_q: List[float] = []
        self.advantages_t: List[float] = []

    def add_episode(self, episode: Episode):
        self.episodes.append(episode)
        self.transitions.extend(episode.transitions)

    def __len__(self):
        return len(self.transitions)

    def compute_returns(
        self,
        value_heads,
        gamma: float = 1.0,
        gae_lambda: float = 0.95,
    ):
        """
        Compute GAE advantages for all three return streams.
        Must be called after all episodes are added.

        Args:
            value_heads: ThreeHeads module (on train device)
            gamma:       discount factor (1.0 for short episodic tasks)
            gae_lambda:  GAE lambda
        """
        self.returns_r.clear()
        self.returns_q.clear()
        self.returns_t.clear()
        self.advantages_r.clear()
        self.advantages_q.clear()
        self.advantages_t.clear()

        for episode in self.episodes:
            _compute_episode_advantages(
                episode.transitions,
                value_heads,
                gamma,
                gae_lambda,
                self.returns_r,
                self.returns_q,
                self.returns_t,
                self.advantages_r,
                self.advantages_q,
                self.advantages_t,
            )

    def stats(self) -> dict:
        """Summary statistics over the batch (for logging)."""
        n_ep = len(self.episodes)
        if n_ep == 0:
            return {}
        avg_reward = sum(e.total_reward for e in self.episodes) / n_ep
        avg_q = sum(e.total_cost_q for e in self.episodes) / n_ep
        avg_t = sum(e.total_cost_t for e in self.episodes) / n_ep
        return {
            "n_episodes": n_ep,
            "avg_reward": avg_reward,
            "avg_questions": avg_q,
            "avg_turns": avg_t,
        }


# ── GAE computation ───────────────────────────────────────────────────────────

def _compute_episode_advantages(
    transitions: List[Transition],
    value_heads,
    gamma: float,
    gae_lambda: float,
    out_ret_r, out_ret_q, out_ret_t,
    out_adv_r, out_adv_q, out_adv_t,
):
    """
    Compute GAE advantages for one episode in-place into the output lists.
    """
    T = len(transitions)
    device = next(value_heads.parameters()).device

    # Get value estimates for each state (cast to float32 — model outputs bfloat16)
    hiddens = torch.stack([t.state_hidden for t in transitions]).to(device).float()
    with torch.no_grad():
        v_r, v_q, v_t = value_heads.predict_all(hiddens)

    v_r = v_r.cpu().tolist()
    v_q = v_q.cpu().tolist()
    v_t = v_t.cpu().tolist()

    rewards = [t.reward  for t in transitions]
    costs_q = [t.cost_q  for t in transitions]
    costs_t = [t.cost_t  for t in transitions]
    dones   = [t.done    for t in transitions]

    # Bootstrap value at T+1 is 0 (episode ends)
    v_r.append(0.0)
    v_q.append(0.0)
    v_t.append(0.0)

    adv_r, adv_q, adv_t = [], [], []
    gae_r = gae_q = gae_t = 0.0

    for t in reversed(range(T)):
        mask = 0.0 if dones[t] else 1.0

        delta_r = rewards[t] + gamma * v_r[t + 1] * mask - v_r[t]
        delta_q = costs_q[t] + gamma * v_q[t + 1] * mask - v_q[t]
        delta_t = costs_t[t] + gamma * v_t[t + 1] * mask - v_t[t]

        gae_r = delta_r + gamma * gae_lambda * mask * gae_r
        gae_q = delta_q + gamma * gae_lambda * mask * gae_q
        gae_t = delta_t + gamma * gae_lambda * mask * gae_t

        adv_r.append(gae_r)
        adv_q.append(gae_q)
        adv_t.append(gae_t)

    adv_r.reverse()
    adv_q.reverse()
    adv_t.reverse()

    ret_r = [adv_r[t] + v_r[t] for t in range(T)]
    ret_q = [adv_q[t] + v_q[t] for t in range(T)]
    ret_t = [adv_t[t] + v_t[t] for t in range(T)]

    out_ret_r.extend(ret_r)
    out_ret_q.extend(ret_q)
    out_ret_t.extend(ret_t)
    out_adv_r.extend(adv_r)
    out_adv_q.extend(adv_q)
    out_adv_t.extend(adv_t)


# ── Episode runner ────────────────────────────────────────────────────────────

async def _run_episode(
    episode_id: int,
    problem: Problem,
    agent,
    env: ClarificationEnv,
) -> Episode:
    """Run one complete episode asynchronously."""
    import time as _time
    state = env.reset(problem)
    transitions = []

    while not state.done:
        # GPU inference (synchronous — called from async context but not awaited)
        t0 = _time.time()
        action_text, action_ids, action_logp, state_hidden, prefix_len = agent.generate(state.prompt)
        gen_time = _time.time() - t0

        # Environment step (async — may call GPT-4o-mini API)
        t0 = _time.time()
        result = await env.step(state, action_text)
        step_time = _time.time() - t0

        if episode_id < 3:  # Only log timing for first few episodes
            action_type = result.info.get("action_type", "?")
            print(f"    [ep={episode_id} turn={state.turn_count}] "
                  f"{action_type} | gen={gen_time:.1f}s step={step_time:.1f}s "
                  f"| tokens={len(action_ids)} prompt_len={len(state.prompt)}")

        info = {**result.info, "prefix_len": prefix_len}
        transitions.append(Transition(
            episode_id=episode_id,
            turn=state.turn_count,
            prompt=state.prompt,
            action_text=action_text,
            action_ids=action_ids,
            action_logp=action_logp,
            state_hidden=state_hidden,
            reward=result.reward,
            cost_q=result.cost_q,
            cost_t=result.cost_t,
            done=result.done,
            info=info,
        ))

        state = result.next_state

    total_reward = sum(t.reward  for t in transitions)
    total_cost_q = sum(t.cost_q  for t in transitions)
    total_cost_t = sum(t.cost_t  for t in transitions)

    return Episode(
        episode_id=episode_id,
        problem_id=problem.task_id,
        transitions=transitions,
        total_reward=total_reward,
        total_cost_q=total_cost_q,
        total_cost_t=total_cost_t,
        n_turns=len(transitions),
    )


async def collect_rollouts(
    agent,
    env: ClarificationEnv,
    problems: List[Problem],
    batch_size: int,
    rng: random.Random,
) -> RolloutBuffer:
    """
    Collect a full rollout buffer of `batch_size` episodes.

    GPU inference calls are sequential (within each episode coroutine).
    API calls to the user simulator are fully async and run concurrently.

    Args:
        agent:      Agent instance
        env:        ClarificationEnv (shared, stateless)
        problems:   pool of training problems to sample from
        batch_size: number of episodes to collect
        rng:        random state for sampling problems

    Returns:
        RolloutBuffer (without advantages — call compute_returns() separately)
    """
    # Sync rollout model weights before collection
    agent.sync_rollout_model()

    sampled = sample_problems(problems, batch_size, rng)
    tasks = [
        _run_episode(i, problem, agent, env)
        for i, problem in enumerate(sampled)
    ]

    completed = await asyncio.gather(*tasks)

    buffer = RolloutBuffer()
    for episode in completed:
        buffer.add_episode(episode)

    return buffer
