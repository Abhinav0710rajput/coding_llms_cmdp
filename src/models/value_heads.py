"""
Three independent MLP value heads that predict expected future returns
from the agent's hidden state (last token embedding from the LLM).

  ValueHead_reward  → predicts expected future pass@1
  ValueHead_q_cost  → predicts expected future atomic questions to be asked
  ValueHead_t_cost  → predicts expected future turns to be used
"""

import torch
import torch.nn as nn
from typing import Tuple


class ValueHead(nn.Module):
    """Single MLP value head: hidden_dim → 1."""

    def __init__(self, input_dim: int = 4096, hidden_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        # Init output layer near zero to avoid large initial value estimates
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: (batch, input_dim)  or  (input_dim,)
        Returns:
            values: (batch,)  or  scalar
        """
        return self.net(hidden_state).squeeze(-1)


class ThreeHeads(nn.Module):
    """
    Wrapper that holds all three value heads and provides a single forward call.
    Placed on train_device (GPU 0) alongside the policy model.
    """

    def __init__(self, input_dim: int = 4096, hidden_dim: int = 1024):
        super().__init__()
        self.reward_head = ValueHead(input_dim, hidden_dim)
        self.q_cost_head = ValueHead(input_dim, hidden_dim)
        self.t_cost_head = ValueHead(input_dim, hidden_dim)

    def forward(
        self, hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_state: (batch, input_dim)

        Returns:
            (v_reward, v_q_cost, v_t_cost) — each shape (batch,)
        """
        return (
            self.reward_head(hidden_state),
            self.q_cost_head(hidden_state),
            self.t_cost_head(hidden_state),
        )

    def predict_all(
        self, hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Alias for forward; detaches gradients (inference use)."""
        with torch.no_grad():
            return self.forward(hidden_state)
