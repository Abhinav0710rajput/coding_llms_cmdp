"""
Lagrange multiplier dual update for the constrained clarification CMDP.

The dual problem:
    λ₁ ← max(0, λ₁ + lr_λ × (avg_questions - d₁))
    λ₂ ← max(0, λ₂ + lr_λ₂ × (avg_turns - d₂))

When avg_questions > d₁: λ₁ increases → questions become more expensive.
When avg_questions < d₁: λ₁ decreases → questions become cheaper.
Over training, this drives avg_questions → d₁.
"""


class LagrangeMultiplier:
    """
    Manages one Lagrange multiplier for one inequality constraint.

    Constraint:  E[cost] ≤ budget
    Update:      λ ← clip(λ + lr × (avg_cost - budget), 0, lambda_max)
    """

    def __init__(self, init: float, lr: float, budget: float, max_val: float):
        self.value = float(init)
        self.lr = float(lr)
        self.budget = float(budget)
        self.max_val = float(max_val)
        self._history: list = []   # for logging / plotting

    def update(self, avg_cost: float) -> float:
        """
        Perform one dual ascent step.

        Args:
            avg_cost: mean cost over the current rollout batch

        Returns:
            new lambda value
        """
        violation = avg_cost - self.budget
        self.value = max(0.0, min(self.max_val, self.value + self.lr * violation))
        self._history.append({
            "avg_cost": avg_cost,
            "budget": self.budget,
            "violation": violation,
            "lambda": self.value,
        })
        return self.value

    def state_dict(self) -> dict:
        return {
            "value": self.value,
            "lr": self.lr,
            "budget": self.budget,
            "max_val": self.max_val,
        }

    def load_state_dict(self, d: dict):
        self.value = d["value"]
        self.lr = d["lr"]
        self.budget = d["budget"]
        self.max_val = d["max_val"]

    def __repr__(self):
        return f"LagrangeMultiplier(λ={self.value:.4f}, budget={self.budget}, lr={self.lr})"


class DualVariables:
    """
    Holds both Lagrange multipliers (λ₁ for questions, λ₂ for turns).
    Provides a single update() call and convenient property access.
    """

    def __init__(self, cfg):
        self.lambda1 = LagrangeMultiplier(
            init=cfg.constraint.lambda_init,
            lr=cfg.constraint.lr_lambda,
            budget=cfg.constraint.d1,
            max_val=cfg.constraint.lambda_max,
        )
        self.lambda2 = LagrangeMultiplier(
            init=cfg.constraint.lambda2_init,
            lr=cfg.constraint.lr_lambda2,
            budget=cfg.constraint.d2,
            max_val=cfg.constraint.lambda2_max,
        )

    def update(self, avg_questions: float, avg_turns: float):
        """
        Update both multipliers given batch averages.

        Args:
            avg_questions: mean total atomic questions per episode
            avg_turns:     mean total turns per episode
        """
        l1 = self.lambda1.update(avg_questions)
        l2 = self.lambda2.update(avg_turns)
        return l1, l2

    @property
    def l1(self) -> float:
        return self.lambda1.value

    @property
    def l2(self) -> float:
        return self.lambda2.value

    def state_dict(self) -> dict:
        return {
            "lambda1": self.lambda1.state_dict(),
            "lambda2": self.lambda2.state_dict(),
        }

    def load_state_dict(self, d: dict):
        self.lambda1.load_state_dict(d["lambda1"])
        self.lambda2.load_state_dict(d["lambda2"])
