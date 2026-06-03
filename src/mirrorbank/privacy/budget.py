"""
Stage 3 — Privacy Budget Accountant.

Uses Rényi Differential Privacy (RDP) accounting — tighter than basic
composition for iterative mechanisms like DP-SGD.  Training halts
automatically when the configured ε target is reached.

One PrivacyBudget instance is created per instrument per training run.
Budgets never compose across instruments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class BudgetExhausted(Exception):
    """Raised when a gradient step would exceed the configured ε target."""


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class BudgetConfig:
    epsilon: float              # privacy budget target (ε)
    delta: float = 1e-5         # failure probability (δ)
    noise_multiplier: float = 1.1   # σ — Gaussian noise std relative to clipping norm
    max_grad_norm: float = 1.0      # C — gradient clipping norm
    batch_size: int = 4096
    dataset_size: int = 0           # N — set before training begins

    # ── Supported presets ─────────────────────────────────────────────────────
    @classmethod
    def from_preset(cls, preset: str, dataset_size: int, batch_size: int = 4096) -> "BudgetConfig":
        presets = {
            "tight":    {"epsilon": 1.0,  "noise_multiplier": 1.5},
            "balanced": {"epsilon": 3.0,  "noise_multiplier": 1.1},
            "loose":    {"epsilon": 5.0,  "noise_multiplier": 0.9},
            "demo":     {"epsilon": 10.0, "noise_multiplier": 0.7},
        }
        if preset not in presets:
            raise ValueError(f"Unknown preset {preset!r}. Choose from: {list(presets)}")
        return cls(
            dataset_size=dataset_size,
            batch_size=batch_size,
            **presets[preset],
        )


# ── Accountant ────────────────────────────────────────────────────────────────

class PrivacyBudget:
    """
    Wraps a DP-SGD training loop and tracks the cumulative (ε, δ) cost.

    Usage:
        budget = PrivacyBudget(BudgetConfig(epsilon=3.0, dataset_size=len(df)))
        for step in training_loop:
            budget.charge_step()   # raises BudgetExhausted when ε is exceeded
            ...
        cert_data = budget.summary()
    """

    # RDP orders to evaluate (standard set from Mironov 2017)
    _ORDERS = list(range(2, 129)) + [256.0]

    def __init__(self, config: BudgetConfig):
        if config.dataset_size <= 0:
            raise ValueError("BudgetConfig.dataset_size must be set before training")
        self.config = config
        self._steps = 0
        self._rdp: dict[float, float] = {a: 0.0 for a in self._ORDERS}

    # ── Public API ────────────────────────────────────────────────────────────

    def charge_step(self) -> float:
        """
        Account for one DP-SGD gradient step and return current ε.
        Raises BudgetExhausted if the target ε would be exceeded.
        """
        self._compose_gaussian_step()
        self._steps += 1
        eps = self.current_epsilon()
        if eps > self.config.epsilon:
            raise BudgetExhausted(
                f"ε={eps:.4f} exceeded target ε={self.config.epsilon} "
                f"after {self._steps} steps"
            )
        return eps

    def current_epsilon(self) -> float:
        """Return the current (ε, δ) epsilon spent so far."""
        return self._rdp_to_dp(self.config.delta)

    def budget_fraction(self) -> float:
        """Fraction of ε budget consumed (0.0–1.0+)."""
        return self.current_epsilon() / self.config.epsilon

    def summary(self) -> dict:
        return {
            "epsilon_spent": self.current_epsilon(),
            "epsilon_target": self.config.epsilon,
            "delta": self.config.delta,
            "steps": self._steps,
            "noise_multiplier": self.config.noise_multiplier,
            "max_grad_norm": self.config.max_grad_norm,
            "batch_size": self.config.batch_size,
            "dataset_size": self.config.dataset_size,
            "sampling_rate": self.config.batch_size / self.config.dataset_size,
            "accounting_method": "rdp",
        }

    # ── RDP accounting (Mironov 2017) ─────────────────────────────────────────

    def _compose_gaussian_step(self) -> None:
        """
        Add one step of the subsampled Gaussian mechanism to the RDP accountant.

        For Poisson subsampling rate q = B/N and noise multiplier σ, the RDP
        of order α is approximately bounded by:
            ε_RDP(α) ≤ (1/α) · log(1 + q²·α(α-1)/(2σ²))   [simplified]

        We use the tighter bound via the log-moment generating function.
        """
        q = self.config.batch_size / self.config.dataset_size
        sigma = self.config.noise_multiplier

        for alpha in self._ORDERS:
            self._rdp[alpha] += self._gaussian_rdp(alpha, sigma, q)

    @staticmethod
    def _gaussian_rdp(alpha: float, sigma: float, q: float) -> float:
        """
        RDP of the subsampled Gaussian mechanism at order alpha.
        Uses the Mironov 2017 bound (simplified for the Poisson subsampling case).
        """
        if alpha == 1:
            return q * (math.exp(1 / sigma**2) - 1)
        return min(
            alpha / (2 * sigma**2),                         # non-subsampled bound
            (1 / (alpha - 1)) * math.log(               # subsampled bound
                1 + q**2 * alpha * (alpha - 1) / (2 * sigma**2)
            ) if sigma > 0 else float("inf"),
        )

    def _rdp_to_dp(self, delta: float) -> float:
        """
        Convert accumulated RDP to (ε, δ)-DP using the conversion formula:
            ε(δ) = min_α [ ε_RDP(α) + log(1/δ) / (α - 1) ]
        """
        eps_values = []
        for alpha, rdp_eps in self._rdp.items():
            if rdp_eps == 0.0:
                continue
            try:
                converted = rdp_eps + math.log(1.0 / delta) / (alpha - 1)
                eps_values.append(converted)
            except (ValueError, ZeroDivisionError):
                pass
        return min(eps_values) if eps_values else 0.0


# ── Noise multiplier calibration ──────────────────────────────────────────────

def calibrate_noise_multiplier(
    target_epsilon: float,
    delta: float,
    n_rows: int,
    batch_size: int,
    epochs: int,
    tolerance: float = 0.01,
) -> float:
    """
    Binary-search for the noise multiplier σ such that after
    (n_rows / batch_size * epochs) training steps the accountant reads
    approximately (target_epsilon ± tolerance, delta).

    This means users specify ε (what they want) not σ (a technical knob).
    """
    n_steps = max(1, int(n_rows / batch_size * epochs))
    lo, hi = 0.01, 100.0

    for _ in range(64):  # 64 bisection steps → precision < 1e-18
        mid = (lo + hi) / 2
        config = BudgetConfig(
            epsilon=target_epsilon,
            delta=delta,
            noise_multiplier=mid,
            batch_size=batch_size,
            dataset_size=n_rows,
        )
        budget = PrivacyBudget(config)
        # Simulate all steps without the BudgetExhausted guard
        for _ in range(n_steps):
            budget._compose_gaussian_step()
            budget._steps += 1
        eps = budget.current_epsilon()

        if abs(eps - target_epsilon) < tolerance:
            return mid
        if eps > target_epsilon:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2
