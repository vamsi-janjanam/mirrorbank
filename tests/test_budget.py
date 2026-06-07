"""Tests for Stage 3 — PrivacyBudget accountant."""

import pytest

from mirrorbank.privacy.budget import (
    BudgetConfig,
    BudgetExhausted,
    PrivacyBudget,
    calibrate_noise_multiplier,
)


def make_budget(epsilon=3.0, noise_multiplier=1.1, n=10_000, batch=256):
    config = BudgetConfig(
        epsilon=epsilon,
        noise_multiplier=noise_multiplier,
        dataset_size=n,
        batch_size=batch,
    )
    return PrivacyBudget(config)


# ── Basic behaviour ───────────────────────────────────────────────────────────


def test_budget_starts_at_zero():
    budget = make_budget()
    assert budget.current_epsilon() == 0.0
    assert budget.budget_fraction() == 0.0


def test_budget_increases_monotonically():
    budget = make_budget()
    prev = 0.0
    for _ in range(50):
        budget.charge_step()
        eps = budget.current_epsilon()
        assert eps >= prev
        prev = eps


def test_budget_fraction_increases_with_steps():
    budget = make_budget()
    for _ in range(20):
        budget.charge_step()
    assert 0.0 < budget.budget_fraction() <= 1.0


def test_budget_exhausted_raises():
    # Very tight budget — should exhaust quickly
    budget = make_budget(epsilon=0.01, noise_multiplier=0.1, n=1000, batch=500)
    with pytest.raises(BudgetExhausted):
        for _ in range(10_000):
            budget.charge_step()


def test_budget_exhausted_message_contains_epsilon():
    budget = make_budget(epsilon=0.01, noise_multiplier=0.1, n=1000, batch=500)
    with pytest.raises(BudgetExhausted, match="ε="):
        for _ in range(10_000):
            budget.charge_step()


# ── Summary ───────────────────────────────────────────────────────────────────


def test_budget_summary_keys():
    budget = make_budget()
    budget.charge_step()
    summary = budget.summary()
    required_keys = {
        "epsilon_spent",
        "epsilon_target",
        "delta",
        "steps",
        "noise_multiplier",
        "accounting_method",
        "max_grad_norm",
        "batch_size",
        "dataset_size",
        "sampling_rate",
    }
    assert required_keys.issubset(summary.keys())
    assert summary["steps"] == 1
    assert summary["accounting_method"] == "rdp"


def test_budget_summary_sampling_rate():
    config = BudgetConfig(epsilon=3.0, dataset_size=10_000, batch_size=500)
    budget = PrivacyBudget(config)
    assert budget.summary()["sampling_rate"] == pytest.approx(0.05)


# ── Presets ───────────────────────────────────────────────────────────────────


def test_preset_tight():
    config = BudgetConfig.from_preset("tight", dataset_size=50_000)
    assert config.epsilon == 1.0
    assert config.noise_multiplier == 1.5


def test_preset_balanced():
    config = BudgetConfig.from_preset("balanced", dataset_size=50_000)
    assert config.epsilon == 3.0
    assert config.noise_multiplier == 1.1


def test_preset_loose():
    config = BudgetConfig.from_preset("loose", dataset_size=50_000)
    assert config.epsilon == 5.0
    assert config.noise_multiplier == 0.9


def test_preset_demo():
    config = BudgetConfig.from_preset("demo", dataset_size=50_000)
    assert config.epsilon == 10.0
    assert config.noise_multiplier == 0.7


def test_preset_unknown_raises():
    with pytest.raises(ValueError, match="Unknown preset"):
        BudgetConfig.from_preset("nonexistent", dataset_size=1000)


def test_preset_batch_size_forwarded():
    config = BudgetConfig.from_preset("balanced", dataset_size=50_000, batch_size=512)
    assert config.batch_size == 512


# ── Guard rails ───────────────────────────────────────────────────────────────


def test_dataset_size_zero_raises():
    config = BudgetConfig(epsilon=3.0, dataset_size=0)
    with pytest.raises(ValueError, match="dataset_size"):
        PrivacyBudget(config)


def test_dataset_size_negative_raises():
    config = BudgetConfig(epsilon=3.0, dataset_size=-100)
    with pytest.raises(ValueError, match="dataset_size"):
        PrivacyBudget(config)


# ── Noise multiplier calibration ──────────────────────────────────────────────


def test_calibrate_noise_multiplier_returns_float():
    sigma = calibrate_noise_multiplier(
        target_epsilon=3.0,
        delta=1e-5,
        n_rows=10_000,
        batch_size=256,
        epochs=10,
    )
    assert isinstance(sigma, float)
    assert 0.1 < sigma < 10.0


def test_calibrate_noise_multiplier_produces_target_epsilon():
    """Calibrated σ should yield actual ε within tolerance of the target."""
    target_eps = 3.0
    tolerance = 0.05
    sigma = calibrate_noise_multiplier(
        target_epsilon=target_eps,
        delta=1e-5,
        n_rows=10_000,
        batch_size=256,
        epochs=10,
        tolerance=tolerance,
    )
    # Simulate the same run and confirm actual ε is close to target
    config = BudgetConfig(
        epsilon=target_eps + 1,  # set high so we don't halt early
        delta=1e-5,
        noise_multiplier=sigma,
        batch_size=256,
        dataset_size=10_000,
    )
    budget = PrivacyBudget(config)
    n_steps = int(10_000 / 256 * 10)
    for _ in range(n_steps):
        budget._compose_gaussian_step()
        budget._steps += 1
    actual_eps = budget.current_epsilon()
    assert abs(actual_eps - target_eps) < tolerance * 3  # generous bound


def test_calibrate_tighter_epsilon_gives_larger_sigma():
    """Tighter privacy budget requires more noise (larger σ)."""
    sigma_tight = calibrate_noise_multiplier(1.0, 1e-5, 10_000, 256, 10)
    sigma_loose = calibrate_noise_multiplier(5.0, 1e-5, 10_000, 256, 10)
    assert sigma_tight > sigma_loose
