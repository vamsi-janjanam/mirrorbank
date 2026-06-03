"""Tests for Stage 3 — PrivacyBudget accountant."""

import pytest

from mirrorbank.privacy.budget import BudgetConfig, BudgetExhausted, PrivacyBudget, calibrate_noise_multiplier


def make_budget(epsilon=3.0, noise_multiplier=1.1, n=10_000, batch=256):
    config = BudgetConfig(
        epsilon=epsilon,
        noise_multiplier=noise_multiplier,
        dataset_size=n,
        batch_size=batch,
    )
    return PrivacyBudget(config)


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


def test_budget_exhausted_raises():
    # Very tight budget — should exhaust quickly
    budget = make_budget(epsilon=0.01, noise_multiplier=0.1, n=1000, batch=500)
    with pytest.raises(BudgetExhausted):
        for _ in range(10_000):
            budget.charge_step()


def test_budget_summary_keys():
    budget = make_budget()
    budget.charge_step()
    summary = budget.summary()
    required_keys = {
        "epsilon_spent", "epsilon_target", "delta",
        "steps", "noise_multiplier", "accounting_method",
    }
    assert required_keys.issubset(summary.keys())
    assert summary["steps"] == 1
    assert summary["accounting_method"] == "rdp"


def test_preset_balanced():
    config = BudgetConfig.from_preset("balanced", dataset_size=50_000)
    assert config.epsilon == 3.0
    assert config.noise_multiplier == 1.1


def test_preset_unknown_raises():
    with pytest.raises(ValueError, match="Unknown preset"):
        BudgetConfig.from_preset("nonexistent", dataset_size=1000)


def test_dataset_size_zero_raises():
    config = BudgetConfig(epsilon=3.0, dataset_size=0)
    with pytest.raises(ValueError, match="dataset_size"):
        PrivacyBudget(config)


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
