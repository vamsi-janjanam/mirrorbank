"""Tests for Stage 4 — membership inference audit and gauntlet integration."""

from mirrorbank.evaluate.gauntlet import run_gauntlet
from mirrorbank.evaluate.mia_audit import run_mia
from mirrorbank.instruments.registry import get_schema
from mirrorbank.sample_data import sample_dataset


def test_run_mia_returns_report():
    real = sample_dataset("ach")
    schema = get_schema("ach")

    report = run_mia(real, real, schema, seed=0)

    assert report is not None
    assert 0.0 <= report.attack_auc <= 1.0
    assert isinstance(report.pass_privacy, bool)


def test_run_gauntlet_backward_compatible():
    real = sample_dataset("ach")
    synth = sample_dataset("ach")
    schema = get_schema("ach")

    report = run_gauntlet(real, synth, schema, run_utility=False, run_privacy=False)

    assert report.utility is None
    assert report.privacy is None
    assert report.pass_overall == report.fidelity.pass_overall


def test_run_gauntlet_with_utility_and_privacy_enabled():
    real = sample_dataset("ach")
    synth = sample_dataset("ach")
    schema = get_schema("ach")

    report = run_gauntlet(real, synth, schema)

    # ACH sample has too few fraud positives, so utility should be None,
    # but privacy (MIA) should produce a report.
    assert report.utility is None
    assert report.privacy is not None
    assert 0.0 <= report.privacy.attack_auc <= 1.0
