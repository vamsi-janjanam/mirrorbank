"""Tests for Stage 4 — fidelity evaluation and gauntlet orchestrator."""

import polars as pl
import pytest

from mirrorbank.evaluate.fidelity import run_fidelity
from mirrorbank.evaluate.gauntlet import run_gauntlet
from mirrorbank.instruments.ach import ACHSchema


# ── run_fidelity ──────────────────────────────────────────────────────────────

def test_fidelity_identical_data_passes(ach_df):
    """A dataset compared to itself should pass all fidelity checks."""
    report = run_fidelity(ach_df, ach_df, schema=ACHSchema())
    assert report.ks_pass_rate == 1.0
    assert report.corr_distance == 0.0


def test_fidelity_completely_different_data_fails(ach_df):
    """Constant columns should fail KS tests against the real data."""
    constant_df = ach_df.with_columns(
        pl.lit(0.0).alias("amount"),
        pl.lit(0).alias("risk_score"),
        pl.lit(0).alias("days_to_settle"),
    )
    report = run_fidelity(ach_df, constant_df, schema=ACHSchema())
    assert report.ks_pass_rate < 1.0


def test_fidelity_report_has_all_fields(ach_df):
    report = run_fidelity(ach_df, ach_df)
    assert isinstance(report.ks_results, dict)
    assert isinstance(report.ks_pass_rate, float)
    assert isinstance(report.corr_distance, float)
    assert isinstance(report.instrument_errors, list)
    assert isinstance(report.pass_overall, bool)


def test_fidelity_summary_string(ach_df):
    report = run_fidelity(ach_df, ach_df)
    summary = report.summary
    assert "PASS" in summary or "FAIL" in summary


def test_fidelity_ks_results_keyed_by_column(ach_df):
    report = run_fidelity(ach_df, ach_df, schema=ACHSchema())
    # KS results should be keyed by column name, each with p_value and pass
    assert len(report.ks_results) > 0
    first = next(iter(report.ks_results.values()))
    assert "p_value" in first
    assert "statistic" in first
    assert "pass" in first


def test_fidelity_without_schema(ach_df):
    """run_fidelity works without an instrument schema (falls back to heuristics)."""
    report = run_fidelity(ach_df, ach_df, schema=None)
    assert isinstance(report.ks_pass_rate, float)


def test_fidelity_instrument_errors_propagated(wire_df):
    """Instrument validation errors are included in the fidelity report."""
    from datetime import datetime
    from mirrorbank.instruments.wire import WireSchema

    weekend_row = wire_df.head(1).with_columns(
        pl.lit(datetime(2024, 1, 6, 10, 0)).alias("timestamp")
    )
    bad_synth = pl.concat([wire_df, weekend_row])
    report = run_fidelity(wire_df, bad_synth, schema=WireSchema())
    assert len(report.instrument_errors) > 0
    assert not report.pass_overall


# ── run_gauntlet ──────────────────────────────────────────────────────────────

def test_gauntlet_returns_report(ach_df):
    """run_gauntlet returns a GauntletReport with the right instrument name."""
    report = run_gauntlet(ach_df, ach_df, schema=ACHSchema())
    assert report.instrument == "ach"
    assert isinstance(report.pass_overall, bool)


def test_gauntlet_identical_data_passes(ach_df):
    # Use data that satisfies the ACH validator: no returned rows without return_code
    clean = ach_df.with_columns(pl.lit(False).alias("is_returned"))
    report = run_gauntlet(clean, clean, schema=ACHSchema())
    assert report.fidelity.ks_pass_rate == 1.0
    assert report.pass_overall is True


def test_gauntlet_print_summary_does_not_raise(ach_df, capsys):
    report = run_gauntlet(ach_df, ach_df, schema=ACHSchema())
    report.print_summary()
    captured = capsys.readouterr()
    assert "ACH" in captured.out
    assert "PASS" in captured.out or "FAIL" in captured.out


def test_gauntlet_without_schema(ach_df):
    """run_gauntlet works without a schema; instrument reported as 'unknown'."""
    report = run_gauntlet(ach_df, ach_df, schema=None)
    assert report.instrument == "unknown"
