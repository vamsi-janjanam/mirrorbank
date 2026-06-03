"""Tests for Stage 4 — fidelity evaluation."""

import polars as pl
import pytest

from mirrorbank.evaluate.fidelity import run_fidelity
from mirrorbank.instruments.ach import ACHSchema


def test_fidelity_identical_data_passes(ach_df):
    """A dataset compared to itself should pass all fidelity checks."""
    report = run_fidelity(ach_df, ach_df, schema=ACHSchema())
    assert report.ks_pass_rate == 1.0
    assert report.corr_distance == 0.0


def test_fidelity_completely_different_data_fails(ach_df):
    """A constant DataFrame should fail KS tests against the real data."""
    import random

    rng = random.Random(0)
    constant_df = ach_df.with_columns(
        pl.lit(0.0).alias("amount"),
        pl.lit(0).alias("risk_score"),
        pl.lit(0).alias("days_to_settle"),
    )
    report = run_fidelity(ach_df, constant_df, schema=ACHSchema())
    # At least some numeric columns should fail KS
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
