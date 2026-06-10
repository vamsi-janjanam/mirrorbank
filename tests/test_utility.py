"""Tests for Stage 4 — TSTR utility evaluation."""

import polars as pl

from mirrorbank.evaluate.utility import run_utility
from mirrorbank.instruments.registry import get_schema
from mirrorbank.sample_data import sample_dataset


def _balanced_fraud_frame(n: int = 400) -> pl.DataFrame:
    """credit_card sample with a learnable, balanced fraud label.

    The base sample fixture has too few positive `is_fraud` rows to train a
    classifier, so overwrite the label with a synthetic balanced target
    derived from `amount` (above-median amount => "fraud").
    """
    df = sample_dataset("credit_card", n=n, seed=1)
    median_amount = df["amount"].median()
    return df.with_columns(
        (pl.col("amount") > median_amount).alias("is_fraud")
    )


def test_run_utility_returns_report_for_balanced_target():
    schema = get_schema("credit_card")
    real = _balanced_fraud_frame()
    synth = _balanced_fraud_frame()  # identical distribution => learnable

    report = run_utility(real, synth, schema=schema, seed=0)

    assert report is not None
    assert 0.0 <= report.auc_tstr <= 1.0
    assert 0.0 <= report.auc_trtr <= 1.0
    assert report.ratio >= 0.0
    assert report.target_column == "is_fraud"
    assert isinstance(report.pass_utility, bool)


def test_run_utility_none_without_target():
    schema = get_schema("ach")
    real = sample_dataset("ach")
    report = run_utility(real, real, schema=schema, target_column="not_a_column")
    assert report is None


def test_run_utility_none_when_too_few_positives():
    """The default credit_card sample has ~0% fraud — too few to train."""
    schema = get_schema("credit_card")
    real = sample_dataset("credit_card")
    report = run_utility(real, real, schema=schema)
    assert report is None
