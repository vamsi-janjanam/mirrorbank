"""Shared pytest fixtures — tiny synthetic DataFrames for every instrument.

The per-instrument builders live in `mirrorbank.sample_data` (the single source
of truth shared with the UI and the sample-CSV generator); these fixtures are
thin wrappers around `sample_dataset()`.
"""

from __future__ import annotations

import random

import polars as pl
import pytest

from mirrorbank.sample_data import sample_dataset


@pytest.fixture
def rng() -> random.Random:
    return random.Random(42)


@pytest.fixture
def ach_df() -> pl.DataFrame:
    return sample_dataset("ach", seed=42)


@pytest.fixture
def wire_df() -> pl.DataFrame:
    return sample_dataset("wire", seed=42)


@pytest.fixture
def zelle_df() -> pl.DataFrame:
    return sample_dataset("zelle", seed=42)


@pytest.fixture
def check_df() -> pl.DataFrame:
    return sample_dataset("check", seed=42)


@pytest.fixture
def debit_card_df() -> pl.DataFrame:
    return sample_dataset("debit_card", seed=42)


@pytest.fixture
def credit_card_df() -> pl.DataFrame:
    return sample_dataset("credit_card", seed=42)
