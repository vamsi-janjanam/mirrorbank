"""Tests for Stage 1 — SchemaProfiler."""

import polars as pl
import pytest

from mirrorbank.instruments.ach import ACHSchema
from mirrorbank.instruments.base import ColumnKind
from mirrorbank.profile.schema_profiler import SchemaProfiler


def test_profile_basic_shape(ach_df):
    profiler = SchemaProfiler(schema=ACHSchema())
    profile = profiler.fit(ach_df)
    assert profile.n_rows == len(ach_df)
    assert profile.n_cols == len(ach_df.columns)
    assert profile.instrument == "ach"


def test_profile_respects_schema_kinds(ach_df):
    profiler = SchemaProfiler(schema=ACHSchema())
    profile = profiler.fit(ach_df)

    amount_col = profile.get_column("amount")
    assert amount_col is not None
    assert amount_col.kind == ColumnKind.CONTINUOUS

    sec_code_col = profile.get_column("sec_code")
    assert sec_code_col is not None
    assert sec_code_col.kind == ColumnKind.CATEGORICAL


def test_training_columns_exclude_pii(ach_df):
    profiler = SchemaProfiler(schema=ACHSchema())
    profile = profiler.fit(ach_df)
    training = profile.training_columns()
    assert "originator_account" not in training
    assert "receiver_account" not in training


def test_profile_without_schema():
    df = pl.DataFrame(
        {
            "amount": [1.0, 2.0, 3.0],
            "category": ["a", "b", "a"],
            "id": ["x1", "x2", "x3"],
        }
    )
    profiler = SchemaProfiler()
    profile = profiler.fit(df)
    assert profile.n_rows == 3
    assert profile.get_column("amount") is not None
