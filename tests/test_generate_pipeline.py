"""Tests for the Stage 2 generation pipeline orchestration."""

from __future__ import annotations

import pytest

pytest.importorskip("mirrorbank.generate.track_a")
pytest.importorskip("mirrorbank.generate.track_b")

import polars as pl

from mirrorbank.generate.pipeline import generate, generate_to_csv
from mirrorbank.instruments.registry import get_schema
from mirrorbank.privacy.budget import BudgetConfig
from mirrorbank.sample_data import sample_dataset

ALL_INSTRUMENTS = ["ach", "wire", "credit_card", "zelle", "check", "debit_card"]


def test_generate_ach():
    real = sample_dataset("ach")
    schema = get_schema("ach")
    cfg = BudgetConfig.from_preset("balanced", dataset_size=real.height)

    result = generate(real, schema, n_rows=40, budget_config=cfg)

    assert result.synthetic.height == 40
    assert set(result.synthetic.columns) == {c.name for c in schema.columns}
    assert result.epsilon_spent > 0
    assert result.instrument == "ach"


@pytest.mark.parametrize("instrument", ALL_INSTRUMENTS)
@pytest.mark.parametrize("seed", [0, 1, 3, 7])
def test_generate_all_instruments_contract_and_rules(instrument, seed):
    """Every instrument must round-trip cleanly and emit ZERO business-rule
    violations — notably wire (no weekend volume) and zelle (<= $2,500 cap).
    Regression guard for the datetime-binning weekend leak."""
    real = sample_dataset(instrument)
    schema = get_schema(instrument)
    cfg = BudgetConfig.from_preset("balanced", dataset_size=real.height)

    result = generate(real, schema, n_rows=200, budget_config=cfg, seed=seed)

    assert result.synthetic.height == 200
    assert set(result.synthetic.columns) == {c.name for c in schema.columns}
    assert result.epsilon_spent > 0
    assert schema.validate(result.synthetic) == []


def test_generate_to_csv_streams_multiple_chunks(tmp_path):
    """Streaming to disk must produce the exact row count across chunk
    boundaries, preserve schema columns and business rules, and charge ε once."""
    real = sample_dataset("wire")
    schema = get_schema("wire")
    cfg = BudgetConfig.from_preset("balanced", dataset_size=real.height)
    path = tmp_path / "wire_synth.csv"

    res = generate_to_csv(
        real, schema, n_rows=2500, budget_config=cfg, path=str(path), chunk_size=1000, seed=2
    )

    df = pl.read_csv(path)
    assert res.n_rows == 2500
    assert df.height == 2500  # 3 chunks (1000 + 1000 + 500), single header
    assert set(df.columns) == {c.name for c in schema.columns}
    assert res.epsilon_spent > 0
    assert schema.validate(df) == []  # weekend rule holds across chunk seams
