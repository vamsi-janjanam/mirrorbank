"""Tests for Stage 2 generators: Track A (DP statistical) and Track B (vocab sampler)."""

from __future__ import annotations

import pytest

from mirrorbank.generate.track_a import DPTabularGenerator
from mirrorbank.generate.track_b import generate_text_columns
from mirrorbank.instruments.registry import get_schema
from mirrorbank.privacy.budget import BudgetConfig
from mirrorbank.profile.schema_profiler import SchemaProfiler
from mirrorbank.sample_data import sample_dataset


@pytest.mark.parametrize("instrument", ["ach", "credit_card", "wire"])
def test_dp_tabular_generator_roundtrip(instrument: str) -> None:
    real = sample_dataset(instrument)
    schema = get_schema(instrument)
    profile = SchemaProfiler(schema=schema).fit(real)

    generator = DPTabularGenerator(
        schema, BudgetConfig.from_preset("balanced", dataset_size=real.height)
    )
    generator.fit(real, profile)

    synth = generator.sample(50)

    expected_cols = {c for c in schema.training_columns() if c in real.columns}
    assert synth.height == 50
    assert set(synth.columns) == expected_cols
    for col in synth.columns:
        assert synth[col].null_count() == 0
    assert generator.budget.current_epsilon() > 0


def test_generate_text_columns() -> None:
    structured = sample_dataset("ach").head(20)
    schema = get_schema("ach")

    cols = generate_text_columns(structured, schema, seed=1)

    text_cols = schema.text_columns()
    if not text_cols:
        assert cols == {}
    else:
        for col in text_cols:
            assert col in cols
            assert len(cols[col]) == 20
