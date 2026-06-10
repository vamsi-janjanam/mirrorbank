"""Tests for the DP-CTGAN engine wired into the generation pipeline."""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("opacus")

from mirrorbank.generate.pipeline import generate
from mirrorbank.instruments.registry import get_schema
from mirrorbank.privacy.budget import BudgetConfig
from mirrorbank.sample_data import sample_dataset


@pytest.mark.parametrize("instrument", ["ach", "credit_card"])
def test_dp_ctgan_end_to_end(instrument):
    real = sample_dataset(instrument)
    schema = get_schema(instrument)

    res = generate(
        real,
        schema,
        n_rows=50,
        budget_config=BudgetConfig.from_preset("balanced", dataset_size=real.height),
        model="dp_ctgan",
        seed=0,
    )

    assert res.synthetic.height == 50
    assert set(res.synthetic.columns) == {c.name for c in schema.columns}
    assert res.epsilon_spent > 0

    for col, dtype in zip(res.synthetic.columns, res.synthetic.dtypes):
        if dtype.is_numeric():
            arr = res.synthetic[col].drop_nulls().cast(__import__("polars").Float64).to_numpy()
            if arr.size:
                assert np.isfinite(arr).all(), f"non-finite values in {col}"

    # Dtypes must match the source so downstream schema.validate()/gauntlet work
    # (regression: CTGAN previously emitted Boolean columns as String).
    for col in res.synthetic.columns:
        if col in real.columns:
            assert res.synthetic.schema[col] == real.schema[col], f"dtype drift in {col}"

    # CTGAN output must flow through the gauntlet without crashing.
    from mirrorbank.evaluate.gauntlet import run_gauntlet

    run_gauntlet(real, res.synthetic, schema=schema)
