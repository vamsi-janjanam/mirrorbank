"""Stage 2 pipeline orchestration — Track A (DP marginals) + Track B (vocab sampler).

`generate()` ties together:
  1. Schema profiling (`mirrorbank.profile.schema_profiler.SchemaProfiler`)
  2. Track A: DP-trained structured-column generator (`generate.track_a`)
  3. Reference / identifier / PII columns: regenerated post-hoc, never DP-trained
  4. Track B: free-text columns from a vocab sampler (`generate.track_b`),
     which never sees real data and consumes zero privacy budget
  5. Assembly into a single `pl.DataFrame` matching `schema.columns` order

Scaling note: the privacy budget is charged once, at *fit* time, against the
real data. Drawing synthetic rows afterwards costs **zero** additional budget,
so output size is unbounded. `generate()` builds the whole frame in memory;
`generate_to_csv()` streams arbitrarily many rows to disk in bounded-memory
chunks for large volumes.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from mirrorbank.instruments.base import InstrumentSchema
from mirrorbank.privacy.budget import BudgetConfig
from mirrorbank.profile.schema_profiler import SchemaProfiler


@dataclass
class GenerationResult:
    synthetic: pl.DataFrame
    epsilon_spent: float
    delta: float
    n_rows: int
    instrument: str


@dataclass
class StreamResult:
    """Result of a streamed-to-disk generation run (no in-memory frame)."""

    path: str
    epsilon_spent: float
    delta: float
    n_rows: int
    instrument: str


def _fit(real, schema, budget_config, seed, model="dp_statistical"):
    """Profile + fit Track A. This is where the privacy budget is charged."""
    profile = SchemaProfiler(schema=schema).fit(real)

    if model == "dp_ctgan":
        from mirrorbank.generate.ctgan import DPCTGANGenerator

        return DPCTGANGenerator(schema, budget_config, seed=seed).fit(real, profile)

    from mirrorbank.generate.track_a import DPTabularGenerator

    return DPTabularGenerator(schema, budget_config, seed=seed).fit(real, profile)


def _generator_epsilon(gen) -> float:
    """Read the privacy spend from either generator type."""
    if hasattr(gen, "budget"):
        return gen.budget.current_epsilon()
    return float(getattr(gen, "epsilon", 0.0))


def _build_chunk(gen, schema, n: int, *, seed: int, offset: int) -> pl.DataFrame:
    """Assemble one chunk of `n` synthetic rows from a fitted generator.

    `offset` is the global row index of this chunk's first row, so generated
    identifiers stay unique across chunks.
    """
    from mirrorbank.generate.track_b import generate_text_columns

    # Track A — DP-sampled structured columns (advances the generator's RNG,
    # so each chunk is fresh).
    structured = gen.sample(n)
    columns: dict[str, list] = {name: structured[name].to_list() for name in structured.columns}

    # Reference columns — regenerated post-hoc, never DP-trained.
    for spec in schema.reference_columns():
        if spec.name in columns:
            continue
        if spec.reference_generator is not None:
            columns[spec.name] = [spec.reference_generator() for _ in range(n)]
        else:
            columns[spec.name] = [f"<ref_{spec.name}_{offset + i}>" for i in range(n)]

    # Identifier columns — unique across chunks via the global offset.
    for name in schema.identifier_columns():
        if name in columns:
            continue
        spec = next((c for c in schema.columns if c.name == name), None)
        if spec is not None and spec.reference_generator is not None:
            columns[name] = [spec.reference_generator() for _ in range(n)]
        else:
            columns[name] = [f"{name}_{offset + i:012d}" for i in range(n)]

    # PII columns — excluded from training, filled with placeholders.
    for name in schema.pii_columns():
        if name in columns:
            continue
        columns[name] = [f"<pii_{name}_{offset + i}>" for i in range(n)]

    # Track B — free-text columns (synthetic seeds only, zero privacy budget).
    if schema.text_columns():
        text = generate_text_columns(structured, schema, seed=seed)
        for name, values in text.items():
            columns[name] = list(values)

    # Assemble in declared schema order; fill any unproduced columns with nulls.
    ordered = {spec.name: columns.get(spec.name, [None] * n) for spec in schema.columns}
    return pl.DataFrame(ordered)


def generate(
    real: pl.DataFrame,
    schema: InstrumentSchema,
    *,
    n_rows: int,
    budget_config: BudgetConfig,
    seed: int = 0,
    model: str = "dp_statistical",
) -> GenerationResult:
    """Run the full dual-track pipeline and return synthetic data in memory.

    For very large `n_rows`, prefer `generate_to_csv()` to keep memory bounded.
    """
    gen = _fit(real, schema, budget_config, seed, model=model)
    synthetic = _build_chunk(gen, schema, n_rows, seed=seed, offset=0)
    return GenerationResult(
        synthetic=synthetic,
        epsilon_spent=_generator_epsilon(gen),
        delta=budget_config.delta,
        n_rows=n_rows,
        instrument=schema.name,
    )


def generate_to_csv(
    real: pl.DataFrame,
    schema: InstrumentSchema,
    *,
    n_rows: int,
    budget_config: BudgetConfig,
    path: str,
    chunk_size: int = 1_000_000,
    seed: int = 0,
    model: str = "dp_statistical",
) -> StreamResult:
    """Stream `n_rows` synthetic rows to `path` (CSV) in bounded-memory chunks.

    Fits once (charging the privacy budget on the real data), then samples in
    batches of `chunk_size`. Peak memory is O(chunk_size), not O(n_rows), so the
    output is limited only by disk and time — not by RAM. Sampling consumes zero
    additional privacy budget.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    gen = _fit(real, schema, budget_config, seed, model=model)

    written = 0
    chunk_idx = 0
    with open(path, "w", newline="") as fh:
        while written < n_rows:
            this = min(chunk_size, n_rows - written)
            chunk = _build_chunk(gen, schema, this, seed=seed + chunk_idx, offset=written)
            fh.write(chunk.write_csv(include_header=(chunk_idx == 0)))
            written += this
            chunk_idx += 1

    return StreamResult(
        path=path,
        epsilon_spent=_generator_epsilon(gen),
        delta=budget_config.delta,
        n_rows=n_rows,
        instrument=schema.name,
    )
