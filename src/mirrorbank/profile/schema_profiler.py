"""
Stage 1 — Schema Profiler.

Reads a tabular dataset and produces a DatasetProfile that is the
input contract for every downstream stage.  Each ColumnProfile captures:
  - inferred column kind (continuous / categorical / datetime / free_text / identifier)
  - PII flag
  - per-column statistics

The profiler respects any instrument schema passed in: column kinds declared
in the schema take precedence over heuristic inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from mirrorbank.instruments.base import ColumnKind, InstrumentSchema

# ── PII keyword heuristic ─────────────────────────────────────────────────────

_PII_KEYWORDS = frozenset(
    {
        "name", "first_name", "last_name", "fullname",
        "ssn", "social_security",
        "email", "phone", "mobile", "telephone",
        "address", "street", "zip", "postal",
        "dob", "birth", "birthdate",
        "account", "card", "pan", "iban",
        "user", "customer", "member",
        "ip", "device_id", "cookie",
    }
)

# ── Column thresholds ─────────────────────────────────────────────────────────

_MAX_CATEGORICAL_UNIQUE = 200       # n_unique ≤ this → CATEGORICAL
_MAX_CATEGORICAL_FRAC = 0.02        # n_unique/n_rows ≤ this → CATEGORICAL
_MIN_FREE_TEXT_AVG_TOKENS = 2.5     # avg word count above this → FREE_TEXT candidate
_MIN_IDENTIFIER_FRAC = 0.90         # n_unique/n_rows above this → IDENTIFIER


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ColumnProfile:
    name: str
    kind: ColumnKind
    dtype: str
    is_pii: bool
    n_unique: int
    null_frac: float
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetProfile:
    instrument: str
    n_rows: int
    n_cols: int
    columns: list[ColumnProfile]
    target_column: str | None = None

    # ── Column routing helpers ────────────────────────────────────────────────

    def training_columns(self) -> list[str]:
        """Columns that go to the DP generator (structured, non-PII)."""
        return [
            c.name
            for c in self.columns
            if c.kind in (ColumnKind.CONTINUOUS, ColumnKind.CATEGORICAL, ColumnKind.DATETIME)
            and not c.is_pii
        ]

    def text_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.kind == ColumnKind.FREE_TEXT]

    def get_column(self, name: str) -> ColumnProfile | None:
        return next((c for c in self.columns if c.name == name), None)

    def __repr__(self) -> str:
        kinds = {k.value: 0 for k in ColumnKind}
        for c in self.columns:
            kinds[c.kind.value] += 1
        return (
            f"<DatasetProfile instrument={self.instrument!r} "
            f"n_rows={self.n_rows:,} n_cols={self.n_cols} kinds={kinds}>"
        )


# ── Profiler ──────────────────────────────────────────────────────────────────

class SchemaProfiler:
    """
    Infers column types and statistics from a Polars DataFrame.

    Usage:
        profiler = SchemaProfiler(schema=ACHSchema())
        profile = profiler.fit(df)
    """

    def __init__(
        self,
        schema: InstrumentSchema | None = None,
        target_column: str | None = None,
    ):
        self.schema = schema
        self.target_column = target_column

    def fit(self, df: pl.DataFrame) -> DatasetProfile:
        n_rows, n_cols = df.shape
        columns = [self._profile_column(df, col) for col in df.columns]
        instrument = self.schema.name if self.schema else "unknown"
        return DatasetProfile(
            instrument=instrument,
            n_rows=n_rows,
            n_cols=n_cols,
            columns=columns,
            target_column=self.target_column,
        )

    # ── Column-level profiling ────────────────────────────────────────────────

    def _profile_column(self, df: pl.DataFrame, col: str) -> ColumnProfile:
        series = df[col]
        n_rows = len(series)
        n_unique = series.n_unique()
        null_frac = series.null_count() / n_rows if n_rows > 0 else 0.0
        dtype = str(series.dtype)

        # Instrument schema takes precedence over heuristics
        if self.schema:
            spec = next((c for c in self.schema.columns if c.name == col), None)
            if spec:
                return ColumnProfile(
                    name=col,
                    kind=spec.kind,
                    dtype=dtype,
                    is_pii=spec.is_pii,
                    n_unique=n_unique,
                    null_frac=null_frac,
                    stats=self._compute_stats(series, spec.kind),
                )

        # Heuristic inference
        kind = self._infer_kind(series, n_unique, n_rows)
        is_pii = self._detect_pii(col)
        return ColumnProfile(
            name=col,
            kind=kind,
            dtype=dtype,
            is_pii=is_pii,
            n_unique=n_unique,
            null_frac=null_frac,
            stats=self._compute_stats(series, kind),
        )

    def _infer_kind(self, series: pl.Series, n_unique: int, n_rows: int) -> ColumnKind:
        dtype = series.dtype

        # Datetime types
        if dtype in (pl.Date, pl.Datetime, pl.Time):
            return ColumnKind.DATETIME

        # Numeric types
        if dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                     pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
            if n_unique <= _MAX_CATEGORICAL_UNIQUE:
                return ColumnKind.CATEGORICAL
            return ColumnKind.CONTINUOUS

        # String types
        if dtype == pl.Utf8 or dtype == pl.String:
            uniq_frac = n_unique / n_rows if n_rows > 0 else 0
            if uniq_frac >= _MIN_IDENTIFIER_FRAC:
                return ColumnKind.IDENTIFIER
            if n_unique <= _MAX_CATEGORICAL_UNIQUE or uniq_frac <= _MAX_CATEGORICAL_FRAC:
                return ColumnKind.CATEGORICAL
            # Check average word count for free-text detection
            avg_tokens = (
                series.drop_nulls()
                .str.split(" ")
                .list.len()
                .mean()
            )
            if avg_tokens and avg_tokens >= _MIN_FREE_TEXT_AVG_TOKENS:
                return ColumnKind.FREE_TEXT
            return ColumnKind.CATEGORICAL

        return ColumnKind.CATEGORICAL

    def _detect_pii(self, col_name: str) -> bool:
        lower = col_name.lower()
        return any(kw in lower for kw in _PII_KEYWORDS)

    def _compute_stats(self, series: pl.Series, kind: ColumnKind) -> dict[str, Any]:
        stats: dict[str, Any] = {}
        try:
            if kind == ColumnKind.CONTINUOUS:
                desc = series.drop_nulls().describe()
                stats = {
                    "min": series.min(),
                    "max": series.max(),
                    "mean": series.mean(),
                    "std": series.std(),
                    "median": series.median(),
                }
            elif kind == ColumnKind.CATEGORICAL:
                vc = series.drop_nulls().value_counts(sort=True).head(20)
                stats["top_values"] = vc.to_dicts()
        except Exception:
            pass
        return stats
