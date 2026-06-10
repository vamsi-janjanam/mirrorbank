"""Stage 2, Track A — DP statistical generator (independent DP marginals).

v1 design: each modeled column is sampled independently from a noisy histogram
(continuous/datetime) or noisy value-count distribution (categorical). No
cross-column correlation is modeled. Each modeled column charges one DP-SGD
"step" against the instrument's `PrivacyBudget`, using `noise_multiplier` (σ)
from the budget config as the Gaussian noise std added to histogram counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from mirrorbank.instruments.base import ColumnKind, InstrumentSchema
from mirrorbank.privacy.budget import BudgetConfig, BudgetExhausted, PrivacyBudget
from mirrorbank.profile.schema_profiler import DatasetProfile

_INTEGER_DTYPES = {
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
}


@dataclass
class _ColumnModel:
    """A fitted DP marginal for a single training column."""

    name: str
    kind: ColumnKind
    dtype: pl.DataType
    # CONTINUOUS / DATETIME: histogram over numeric range
    bin_edges: np.ndarray | None = None
    probs: np.ndarray | None = None
    # CATEGORICAL: discrete distribution over observed categories
    categories: list | None = None
    # Real column had no non-null values: emit all-null to match the source.
    all_null: bool = False


class DPTabularGenerator:
    """Independent per-column DP marginal generator (Track A, v1)."""

    def __init__(
        self,
        schema: InstrumentSchema,
        budget_config: BudgetConfig,
        *,
        n_bins: int = 32,
        seed: int = 0,
    ):
        self.schema = schema
        self.budget_config = budget_config
        self.n_bins = n_bins
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._budget: PrivacyBudget | None = None
        self._models: dict[str, _ColumnModel] = {}
        self._modeled_columns: list[str] = []

    @property
    def budget(self) -> PrivacyBudget:
        if self._budget is None:
            raise RuntimeError("DPTabularGenerator.fit() must be called first")
        return self._budget

    def fit(self, real: pl.DataFrame, profile: DatasetProfile) -> DPTabularGenerator:
        modeled = [c for c in self.schema.training_columns() if c in real.columns]
        self._modeled_columns = modeled

        if self.budget_config.dataset_size <= 0:
            self.budget_config.dataset_size = real.height
        self._budget = PrivacyBudget(self.budget_config)
        sigma = self.budget_config.noise_multiplier

        spec_by_name = {c.name: c for c in self.schema.columns}

        for col in modeled:
            try:
                self._budget.charge_step()
            except BudgetExhausted:
                pass

            kind = spec_by_name[col].kind
            series = real[col]
            dtype = series.dtype

            if kind == ColumnKind.CATEGORICAL:
                self._models[col] = self._fit_categorical(col, series, dtype, sigma)
            elif kind == ColumnKind.DATETIME:
                self._models[col] = self._fit_datetime(col, series, dtype, sigma)
            else:
                # CONTINUOUS and any other numeric training column
                self._models[col] = self._fit_continuous(
                    col, series, dtype, sigma, ColumnKind.CONTINUOUS
                )

        return self

    # ── Per-kind fitting ──────────────────────────────────────────────────────

    def _fit_categorical(
        self, name: str, series: pl.Series, dtype: pl.DataType, sigma: float
    ) -> _ColumnModel:
        vc = series.drop_nulls().value_counts()
        cat_col, count_col = vc.columns
        categories = vc[cat_col].to_list()
        counts = vc[count_col].to_numpy().astype(float)

        if len(categories) == 0:
            # No non-null observations: fall back to a single placeholder
            # category so sample() never produces nulls.
            categories = [""]
            counts = np.array([1.0])

        probs = self._noisy_probs(counts, sigma)
        return _ColumnModel(
            name=name, kind=ColumnKind.CATEGORICAL, dtype=dtype, categories=categories, probs=probs
        )

    def _fit_continuous(
        self,
        name: str,
        series: pl.Series,
        dtype: pl.DataType,
        sigma: float,
        kind: ColumnKind,
        n_bins: int | None = None,
    ) -> _ColumnModel:
        values = series.drop_nulls().cast(pl.Float64).to_numpy()
        if values.size == 0:
            # No non-null observations (e.g. an all-null column like
            # overdraft_fee): emit all-null to match the source data.
            return _ColumnModel(name=name, kind=kind, dtype=dtype, all_null=True)

        lo, hi = float(values.min()), float(values.max())
        if lo == hi:
            hi = lo + 1.0

        edges = np.linspace(lo, hi, (n_bins or self.n_bins) + 1)
        counts, _ = np.histogram(values, bins=edges)
        probs = self._noisy_probs(counts.astype(float), sigma)

        return _ColumnModel(
            name=name, kind=kind, dtype=dtype, bin_edges=edges, probs=probs
        )

    def _fit_datetime(
        self, name: str, series: pl.Series, dtype: pl.DataType, sigma: float
    ) -> _ColumnModel:
        non_null = series.drop_nulls()
        if non_null.len() == 0:
            return _ColumnModel(name=name, kind=ColumnKind.DATETIME, dtype=dtype, all_null=True)
        epoch = non_null.dt.epoch("s").cast(pl.Float64)
        # Bin time at ~1-hour resolution (derived from the actual span, capped)
        # so no bin straddles a day boundary. Combined with support preservation,
        # this zeroes out entire structural gaps exactly — e.g. Fedwire weekends
        # and overnight closures — so they are never sampled, for any time range.
        span = float(epoch.max() - epoch.min())
        n_bins = int(np.clip(span / 3600.0, 32, 20000))
        return self._fit_continuous(
            name, epoch, dtype, sigma, ColumnKind.DATETIME, n_bins=n_bins
        )

    def _noisy_probs(self, counts: np.ndarray, sigma: float) -> np.ndarray:
        noisy = counts + self._rng.normal(0.0, sigma, size=counts.shape)
        noisy = np.clip(noisy, 0.0, None)
        # Support preservation: never invent mass in bins that had no real
        # observations. Keeps structural gaps intact (wires never settle on
        # weekends, Zelle never exceeds its cap) at the cost of revealing which
        # bins were empty — an accepted v1 tradeoff.
        noisy = np.where(counts > 0, noisy, 0.0)
        total = noisy.sum()
        if total <= 0:
            return np.full(counts.shape, 1.0 / counts.size)
        return noisy / total

    # ── Sampling ──────────────────────────────────────────────────────────────

    def sample(self, n: int) -> pl.DataFrame:
        data: dict[str, list | np.ndarray] = {}
        for col in self._modeled_columns:
            model = self._models[col]
            if model.kind == ColumnKind.CATEGORICAL:
                data[col] = self._sample_categorical(model, n)
            elif model.kind == ColumnKind.DATETIME:
                data[col] = self._sample_datetime(model, n)
            else:
                data[col] = self._sample_continuous(model, n)

        df = pl.DataFrame(data)
        # Ensure column order matches modeled order and dtypes match originals.
        return df.select(self._modeled_columns)

    def _sample_categorical(self, model: _ColumnModel, n: int) -> list:
        idx = self._rng.choice(len(model.categories), size=n, p=model.probs)
        return [model.categories[i] for i in idx]

    def _sample_continuous(self, model: _ColumnModel, n: int) -> pl.Series:
        if model.all_null:
            return pl.Series(model.name, [None] * n, dtype=model.dtype)
        edges = model.bin_edges
        bin_idx = self._rng.choice(len(model.probs), size=n, p=model.probs)
        lo = edges[bin_idx]
        hi = edges[bin_idx + 1]
        values = self._rng.uniform(lo, hi)

        dtype = model.dtype
        if str(dtype) in _INTEGER_DTYPES:
            values = np.round(values).astype(np.int64)
            return pl.Series(model.name, values, dtype=dtype)
        values = np.round(values, 2)
        return pl.Series(model.name, values, dtype=pl.Float64).cast(dtype)

    def _sample_datetime(self, model: _ColumnModel, n: int) -> pl.Series:
        if model.all_null:
            return pl.Series(model.name, [None] * n, dtype=model.dtype)
        edges = model.bin_edges
        bin_idx = self._rng.choice(len(model.probs), size=n, p=model.probs)
        lo = edges[bin_idx]
        hi = edges[bin_idx + 1]
        epoch_secs = self._rng.uniform(lo, hi)
        epoch_secs = np.round(epoch_secs).astype(np.int64)
        series = pl.Series(model.name, epoch_secs, dtype=pl.Int64)
        return pl.from_epoch(series, time_unit="s").cast(model.dtype)
