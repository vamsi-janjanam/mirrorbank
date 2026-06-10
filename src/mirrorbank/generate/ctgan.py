"""Stage 2, Track A (alternative engine) — DP-CTGAN baseline.

A compact, vanilla GAN over a single standardized/one-hot float matrix.
Only the discriminator is trained with DP-SGD (via Opacus): it is the only
network exposed to real data, so it is the only network charged against the
privacy budget. The generator never sees real data and trains normally.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import torch
from torch import nn

from mirrorbank.instruments.base import ColumnKind, InstrumentSchema
from mirrorbank.privacy.budget import BudgetConfig
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

_NULL_CATEGORY = "<null>"


class _Generator(nn.Module):
    def __init__(self, z_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _Discriminator(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DPCTGANGenerator:
    """Compact DP-CTGAN baseline (Track A, alternative engine)."""

    def __init__(
        self,
        schema: InstrumentSchema,
        budget_config: BudgetConfig,
        *,
        epochs: int = 50,
        batch_size: int = 256,
        seed: int = 0,
    ):
        self.schema = schema
        self.budget_config = budget_config
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.epsilon: float = 0.0

        self._z_dim = 64
        self._hidden = 128
        self._modeled_columns: list[str] = []
        self._col_specs: dict = {}
        self._continuous_cols: list[str] = []
        self._categorical_cols: list[str] = []
        self._cont_stats: dict[str, tuple[float, float]] = {}
        self._cat_categories: dict[str, list] = {}
        self._dtypes: dict[str, pl.DataType] = {}
        self._kinds: dict[str, ColumnKind] = {}
        self._all_null: dict[str, bool] = {}
        self._slices: dict[str, slice] = {}
        self._width: int = 0
        self._generator: _Generator | None = None

    # ── Fitting ──────────────────────────────────────────────────────────────

    def fit(self, real: pl.DataFrame, profile: DatasetProfile) -> DPCTGANGenerator:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        spec_by_name = {c.name: c for c in self.schema.columns}
        modeled = [c for c in self.schema.training_columns() if c in real.columns]
        self._modeled_columns = modeled
        self._col_specs = {c: spec_by_name[c] for c in modeled}

        if self.budget_config.dataset_size <= 0:
            self.budget_config.dataset_size = real.height

        # Build the float matrix.
        blocks: list[np.ndarray] = []
        offset = 0
        for col in modeled:
            spec = spec_by_name[col]
            kind = spec.kind
            if kind == ColumnKind.DATETIME:
                kind = ColumnKind.CONTINUOUS  # treat as continuous via epoch seconds
                series = real[col]
                non_null = series.drop_nulls()
                if non_null.len() == 0:
                    self._all_null[col] = True
                    self._dtypes[col] = series.dtype
                    self._kinds[col] = ColumnKind.DATETIME
                    self._cont_stats[col] = (0.0, 1.0)
                    block = np.zeros((real.height, 1), dtype=np.float32)
                else:
                    epoch = real[col].dt.epoch("s").cast(pl.Float64).to_numpy()
                    epoch = np.nan_to_num(epoch, nan=np.nanmean(epoch))
                    mean, std = float(epoch.mean()), float(epoch.std())
                    if std == 0.0:
                        std = 1.0
                    self._cont_stats[col] = (mean, std)
                    self._all_null[col] = False
                    self._dtypes[col] = real[col].dtype
                    self._kinds[col] = ColumnKind.DATETIME
                    block = ((epoch - mean) / std).astype(np.float32).reshape(-1, 1)
                self._continuous_cols.append(col)
                self._slices[col] = slice(offset, offset + 1)
                offset += 1
                blocks.append(block)
                continue

            if spec.kind == ColumnKind.CATEGORICAL:
                series = real[col].cast(pl.Utf8)
                values = series.fill_null(_NULL_CATEGORY).to_list()
                categories = sorted({v for v in values})
                if not categories:
                    categories = [_NULL_CATEGORY]
                self._cat_categories[col] = categories
                self._dtypes[col] = real[col].dtype
                self._kinds[col] = ColumnKind.CATEGORICAL
                self._all_null[col] = False
                self._categorical_cols.append(col)
                cat_idx = {c: i for i, c in enumerate(categories)}
                onehot = np.zeros((real.height, len(categories)), dtype=np.float32)
                for i, v in enumerate(values):
                    onehot[i, cat_idx.get(v, 0)] = 1.0
                self._slices[col] = slice(offset, offset + len(categories))
                offset += len(categories)
                blocks.append(onehot)
                continue

            # CONTINUOUS (or any other numeric training column)
            series = real[col]
            non_null = series.drop_nulls().cast(pl.Float64)
            self._dtypes[col] = series.dtype
            self._kinds[col] = ColumnKind.CONTINUOUS
            if non_null.len() == 0:
                self._all_null[col] = True
                self._cont_stats[col] = (0.0, 1.0)
                block = np.zeros((real.height, 1), dtype=np.float32)
            else:
                values = series.cast(pl.Float64).to_numpy()
                fill = float(non_null.mean())
                values = np.nan_to_num(values, nan=fill)
                mean, std = float(values.mean()), float(values.std())
                if std == 0.0:
                    std = 1.0
                self._cont_stats[col] = (mean, std)
                self._all_null[col] = False
                block = ((values - mean) / std).astype(np.float32).reshape(-1, 1)
            self._continuous_cols.append(col)
            self._slices[col] = slice(offset, offset + 1)
            offset += 1
            blocks.append(block)

        self._width = offset
        if blocks:
            X = np.concatenate(blocks, axis=1).astype(np.float32)
        else:
            X = np.zeros((real.height, 0), dtype=np.float32)
            self._width = 1
            X = np.zeros((real.height, 1), dtype=np.float32)

        self._train(X)
        return self

    def _train(self, X: np.ndarray) -> None:
        n, width = X.shape
        device = torch.device("cpu")

        generator = _Generator(self._z_dim, self._hidden, width).to(device)
        discriminator = _Discriminator(width, self._hidden).to(device)

        g_opt = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.9))
        d_opt = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.9))

        bs = min(self.batch_size, max(n, 2))
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(X))
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=bs, shuffle=True, drop_last=False
        )

        criterion = nn.BCEWithLogitsLoss()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from opacus import PrivacyEngine

            privacy_engine = PrivacyEngine(secure_mode=False)
            discriminator, d_opt, loader = privacy_engine.make_private(
                module=discriminator,
                optimizer=d_opt,
                data_loader=loader,
                noise_multiplier=self.budget_config.noise_multiplier,
                max_grad_norm=self.budget_config.max_grad_norm,
            )

            generator.train()
            discriminator.train()

            for _ in range(self.epochs):
                for (real_batch,) in loader:
                    real_batch = real_batch.to(device)
                    cur_bs = real_batch.shape[0]
                    if cur_bs < 2:
                        continue

                    # ── Discriminator step (DP) ──
                    z = torch.randn(cur_bs, self._z_dim, device=device)
                    with torch.no_grad():
                        fake_batch = self._apply_activations(generator(z))

                    d_opt.zero_grad()
                    real_pred = discriminator(real_batch)
                    fake_pred = discriminator(fake_batch)
                    d_loss = criterion(
                        real_pred, torch.ones_like(real_pred)
                    ) + criterion(fake_pred, torch.zeros_like(fake_pred))
                    d_loss.backward()
                    d_opt.step()

                    # ── Generator step (no real data, no privacy cost) ──
                    # Disable Opacus's per-sample-grad hooks on the
                    # discriminator while back-propagating into the
                    # generator: this pass must not be charged against the
                    # privacy budget and Opacus's Poisson-sampled DataLoader
                    # forbids extra backward passes between optimizer steps.
                    z = torch.randn(cur_bs, self._z_dim, device=device)
                    fake_batch = self._apply_activations(generator(z))
                    g_opt.zero_grad()
                    discriminator.disable_hooks()
                    try:
                        gen_pred = discriminator(fake_batch)
                        g_loss = criterion(gen_pred, torch.ones_like(gen_pred))
                        g_loss.backward()
                    finally:
                        discriminator.enable_hooks()
                    g_opt.step()

            self.epsilon = float(privacy_engine.get_epsilon(delta=self.budget_config.delta))

        generator.eval()
        self._generator = generator

    def _apply_activations(self, raw: torch.Tensor) -> torch.Tensor:
        """Apply identity to continuous slices and softmax to categorical slices."""
        out = raw.clone()
        for col in self._categorical_cols:
            sl = self._slices[col]
            out[:, sl] = torch.softmax(raw[:, sl], dim=-1)
        return out

    # ── Sampling ─────────────────────────────────────────────────────────────

    def sample(self, n: int) -> pl.DataFrame:
        if self._generator is None:
            raise RuntimeError("DPCTGANGenerator.fit() must be called first")

        self._generator.eval()
        with torch.no_grad():
            z = torch.randn(max(n, 1), self._z_dim)
            raw = self._apply_activations(self._generator(z))
        arr = raw.numpy()[:n]
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        data: dict[str, list | np.ndarray] = {}
        for col in self._modeled_columns:
            sl = self._slices[col]
            kind = self._kinds[col]
            dtype = self._dtypes[col]

            if kind == ColumnKind.CATEGORICAL:
                idx = arr[:, sl].argmax(axis=1)
                categories = self._cat_categories[col]
                values = []
                for i in idx:
                    cat = categories[int(i)]
                    values.append(None if cat == _NULL_CATEGORY else cat)
                # Categories are stored as strings (Utf8), but the source column
                # may be Boolean/Int/etc. Cast back to the original dtype so
                # downstream schema.validate() (which relies on dtypes) works.
                if dtype == pl.Boolean:
                    _boolmap = {"true": True, "false": False, "True": True, "False": False, "1": True, "0": False}
                    values = [None if v is None else _boolmap.get(str(v)) for v in values]
                    data[col] = pl.Series(col, values, dtype=pl.Boolean)
                else:
                    series = pl.Series(col, values, dtype=pl.Utf8)
                    try:
                        series = series.cast(dtype, strict=False)
                    except Exception:
                        pass
                    data[col] = series
                continue

            if self._all_null.get(col):
                data[col] = pl.Series(col, [None] * n, dtype=dtype)
                continue

            mean, std = self._cont_stats[col]
            values = arr[:, sl].reshape(-1) * std + mean
            values = np.nan_to_num(values, nan=mean, posinf=mean, neginf=mean)

            if kind == ColumnKind.DATETIME:
                epoch_secs = np.round(values).astype(np.int64)
                series = pl.from_epoch(pl.Series(col, epoch_secs), time_unit="s")
                data[col] = series.cast(dtype)
                continue

            if str(dtype) in _INTEGER_DTYPES:
                values = np.round(values).astype(np.int64)
                data[col] = pl.Series(col, values, dtype=dtype)
            else:
                values = np.round(values, 2)
                data[col] = pl.Series(col, values, dtype=pl.Float64).cast(dtype)

        df = pl.DataFrame(data)
        return df.select(self._modeled_columns)
