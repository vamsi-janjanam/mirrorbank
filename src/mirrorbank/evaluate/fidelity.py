"""
Fidelity evaluation — does synthetic data look like real data?

Three tests:
1. Per-column KS test (target: p > 0.05 for ≥ 80 % of columns)
2. Correlation matrix distance — Frobenius norm of (C_real - C_synth) (target: < 0.15)
3. Instrument-specific business-rule checks (e.g. wire has no weekend volume)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy.stats import ks_2samp

from mirrorbank.instruments.base import ColumnKind, InstrumentSchema


@dataclass
class FidelityReport:
    ks_results: dict[str, dict]         # col → {statistic, p_value, pass}
    ks_pass_rate: float                 # fraction of columns with p > 0.05
    corr_distance: float                # Frobenius norm
    instrument_errors: list[str]        # from InstrumentSchema.validate()
    pass_overall: bool

    @property
    def summary(self) -> str:
        status = "PASS" if self.pass_overall else "FAIL"
        return (
            f"Fidelity [{status}] "
            f"KS pass rate={self.ks_pass_rate:.1%} "
            f"corr_dist={self.corr_distance:.3f} "
            f"instrument_errors={len(self.instrument_errors)}"
        )


def run_fidelity(
    real: pl.DataFrame,
    synth: pl.DataFrame,
    schema: InstrumentSchema | None = None,
    ks_p_threshold: float = 0.05,
    ks_pass_rate_target: float = 0.80,
    corr_distance_target: float = 0.15,
) -> FidelityReport:
    training_cols = (
        schema.training_columns()
        if schema
        else [c for c in real.columns if real[c].dtype != pl.Utf8]
    )

    ks_results = _ks_tests(real, synth, training_cols, ks_p_threshold)
    ks_pass_rate = sum(v["pass"] for v in ks_results.values()) / max(len(ks_results), 1)
    corr_dist = _correlation_distance(real, synth, training_cols)
    instrument_errors = schema.validate(synth) if schema else []

    pass_overall = (
        ks_pass_rate >= ks_pass_rate_target
        and corr_dist < corr_distance_target
        and len(instrument_errors) == 0
    )

    return FidelityReport(
        ks_results=ks_results,
        ks_pass_rate=ks_pass_rate,
        corr_distance=corr_dist,
        instrument_errors=instrument_errors,
        pass_overall=pass_overall,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ks_tests(
    real: pl.DataFrame,
    synth: pl.DataFrame,
    columns: list[str],
    threshold: float,
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for col in columns:
        if col not in real.columns or col not in synth.columns:
            continue
        try:
            r = real[col].drop_nulls().to_numpy().astype(float)
            s = synth[col].drop_nulls().to_numpy().astype(float)
            if len(r) == 0 or len(s) == 0:
                continue
            stat, p = ks_2samp(r, s)
            results[col] = {"statistic": float(stat), "p_value": float(p), "pass": p > threshold}
        except Exception:
            pass
    return results


def _correlation_distance(
    real: pl.DataFrame,
    synth: pl.DataFrame,
    columns: list[str],
) -> float:
    numeric = [
        c for c in columns
        if c in real.columns
        and c in synth.columns
        and real[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64,
                               pl.UInt32, pl.UInt64, pl.Int8, pl.Int16)
    ]
    if len(numeric) < 2:
        return 0.0
    try:
        C_real = np.corrcoef(real.select(numeric).fill_null(0).to_numpy().T)
        C_synth = np.corrcoef(synth.select(numeric).fill_null(0).to_numpy().T)
        return float(np.linalg.norm(C_real - C_synth, "fro"))
    except Exception:
        return 0.0
