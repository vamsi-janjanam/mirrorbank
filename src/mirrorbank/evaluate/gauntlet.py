"""Stage 4 — Evaluation Gauntlet orchestrator."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from mirrorbank.evaluate.fidelity import FidelityReport, run_fidelity
from mirrorbank.instruments.base import InstrumentSchema


@dataclass
class GauntletReport:
    instrument: str
    fidelity: FidelityReport
    pass_overall: bool

    def print_summary(self) -> None:
        status = "✅ PASS" if self.pass_overall else "❌ FAIL"
        print(f"\n{'=' * 60}")
        print(f"  Mirrorbank Gauntlet — {self.instrument.upper()}  {status}")
        print(f"{'=' * 60}")
        print(f"  Fidelity   {self.fidelity.summary}")
        print(f"{'=' * 60}\n")


def run_gauntlet(
    real: pl.DataFrame,
    synth: pl.DataFrame,
    schema: InstrumentSchema | None = None,
) -> GauntletReport:
    """Run all evaluation batteries and return a unified report."""
    instrument = schema.name if schema else "unknown"

    fidelity = run_fidelity(real, synth, schema=schema)

    return GauntletReport(
        instrument=instrument,
        fidelity=fidelity,
        pass_overall=fidelity.pass_overall,
    )
