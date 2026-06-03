"""
Stage 4 — Evaluation Gauntlet orchestrator.

Runs fidelity, utility, and privacy tests in parallel and emits a
unified GauntletReport.  All three batteries must pass for a release
to be certified.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass

import polars as pl

from mirrorbank.evaluate.fidelity import FidelityReport, run_fidelity
from mirrorbank.instruments.base import InstrumentSchema


@dataclass
class GauntletReport:
    instrument: str
    fidelity: FidelityReport
    # utility and privacy reports added in later weeks
    pass_overall: bool

    def print_summary(self) -> None:
        status = "✅ PASS" if self.pass_overall else "❌ FAIL"
        print(f"\n{'='*60}")
        print(f"  Mirrorbank Gauntlet — {self.instrument.upper()}  {status}")
        print(f"{'='*60}")
        print(f"  Fidelity   {self.fidelity.summary}")
        print(f"{'='*60}\n")


def run_gauntlet(
    real: pl.DataFrame,
    synth: pl.DataFrame,
    schema: InstrumentSchema | None = None,
) -> GauntletReport:
    """Run all evaluation batteries and return a unified report."""
    instrument = schema.name if schema else "unknown"

    # Fidelity runs synchronously for now; utility and MIA added in Week 3
    fidelity = run_fidelity(real, synth, schema=schema)

    return GauntletReport(
        instrument=instrument,
        fidelity=fidelity,
        pass_overall=fidelity.pass_overall,
    )
