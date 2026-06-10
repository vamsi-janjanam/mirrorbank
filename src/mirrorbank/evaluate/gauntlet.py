"""Stage 4 — Evaluation Gauntlet orchestrator."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from mirrorbank.evaluate.fidelity import FidelityReport, run_fidelity
from mirrorbank.evaluate.mia_audit import MIAReport, run_mia
from mirrorbank.evaluate.utility import UtilityReport
from mirrorbank.evaluate.utility import run_utility as _run_utility
from mirrorbank.instruments.base import InstrumentSchema


@dataclass
class GauntletReport:
    instrument: str
    fidelity: FidelityReport
    pass_overall: bool
    utility: UtilityReport | None = None
    privacy: MIAReport | None = None

    def print_summary(self) -> None:
        status = "✅ PASS" if self.pass_overall else "❌ FAIL"
        print(f"\n{'=' * 60}")
        print(f"  Mirrorbank Gauntlet — {self.instrument.upper()}  {status}")
        print(f"{'=' * 60}")
        print(f"  Fidelity   {self.fidelity.summary}")
        if self.utility is not None:
            u = self.utility
            u_status = "PASS" if u.pass_utility else "FAIL"
            print(
                f"  Utility    [{u_status}] AUC_TSTR={u.auc_tstr:.3f} "
                f"AUC_TRTR={u.auc_trtr:.3f} ratio={u.ratio:.3f} "
                f"target={u.target_column}"
            )
        if self.privacy is not None:
            p = self.privacy
            p_status = "PASS" if p.pass_privacy else "FAIL"
            print(f"  Privacy    [{p_status}] attack_auc={p.attack_auc:.3f}")
        print(f"{'=' * 60}\n")


def run_gauntlet(
    real: pl.DataFrame,
    synth: pl.DataFrame,
    schema: InstrumentSchema | None = None,
    *,
    target_column: str | None = None,
    run_utility: bool = True,
    run_privacy: bool = True,
) -> GauntletReport:
    """Run all evaluation batteries and return a unified report."""
    instrument = schema.name if schema else "unknown"

    fidelity = run_fidelity(real, synth, schema=schema)

    utility = (
        _run_utility(real, synth, schema=schema, target_column=target_column)
        if run_utility
        else None
    )
    privacy = run_mia(real, synth, schema=schema) if run_privacy else None

    pass_overall = (
        fidelity.pass_overall
        and (utility.pass_utility if utility else True)
        and (privacy.pass_privacy if privacy else True)
    )

    return GauntletReport(
        instrument=instrument,
        fidelity=fidelity,
        pass_overall=pass_overall,
        utility=utility,
        privacy=privacy,
    )
