"""Write example CSVs to data/sample/ — one "real" and one "synthetic" frame
per instrument.

The two frames are drawn from the same builders with different seeds, so they
are statistically similar but not identical — exactly what the Fidelity tab
needs for a meaningful (mostly-passing) gauntlet demo.

Usage:
    pipenv run python scripts/generate_sample_data.py
    # or: make sample-data
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from mirrorbank.sample_data import _BUILDERS, sample_dataset  # noqa: E402

OUT_DIR = _ROOT / "data" / "sample"

REAL_SEED = 42
SYNTH_SEED = 7


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for instrument in _BUILDERS:
        real = sample_dataset(instrument, seed=REAL_SEED)
        synth = sample_dataset(instrument, seed=SYNTH_SEED)
        real_path = OUT_DIR / f"{instrument}_real.csv"
        synth_path = OUT_DIR / f"{instrument}_synth.csv"
        real.write_csv(real_path)
        synth.write_csv(synth_path)
        print(f"  {instrument:12s} → {real.height:>4} rows  ({real_path.name}, {synth_path.name})")
    print(f"\nWrote {2 * len(_BUILDERS)} CSVs to {OUT_DIR}")


if __name__ == "__main__":
    main()
