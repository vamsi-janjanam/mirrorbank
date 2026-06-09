# Getting Started with Mirrorbank

A plain-English tour for someone opening this project for the first time.

## What is Mirrorbank?

Mirrorbank is a **differentially private synthetic financial data generator**.
You give it a sensitive table of real transactions for one of six payment
instruments — **ACH, check, Zelle, wire, credit card, debit card** — and it is
designed to produce three things:

1. A **synthetic CSV** — fake transactions that statistically mimic the real
   ones but contain *no real customer records*.
2. A **privacy certificate** — a formal (ε, δ) differential-privacy guarantee
   plus a membership-inference audit result.
3. A **utility scorecard** — evidence the synthetic data is still useful (KS
   tests, and a fraud model trained on synthetic data scoring close to one
   trained on real data).

One-liner: *share and model on transaction data without exposing real
customers, and prove with numbers that it's both safe and useful.*

## Who it's for / how it helps

Banks and fintechs can't hand real customer data to data scientists, vendors,
or ML teams without legal/compliance risk, and classic anonymization (masking,
k-anonymity) is breakable. Mirrorbank gives you a drop-in synthetic dataset
backed by a *mathematical* privacy guarantee. It helps you:

- Unblock ML/analytics teams stuck waiting on data-access approvals.
- Share realistic data with vendors and partners safely.
- Build fraud models on data that behaves like production.

## What works today vs. not-yet-built

This is an **early build**. Be aware of what's implemented before relying on it:

| Capability | Status | Where |
|---|---|---|
| Instrument schemas (6) + auto-detection | ✅ done | `instruments/` |
| Fake reference identifiers (routing, SWIFT, IMAD, MICR) | ✅ done | `reference/` |
| Schema profiler (column kinds, PII, stats) | ✅ done | `profile/` |
| Privacy budget accountant (RDP, ε→σ calibration) | ✅ done | `privacy/budget.py` |
| Fidelity evaluation (KS tests, correlation, business rules) | ✅ done | `evaluate/` |
| Interactive UI | ✅ done | `ui/app.py` |
| **Synthetic data generation (DP-CTGAN + vocab sampler)** | 🔲 **not yet built** | `generate/` |
| Utility eval (TSTR AUC) | 🔲 not yet built | `evaluate/` |
| Privacy audit (shadow-model MIA) | 🔲 not yet built | `evaluate/` |
| Release bundling (CSV + cert PDF + scorecard HTML) | 🔲 not yet built | `release/` |
| CLI (`mirrorbank ...`) | 🔲 not yet built | `cli.py` |

> ⚠️ The actual **generation** step (Stage 2) doesn't exist yet — you can't yet
> click a button and *get* synthetic data out. Today you can profile data, plan
> a privacy budget, and evaluate a synthetic dataset you already have.

## Quickstart

```bash
# 1. Install dependencies
pipenv install
pipenv install --dev          # pytest, ruff

# 2. Generate the bundled example CSVs (real + synthetic per instrument)
make sample-data              # writes data/sample/*.csv

# 3. Launch the interactive UI
make ui                       # → http://localhost:8501

# 4. (optional) run the test suite
pipenv run pytest
```

In the UI, the fastest way to see it work: open **📊 Data Profiler** or
**📐 Fidelity Evaluation** and click *“Load a built-in example”* — no data of
your own required.

## Integration paths

**1. Python API** (the real integration surface today):

```python
import polars as pl
from mirrorbank.instruments.registry import get_schema, detect_instrument
from mirrorbank.profile.schema_profiler import SchemaProfiler
from mirrorbank.privacy.budget import calibrate_noise_multiplier
from mirrorbank.evaluate.gauntlet import run_gauntlet

df = pl.read_csv("transactions.csv")
schema = get_schema("wire")            # or: detect_instrument(df)

# Profile your data
profile = SchemaProfiler(schema=schema).fit(df)

# Calibrate a privacy budget for your dataset
sigma = calibrate_noise_multiplier(
    target_epsilon=3.0, delta=1e-5,
    n_rows=profile.n_rows, batch_size=4096, epochs=30,
)

# Evaluate a synthetic dataset against the real one
report = run_gauntlet(df, synth_df, schema=schema)
report.print_summary()
```

Need example frames in code? `from mirrorbank.sample_data import sample_dataset`
then `sample_dataset("wire")`.

**2. This Streamlit UI** — interactive, no code (`make ui`).

**3. CLI** — `mirrorbank ...` is scaffolded but not implemented yet.

**4. Config files** — `configs/*.yaml` hold reproducible run definitions
(instrument, target/exclude columns, generation + training params, privacy ε /
preset, evaluation targets); consumed once the generation pipeline lands.

## Where to go next

- `README.md` — full target architecture and roadmap.
- `CLAUDE.md` — build status table and design decisions.
- `tests/conftest.py` / `src/mirrorbank/sample_data.py` — example rows per
  instrument (also the canonical schema reference).
