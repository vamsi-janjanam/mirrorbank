# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Mirrorbank is a differentially private synthetic financial data generator. Given a sensitive tabular dataset, it will produce a synthetic CSV, a privacy certificate (ε, δ values + MIA audit result), and a utility scorecard (KS tests, TSTR AUC) across six payment instruments: ACH, check, Zelle, wire, credit card, and debit card.

**Hero metric:** Generate 1M synthetic transactions at ε=3 in under 10 minutes, with a fraud classifier trained on synthetic data reaching ≥90% of real-data AUC, and membership inference attack accuracy below 53%.

**Current state:** early build. The instrument schemas, schema profiler, privacy budget accountant, and fidelity evaluation are implemented and tested. The generation pipeline, utility/privacy evaluation, release bundling, UI, and CLI are scaffolded (empty `__init__.py` only) — see "Build status" below before assuming a module exists.

## Commands

```bash
# Install dependencies
pipenv install

# Install dev dependencies (pytest, ruff)
pipenv install --dev

# Install DP generation stack (torch, opacus, sdv) — needed once Track A is implemented
pipenv install --categories generate

# Run all tests
pipenv run pytest
# or: make test

# Run a single test file
pipenv run pytest tests/test_budget.py

# Run a single test
pipenv run pytest tests/test_budget.py::test_name

# Lint / format
pipenv run ruff check src/ tests/      # or: make lint
pipenv run ruff format src/ tests/     # or: make fmt

# List available payment instruments
make instruments

# Run the evaluation gauntlet (requires real + synthetic CSVs on disk — see Makefile target)
make gauntlet DATASET=tabformer SYNTH=dp_ctgan_eps3
```

`make ui` / `pipenv run streamlit run src/mirrorbank/ui/app.py` and the `mirrorbank` CLI entry point are wired into the Makefile/`pyproject.toml` but their source files don't exist yet (see below).

## Build status

This matters because several files referenced by the README's target architecture and by stale `__pycache__` entries are not yet written — don't assume they exist.

| Stage / area | Status | Notes |
|---|---|---|
| `instruments/` (all 6 schemas + registry + `detect_instrument`) | ✅ done | |
| `reference/` (routing numbers, SWIFT/BIC, ACH trace, Fedwire IMAD, MICR) | ✅ done | |
| `profile/schema_profiler.py` (`SchemaProfiler`, `DatasetProfile`) | ✅ done | |
| `privacy/budget.py` (`PrivacyBudget`, `BudgetConfig`, presets, calibration) | ✅ done | |
| `evaluate/fidelity.py` + `evaluate/gauntlet.py` (KS tests, corr distance, `GauntletReport`) | ✅ done | `GauntletReport` currently only wires up fidelity; utility/privacy fields are commented `# added in later weeks` |
| `evaluate/utility.py` (TSTR AUC) | 🔲 not written | |
| `evaluate/mia_audit.py` (shadow-model MIA) | 🔲 not written | |
| `generate/` — Track A (DP-CTGAN/DP-TabDDPM + Opacus), Track B (vocab sampler), `pipeline.py` | 🔲 not written | directory has only `__init__.py` |
| `release/` — `bundler.py` (CSV + privacy certificate PDF + scorecard HTML) | 🔲 not written | directory has only `__init__.py` |
| `ui/app.py` (Streamlit UI) | 🔲 not written | directory has only `__init__.py` |
| `cli.py` (entry point) | 🔲 not written | commented out in `pyproject.toml` `[project.scripts]` |
| `.github/workflows/ci.yml` | 🔲 not written | `.github/workflows/` exists but is empty |

Run `git log --oneline -- <path>` before assuming a described module is the current reality — the README documents the *target* architecture, with planned files explicitly marked `[planned]`.

## Architecture (target — see Build status for what's actually implemented)

Mirrorbank is designed as a five-stage pipeline. Each stage produces an artifact consumed by the next.

### Stage 1 — Ingest and profile (`src/mirrorbank/profile/`) ✅
`SchemaProfiler` reads the input dataset and infers column types (continuous, categorical, datetime, free-text, identifier), detects likely PII fields, computes per-column statistics. When an `InstrumentSchema` is provided, schema-declared kinds (`ColumnKind`) take precedence over heuristic inference. The profile is the input contract for all downstream stages.

### Stage 2 — Dual-track generation (`src/mirrorbank/generate/`) 🔲 not yet written
Planned design: two models running in parallel, since no single architecture handles both numerical and text data well.

- **Track A:** DP-CTGAN baseline first (per the README roadmap), upgrading later to DP-TabDDPM. Handles all structured fields, trained with DP-SGD via Opacus, charged against the `PrivacyBudget`.
- **Track B (vocab sampler):** handles free-text fields (merchant names, memos), seeded from Track A's synthetic rows. **Must never see real data — consumes zero privacy budget.** This invariant is meant to be enforced at the pipeline level once written.

A `pipeline.py` is expected to orchestrate both tracks and merge their outputs.

### Stage 3 — Privacy budget accountant (`src/mirrorbank/privacy/budget.py`) ✅
`PrivacyBudget`/`BudgetConfig` wrap the training loop using Rényi DP (RDP) accounting. Every gradient step charges the budget; training halts (`BudgetExhausted`) when ε is exhausted. `calibrate_noise_multiplier()` auto-solves for σ given (ε, δ, training duration) — users specify ε, not σ. One budget instance per instrument; budgets never compose across instruments.

Presets: `tight` (ε=1), `balanced` (ε=3), `loose` (ε=5), `demo` (ε=10). All use δ=1e-5.

### Stage 4 — Evaluation gauntlet (`src/mirrorbank/evaluate/`) — partially done
`gauntlet.py` orchestrates the batteries into a `GauntletReport`:
- **Fidelity** (`fidelity.py`, ✅ done): per-column KS tests + correlation matrix distance + instrument business-rule checks via `InstrumentSchema.validate()` (e.g. wires must have zero weekend volume).
- **Utility** (`utility.py`, 🔲 not written): planned TSTR — train XGBoost on synthetic, test on held-out real. Target: `AUC_TSTR / AUC_TRTR ≥ 0.90`.
- **Privacy** (`mia_audit.py`, 🔲 not written): planned shadow-model MIA (Shokri et al. 2017), implemented from scratch. Target: attack AUC ≤ 0.55.

### Stage 5 — Release (`src/mirrorbank/release/`) 🔲 not yet written
Planned to bundle synthetic CSV, privacy certificate PDF, and utility scorecard HTML into a versioned release artifact. The Streamlit UI (`src/mirrorbank/ui/app.py`, also not yet written) is meant to provide one-click download.

## Instruments (`src/mirrorbank/instruments/`) ✅

Each payment instrument is a subclass of `InstrumentSchema` (`base.py`) declaring `name`, `display_name`, `fraud_label`, `columns` (each a `ColumnSpec` with a `ColumnKind`), and a `validate()` method. `training_columns()` / `text_columns()` route fields to Track A vs Track B based on `ColumnKind` and `is_pii`.

| Instrument | Key statistical constraints |
|---|---|
| `ach` | Bimodal amounts; Friday payroll spike; R01 return code is ~40% of returns |
| `check` | Log-normal amounts; 1–5 day clearing; check washing is dominant fraud type |
| `zelle` | Hard $2,500 cap enforced in `validate()`; peak 6–10 pm; scam-driven disputes |
| `wire` | Log-normal heavy tail; **zero weekend volume** (Fedwire closed); BEC fraud |
| `credit_card` | ~0.17% fraud rate; CNP 10× fraud rate vs card-present |
| `debit_card` | PIN vs signature distinction; overdraft field; higher fraud rate than credit |

`registry.py` maps instrument names → schema classes; `detect_instrument(df)` auto-detects type from column fingerprints (`sec_code` → ACH, `imad` → wire, `micr_line` → check, `sender_token` → Zelle, `mcc_code`+`pin_used` → debit, `mcc_code`+`card_present` → credit, etc. — order matters, most specific first).

## Reference data (`src/mirrorbank/reference/`) ✅

Generates syntactically valid but entirely fake identifiers — never DP-trained, assigned post-hoc:
- **Routing numbers** (`routing_numbers.py`): 9-digit ABA with correct check-digit (3/7/1 weighted sum over real Fed-district prefixes)
- **SWIFT/BIC codes** (`swift_codes.py`): 11-char BIC (BBBBCCLLXXX format)
- **ACH trace numbers, Fedwire IMAD, MICR lines** (`identifiers.py`): trace = routing[:8] + 7-digit sequence; IMAD = YYYYMMDD + 8-char bank-id + 6-digit sequence; MICR = `|routing|  account  check_number`

## Key design decisions

- **RDP over basic composition.** Tighter ε bounds for iterative mechanisms; plan to lean on Opacus's accountant once Track A exists.
- **LLM/vocab sampler outside the privacy boundary.** Only sees synthetic seeds. Never pass real records to Track B.
- **Diffusion over GANs (eventually).** Roadmap starts with a DP-CTGAN baseline, then upgrades to DP-TabDDPM for stability under DP-SGD (no adversarial training instability).
- **MIA implemented from scratch.** Interview-defensible — targeting ~200 lines covering shadow models, attack classifier, and AUC reporting.
- **Polars throughout** (convert to pandas only at the model boundary). 10–50× faster than pandas on 24M-row datasets.
- **One PrivacyBudget per instrument.** Budgets are independent and never compose across instruments.

## Datasets

- **IBM TabFormer** (default): 24M credit card transactions, downloaded via `huggingface_hub` — see `data/README.md`.
- **Amex Default Prediction**: 5.5M rows, 190 features, via Kaggle — alternative credit-card dataset.
- No public reference dataset exists for ACH/wire/Zelle/check; until a sample generator script is written, use `tests/conftest.py` fixtures as the schema reference for those instruments.
- Raw data → `data/raw/` (gitignored). Synthetic outputs → `data/synthetic/` and `outputs/` (gitignored). Only `data/sample/` is committed.

## Config files

Dataset-specific hyperparameters live in `configs/` (`tabformer.yaml`, `ach.yaml`, `wire.yaml`, `zelle.yaml`). Key fields: `instrument`, `dataset.{target_column,exclude_columns,...}`, `generation.{model,n_rows,batch_size,epochs,noise_multiplier,max_grad_norm}`, `privacy.{epsilon,delta,preset,accounting}`, `evaluation.*_target`.
