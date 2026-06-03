# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Mirrorbank is a differentially private synthetic financial data generator. Given a sensitive tabular dataset, it produces a synthetic CSV, a privacy certificate (ε, δ values + MIA audit result), and a utility scorecard (KS tests, TSTR AUC). Supports six payment instruments: ACH, check, Zelle, wire, credit card, and debit card.

**Hero metric:** Generate 1M synthetic transactions at ε=3 in under 10 minutes, with a fraud classifier trained on synthetic data reaching ≥90% of real-data AUC, and membership inference attack accuracy below 53%.

## Commands

```bash
# Install dependencies
pipenv install

# Install dev dependencies (pytest, ruff)
pipenv install --dev

# Install DP generation stack (torch, opacus, sdv)
pipenv install --categories generate

# Run the Streamlit UI
pipenv run streamlit run src/mirrorbank/ui/app.py

# Run the full evaluation gauntlet
make gauntlet DATASET=tabformer SYNTH=dp_tabddpm_eps3

# Run all tests
pipenv run pytest

# Run a single test file
pipenv run pytest tests/test_budget.py

# Run a single test
pipenv run pytest tests/test_budget.py::test_name

# Lint / format
pipenv run ruff check src/ tests/
pipenv run ruff format src/ tests/

# List available payment instruments
make instruments
```

## Architecture

Mirrorbank is a five-stage pipeline. Each stage produces an artifact consumed by the next.

### Stage 1 — Ingest and profile (`src/mirrorbank/profile/`)
`SchemaProfiler` reads the input dataset and infers column types (continuous, categorical, datetime, free-text, identifier), detects likely PII fields, computes per-column statistics. When an `InstrumentSchema` is provided, schema-declared kinds take precedence over heuristic inference. The profile is the input contract for all downstream stages.

### Stage 2 — Dual-track generation (`src/mirrorbank/generate/`)
Two models run in parallel because no single architecture handles both numerical and text data well.

- **Track A (DP-TabDDPM):** `tabddpm.py` + `dp_trainer.py`. Handles all structured fields. Trained with DP-SGD via Opacus.
- **Track B (vocab sampler):** `vocab_sampler.py`. Handles free-text fields (merchant names, memos). Takes synthetic rows from Track A as seeds. **Never sees real data — consumes zero privacy budget.** This invariant is enforced at the pipeline level.

`pipeline.py` orchestrates both tracks and merges their outputs.

### Stage 3 — Privacy budget accountant (`src/mirrorbank/privacy/budget.py`)
`PrivacyBudget` wraps the training loop using Rényi DP (RDP) accounting. Every gradient step charges the budget; training halts (`BudgetExhausted`) when ε is exhausted. `calibrate_noise_multiplier()` auto-solves for σ given (ε, δ, training duration) — users specify ε, not σ. One budget instance per instrument; budgets never compose across instruments.

Presets: `tight` (ε=1), `balanced` (ε=3), `loose` (ε=5), `demo` (ε=10). All use δ=1e-5.

### Stage 4 — Evaluation gauntlet (`src/mirrorbank/evaluate/`)
Three independent batteries run via `gauntlet.py`:
- **Fidelity** (`fidelity.py`): per-column KS tests + correlation matrix distance + instrument business-rule checks (e.g. wires must have zero weekend volume).
- **Utility** (`utility.py`): TSTR — train XGBoost on synthetic, test on held-out real. Target: `AUC_TSTR / AUC_TRTR ≥ 0.90`.
- **Privacy** (`mia_audit.py`): shadow-model MIA (Shokri et al. 2017), implemented from scratch. Target: attack AUC ≤ 0.55.

### Stage 5 — Release (`src/mirrorbank/release/`)
Bundles synthetic CSV, privacy certificate PDF, and utility scorecard HTML into a versioned release artifact. The Streamlit UI (`src/mirrorbank/ui/app.py`) provides one-click download.

## Instruments (`src/mirrorbank/instruments/`)

Each payment instrument is a subclass of `InstrumentSchema` declaring its columns, PII flags, and `validate()` rules.

| Instrument | Key statistical constraints |
|---|---|
| `ach` | Bimodal amounts; Friday payroll spike; R01 return code is ~40% of returns |
| `check` | Log-normal amounts; 1–5 day clearing; check washing is dominant fraud type |
| `zelle` | Hard $2,500 cap enforced in `validate()`; peak 6–10 pm; scam-driven disputes |
| `wire` | Log-normal heavy tail; **zero weekend volume** (Fedwire closed); BEC fraud |
| `credit_card` | ~0.17% fraud rate; CNP 10× fraud rate vs card-present |
| `debit_card` | PIN vs signature distinction; overdraft field; higher fraud rate than credit |

`detect_instrument(df)` auto-detects type from column fingerprints (`sec_code` → ACH, `imad` → wire, etc.).

## Reference data (`src/mirrorbank/reference/`)

Generates syntactically valid but entirely fake identifiers — never DP-trained, assigned post-hoc:
- **Routing numbers:** 9-digit ABA with correct check-digit (3/7/1 weighted sum)
- **SWIFT codes:** 11-char BIC (BBBBCCLLXXX format)
- **ACH trace numbers:** routing[:8] + 7-digit sequence
- **Fedwire IMAD:** YYYYMMDD + 8-char bank-id + 6-digit sequence
- **MICR lines:** `|routing|  account  check_number`

## Key design decisions

- **RDP over basic composition.** Tighter ε bounds for iterative mechanisms. Opacus's accountant.
- **LLM/vocab sampler outside the privacy boundary.** Only sees synthetic seeds. Never pass real records to Track B.
- **Diffusion over GANs.** TabDDPM is more stable under DP-SGD (no adversarial training instability).
- **MIA implemented from scratch.** Interview-defensible — ~200 lines covering shadow models, attack classifier, and AUC reporting.
- **Polars throughout** (convert to pandas only at the model boundary). 10–50× faster on 24M-row datasets.
- **One PrivacyBudget per instrument.** Budgets are independent and never compose across instruments.

## Datasets

- **IBM TabFormer** (default): 24M credit card transactions. Download instructions in `data/README.md`.
- **Amex Default Prediction**: 5.5M rows, 190 features. Available on Kaggle.
- Raw data → `data/raw/` (gitignored). Generated outputs → `data/synthetic/`.

## Config files

Dataset-specific hyperparameters live in `configs/` (`tabformer.yaml`, `ach.yaml`, `wire.yaml`, `zelle.yaml`). Key fields: `instrument`, `generation.model`, `privacy.epsilon`, `privacy.preset`, `evaluation.*_target`.

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs lint, tests, and the evaluation gauntlet on every PR.
