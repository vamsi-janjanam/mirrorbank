# mirrorbank

> **Differentially private synthetic financial data — generate realistic transaction records across six payment instruments with a mathematical privacy guarantee.**

Banks and fintech teams can't share real customer data with data scientists, vendors, or ML teams without legal and compliance risk. Mirrorbank generates *synthetic* data that is statistically indistinguishable from real data but contains no actual customer records — backed by a formal (ε, δ)-differential privacy certificate and a membership inference audit.

**Hero metric:** Generate 1M synthetic transactions at ε = 3 in under 10 minutes, with a fraud classifier trained on synthetic data reaching ≥ 90% of real-data AUC, and membership inference attack accuracy below 53%.

> 🆕 **New here?** Read [GETTING_STARTED.md](GETTING_STARTED.md) for a plain-English tour of what works today and how to try it in 30 seconds.

---

## Supported Payment Instruments

| Instrument | Key fields | Fraud label | Notes |
|---|---|---|---|
| **ACH Transfer** | amount, sec_code, settlement_date, return_code | `is_fraud` | Bimodal amounts; Friday payroll spike; R01 = 40% of returns |
| **Paper Check** | amount, date_written, days_to_clear, micr_line | `is_fraud` | Log-normal; 1–5 day clearing; check washing is surging |
| **Zelle P2P** | amount, direction, is_new_recipient, account_age_days | `is_disputed` | Hard $2,500 cap; scam-driven disputes; real-time settlement |
| **Wire Transfer** | amount, network, imad, ofac_screened, sar_reason | `is_suspicious` | Log-normal heavy tail; zero weekend volume (Fedwire closed) |
| **Credit Card** | amount, mcc_code, entry_mode, card_present | `is_fraud` | ~0.17% fraud rate; CNP has 10× fraud rate vs card-present |
| **Debit Card** | amount, pin_used, is_overdraft, transaction_type | `is_fraud` | PIN vs signature; overdraft field; higher fraud rate than credit |

---

## Table of Contents

- [Core Concept](#core-concept)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [CLI](#cli)
- [Configuration](#configuration)
- [Running Evaluations](#running-evaluations)
- [Privacy Dashboard](#privacy-dashboard)
- [Key Design Decisions](#key-design-decisions)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [Background Reading](#background-reading)

---

## Core Concept

### The Problem

Real financial transaction data is highly sensitive. A credit card dataset contains spending patterns that reveal medical conditions, religion, and relationships. ACH files contain employer payroll details. Wire records expose business counterparties. Teams either lock data behind slow access controls or work with unrepresentative samples that produce unreliable models.

Traditional anonymization (masking, k-anonymity) is broken — re-identification attacks routinely succeed on "anonymized" datasets.

### The Solution

Mirrorbank trains a generative model on real data and produces a *synthetic* dataset. Three guarantees:

1. **Statistical fidelity** — distributions, correlations, and rare-event rates match the real data well enough that ML models trained on synthetic data perform almost as well as those trained on real data.
2. **Differential privacy** — the training algorithm (DP-SGD) adds calibrated Gaussian noise to every gradient update, providing a mathematical proof that an attacker with unlimited compute cannot determine whether any specific individual's record was in the training set, beyond a probability bounded by ε.
3. **Audited** — a shadow-model membership inference attack (Shokri et al. 2017) is run as a black-box sanity check on top of the DP guarantee.

### What You Get Out

```
outputs/release/
├── synthetic_transactions.csv     # Drop-in replacement for your real dataset
├── privacy_certificate.html       # Formal (ε, δ) certificate + MIA audit badge
└── utility_scorecard.html         # KS tests, TSTR AUC, correlation heatmaps
```

---

## Architecture

```
 ┌──────────────────────────────────────────────────────────────┐
 │                     MIRRORBANK PIPELINE                      │
 │                                                              │
 │  Real CSV  +  Instrument type (ach | check | zelle |         │
 │                                wire | credit_card | debit)   │
 │     │                                                        │
 │     ▼                                                        │
 │  ┌────────────────────────────────────────────────────┐      │
 │  │  Stage 1 — Profiler                                │      │
 │  │  Column types · PII flags · Stats · Corr graph     │      │
 │  │  InstrumentSchema takes precedence over heuristics │      │
 │  └──────────────────────────┬─────────────────────────┘      │
 │                             │  schema + stats                │
 │                             ▼                                │
 │  ┌────────────────────────────────────────────────────┐      │
 │  │  Stage 2 — Dual-Track Generator                    │      │
 │  │                                                    │      │
 │  │  Track A: DP statistical (default) | DP-CTGAN     │      │
 │  │  Structured fields: amounts, timestamps,           │      │
 │  │  codes, velocity features                          │      │
 │  │  DP marginals (RDP) or DP-SGD via Opacus           │      │
 │  │                                                    │      │
 │  │  Track B (Vocab Sampler)                           │      │
 │  │  Free-text fields: merchant names, memos           │      │
 │  │  Seeds come from Track A only — never real data    │      │
 │  │  Consumes zero privacy budget                      │      │
 │  └──────────────────────────┬─────────────────────────┘      │
 │                             │  synthetic rows                │
 │                             ▼                                │
 │  ┌────────────────────────────────────────────────────┐      │
 │  │  Stage 3 — Privacy Budget Accountant               │      │
 │  │  RDP accounting · Halts at ε target                │      │
 │  │  One budget per instrument — never composed        │      │
 │  │  Presets: tight (ε=1) · balanced (ε=3) · demo (ε=10)│    │
 │  └──────────────────────────┬─────────────────────────┘      │
 │                             │  (ε, δ) + trained model        │
 │                             ▼                                │
 │  ┌────────────────────────────────────────────────────┐      │
 │  │  Stage 4 — Evaluation Gauntlet                     │      │
 │  │  Fidelity  KS test per column · Corr Frobenius     │      │
 │  │            + instrument business-rule checks       │      │
 │  │  Utility   TSTR AUC / TRTR AUC ≥ 0.90             │      │
 │  │  Privacy   Shadow-model MIA · attack AUC ≤ 0.55   │      │
 │  └──────────────────────────┬─────────────────────────┘      │
 │                             ▼                                │
 │  ┌────────────────────────────────────────────────────┐      │
 │  │  Stage 5 — Release Bundle                          │      │
 │  │  synthetic_transactions.csv                        │      │
 │  │  privacy_certificate.html (ε, δ, MIA badge)        │      │
 │  │  utility_scorecard.html   (charts + metrics)       │      │
 │  └────────────────────────────────────────────────────┘      │
 └──────────────────────────────────────────────────────────────┘
```

---

### Stage 1 — Ingest & Profile

**Module:** `src/mirrorbank/profile/schema_profiler.py`

Reads the input CSV and produces a `DatasetProfile` used by all downstream stages. When an `InstrumentSchema` is provided, declared column kinds (continuous / categorical / datetime / free-text / identifier / reference) take precedence over heuristic inference. PII detection uses column-name heuristics (`account`, `name`, `ssn`, `email`, etc.) — flagged columns are excluded from DP training.

---

### Stage 2 — Dual-Track Generation

#### Track A — Structured Fields

Two interchangeable engines, selected with `--model`:

**`dp_statistical` (default):** Per-column DP marginals — noisy histograms for continuous/datetime fields and noisy value-counts for categoricals, with support preservation so structural gaps (e.g. zero weekend wires) survive the noise. Fast, dependency-light (no torch), and charges the RDP accountant directly. *Limitation:* models columns independently, so cross-column correlations are not preserved — the correlation-distance fidelity check will often fail on highly-correlated schemas.

**`dp_ctgan`:** A from-scratch CTGAN whose discriminator (the only network that touches real data) is wrapped with the Opacus `PrivacyEngine` — per-sample gradient clipping + Gaussian noise, charged to the RDP accountant. The generator trains normally and consumes zero budget. Preserves correlations better but needs `torch` + `opacus` and yields looser ε on small datasets.

**v2 upgrade (planned) — DP-TabDDPM:** Denoising diffusion model for mixed-type tabular data. More stable under DP-SGD (single training objective — no adversarial instability).

#### Track B — Free-Text Fields (Vocab Sampler)

Samples text fields (merchant names, memos, company descriptions) conditioned on synthetic structured context from Track A (e.g., MCC code → merchant category → plausible merchant name).

**Critical invariant:** Track B never receives real records. It only sees synthetic seeds from Track A. It consumes zero privacy budget. This is enforced at the pipeline level.

---

### Stage 3 — Privacy Budget Accountant

**Module:** `src/mirrorbank/privacy/budget.py`

Uses **Rényi Differential Privacy (RDP)** accounting — tighter than basic composition for iterative mechanisms like DP-SGD.

```
ε(δ) = min_α [ ε_RDP(α) + log(1/δ) / (α − 1) ]
```

`calibrate_noise_multiplier()` auto-solves for σ via binary search so users specify ε (what they want), not σ (an internal knob). One `PrivacyBudget` instance per instrument — budgets never compose across instruments.

| Preset | ε | δ | Use case |
|---|---|---|---|
| `tight` | 1 | 1e-5 | Maximum privacy, noticeable utility loss |
| `balanced` | 3 | 1e-5 | Recommended — strong privacy, good utility |
| `loose` | 5 | 1e-5 | Better utility, moderate privacy |
| `demo` | 10 | 1e-5 | Quick demos only |

---

### Stage 4 — Evaluation Gauntlet

**Module:** `src/mirrorbank/evaluate/`

#### Fidelity
| Test | Target |
|---|---|
| Per-column KS test | p > 0.05 for ≥ 80% of columns |
| Correlation Frobenius norm | < 0.15 |
| Instrument business rules | Zero violations (e.g. no weekend wires, Zelle ≤ $2,500) |

#### Utility — TSTR (Train Synthetic, Test Real)
Train a RandomForest classifier on synthetic data, evaluate on held-out real data, and compare against the same model trained on real data (TRTR). Target: `AUC_TSTR / AUC_TRTR ≥ 0.90` (instrument-specific targets in `configs/`). Returns `None` when the schema has no usable fraud label or too few minority examples.

#### Privacy — Membership Inference Audit
A nearest-neighbour distance attack (`run_mia`): for each real record, its distance to the nearest synthetic record is compared against held-out reals, and the separability is scored as an attack AUC. Target: ≤ 0.55. Implemented from scratch (scikit-learn `NearestNeighbors`). This is a v1 heuristic; a full shadow-model MIA (Shokri et al. 2017) remains on the roadmap.

---

### Stage 5 — Release Bundle

**Module:** `src/mirrorbank/release/bundler.py` — `bundle_release()` writes a versioned directory:

```
outputs/release/
├── synthetic_transactions.csv
├── privacy_certificate.html   # (ε, δ), MIA audit badge, instrument, row count
└── utility_scorecard.html     # KS results, TSTR/MIA metrics, fidelity verdict
```

Certificates are self-contained HTML (no PDF toolchain / extra dependencies). The CLI `mirrorbank release` runs generate → evaluate → bundle in one command.

---

## Quick Start

```bash
# Clone
git clone https://github.com/vamsi-janjanam/mirrorbank.git
cd mirrorbank

# Install core dependencies
pipenv install

# Install dev tools (pytest, ruff)
pipenv install --dev

# Run tests — all 129 should pass
pipenv run pytest

# Lint the codebase
pipenv run ruff check src/ tests/

# List available instruments
make instruments

# Write the bundled demo CSVs into data/sample/ (real + synthetic per instrument)
make sample-data

# Generate synthetic data end-to-end from a sample dataset
pipenv run mirrorbank generate data/sample/ach_real.csv --instrument ach --rows 1000

# Launch the Streamlit dashboard
make ui        # or: pipenv run streamlit run src/mirrorbank/ui/app.py
```

> **The full pipeline is functional** — generation (`dp_statistical` default,
> optional `dp_ctgan`), profiler, privacy accountant, evaluation gauntlet
> (fidelity + utility + MIA), release bundler, CLI, and Streamlit UI all work.
> `dp_ctgan` additionally requires `torch` + `opacus`
> (`pipenv install --categories generate`).

---

## CLI

The `mirrorbank` console script (installed by `pipenv install`) exposes the whole
pipeline:

```bash
mirrorbank instruments                       # list supported instruments
mirrorbank profile  <csv> [-i INSTRUMENT]    # column types, PII flags, stats
mirrorbank budget   [--preset balanced]      # show (ε, δ) + calibrated σ
mirrorbank generate <csv> [options]          # real CSV  → synthetic CSV
mirrorbank evaluate --real R.csv --synth S.csv   # run the gauntlet
mirrorbank release  <csv> [options]          # generate → evaluate → bundle
```

Common `generate` / `release` options: `-i/--instrument` (auto-detected if
omitted), `--preset {tight,balanced,loose,demo}` (default `balanced`),
`--rows N` (default 10,000), `--model {dp_statistical,dp_ctgan}` (default
`dp_statistical`), `--seed`, `--out` / `--out-dir`.

### Examples

```bash
# Quick synthetic ACH dataset from a committed sample
mirrorbank generate data/sample/ach_real.csv --instrument ach --rows 5000

# Full release bundle (CSV + privacy cert + scorecard) at ε=3
mirrorbank release data/sample/wire_real.csv -i wire --preset balanced --rows 10000

# Neural engine (needs torch + opacus)
mirrorbank generate data/sample/credit_card_real.csv -i credit_card --model dp_ctgan
```

### Full datasets

For the 24M-row IBM TabFormer credit-card set, see `data/README.md` for download
instructions, then point `generate` at the downloaded CSV. Instrument type is
auto-detected from column fingerprints when `--instrument` is omitted:
`sec_code` → ACH · `imad` → wire · `sender_token` → Zelle · `micr_line` → check.

---

## Configuration

Dataset-specific hyperparameters in `configs/`:

```yaml
# configs/ach.yaml (example)
instrument: ach

dataset:
  target_column: is_fraud
  datetime_columns: [settlement_date, effective_date, return_date]
  exclude_columns: [originator_account, receiver_account]  # PII

generation:
  model: dp_ctgan           # dp_ctgan | dp_tabddpm
  n_rows: 500_000
  batch_size: 2048
  epochs: 200
  max_grad_norm: 1.0
  # noise_multiplier omitted → auto-calibrated from epsilon target

privacy:
  epsilon: 3.0
  delta: 1.0e-5
  preset: balanced          # tight | balanced | loose | demo
  accounting: rdp

evaluation:
  n_shadow_models: 10
  utility_ratio_target: 0.88
  mia_auc_target: 0.55
```

---

## Running Evaluations

```bash
# Run all unit tests (129 passing)
pipenv run pytest

# Lint / format
pipenv run ruff check src/ tests/
pipenv run ruff format src/ tests/

# Gauntlet on a real vs synthetic CSV pair (fidelity + utility + MIA)
mirrorbank evaluate --real data/sample/wire_real.csv --synth data/sample/wire_synth.csv -i wire
```

The underlying Python functions (`run_fidelity`, `run_utility`, `run_mia`,
`run_gauntlet`) are also directly importable from `mirrorbank.evaluate`.

---

## Privacy Dashboard

A Streamlit UI (`src/mirrorbank/ui/app.py`) wraps the pipeline with tabs for
Overview, Generate, Schema, Profiler, Privacy Budget, and Fidelity. Every tab
works with **zero setup** via built-in sample data, or you can upload your own
CSV. Generation streams to disk for large row counts.

**Run:** `make ui` (or `pipenv run streamlit run src/mirrorbank/ui/app.py`)

```
┌──────────────────────────────────────────────────────────────┐
│  mirrorbank                             [Upload Dataset]     │
├──────────────────────────────────────────────────────────────┤
│  INSTRUMENT                                                  │
│  ● ACH  ○ Wire  ○ Zelle  ○ Check  ○ Credit Card  ○ Debit    │
├──────────────────────┬───────────────────────────────────────┤
│  PRIVACY BUDGET      │  GENERATION PROGRESS                  │
│                      │                                       │
│  ε target:  3.0      │  ████████████░░░░  62%  Step 6,200    │
│  ε spent:   1.87  ───┤── live meter                         │
│  δ:         1e-5     │  ETA: 3m 42s                          │
│  [=========    ]     │                                       │
├──────────────────────┴───────────────────────────────────────┤
│  EVALUATION RESULTS                                          │
│  Fidelity    KS pass rate:  84%  ✓   Corr norm:  0.11  ✓    │
│              Instrument checks:  ✓   0 violations            │
│  Utility     TSTR AUC: 0.907        TRTR AUC:  0.961         │
│              Ratio: 94.4%        ✓                           │
│  Privacy     MIA AUC: 0.512         [  PASS  ]   ✓           │
├──────────────────────────────────────────────────────────────┤
│  [Download synthetic_transactions.csv]                       │
│  [Download privacy_certificate.html ]                        │
│  [Download utility_scorecard.html   ]                        │
└──────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### One model per instrument, not one unified model
ACH amounts peak at $50–$5,000. Wire amounts peak at $50,000–$500,000. A unified model requires extreme normalization and produces neither well. Each instrument trains its own DP model with its own independent privacy budget.

### RDP over basic composition
For a 10,000-step training run, RDP gives ε ≈ 3 where basic composition gives ε ≈ 30. This directly translates to better utility at the same privacy level.

### Noise multiplier auto-calibration
`calibrate_noise_multiplier()` binary-searches for the σ that lands on the ε target after the full training run. Users specify ε; σ is derived automatically.

### Diffusion over GANs for v2
DP-SGD applied to a GAN discriminator interacts badly with adversarial training instability, causing frequent collapse under tight noise budgets. Diffusion models have a single stable loss and respond predictably.

### MIA implemented from scratch
The v1 audit is a nearest-neighbour distance attack (no library beyond scikit-learn's `NearestNeighbors`): if synthetic records sit suspiciously close to specific training records, an attacker can distinguish members from non-members. Reported as an attack AUC against a held-out set. Every design choice (why distance? AUC vs accuracy? held-out calibration?) is interview-defensible; a full shadow-model MIA is the planned upgrade.

### Track B outside the privacy boundary
Text generation is conditioned on synthetic structured fields — never on real records. The text step consumes zero privacy budget regardless of whether a vocab sampler or an LLM is used for Track B.

### Reference data for regulated identifiers
ABA routing numbers and SWIFT BIC codes have rigid check-digit constraints a generative model cannot reliably learn. They are generated post-hoc by `reference/routing_numbers.py` (correct ABA check-digit math) and `reference/swift_codes.py`.

---

## Project Structure

Files marked **[planned]** are on the roadmap but not yet written.

```
mirrorbank/
├── Pipfile                         # Python dependencies
├── Pipfile.lock                    # Pinned dependency graph
├── Makefile                        # install · test · lint · gauntlet · ui
├── pyproject.toml                  # Package metadata, ruff + pytest config
│
├── configs/
│   ├── tabformer.yaml              # Credit card (IBM TabFormer)
│   ├── ach.yaml                    # ACH transfers
│   ├── wire.yaml                   # Wire transfers
│   └── zelle.yaml                  # Zelle P2P payments
│
├── data/
│   ├── README.md                   # Dataset download instructions
│   ├── sample/                     # Small demo datasets
│   ├── raw/                        # Full datasets (gitignored)
│   └── synthetic/                  # Generated outputs (gitignored)
│
├── src/mirrorbank/
│   ├── instruments/                # ✅ Implemented
│   │   ├── base.py                 # InstrumentSchema, ColumnSpec, ColumnKind
│   │   ├── registry.py             # REGISTRY + detect_instrument()
│   │   ├── ach.py                  # validate(): zero-amount, return_code consistency
│   │   ├── check.py                # validate(): return_reason consistency
│   │   ├── zelle.py                # validate(): $2,500 cap, dispute_reason
│   │   ├── wire.py                 # validate(): zero weekend volume (Fedwire)
│   │   ├── credit_card.py
│   │   └── debit_card.py
│   │
│   ├── reference/                  # ✅ Implemented
│   │   ├── routing_numbers.py      # ABA check-digit math (3/7/1 weighted sum)
│   │   ├── swift_codes.py          # SWIFT BIC format (BBBBCCLLXXX)
│   │   └── identifiers.py          # ACH trace, Fedwire IMAD, MICR, check numbers
│   │
│   ├── profile/                    # ✅ Implemented
│   │   └── schema_profiler.py      # SchemaProfiler → DatasetProfile
│   │
│   ├── privacy/                    # ✅ Implemented
│   │   └── budget.py               # PrivacyBudget, BudgetConfig, calibrate_noise_multiplier
│   │
│   ├── evaluate/                   # ✅ Implemented
│   │   ├── gauntlet.py             # Orchestrator → fidelity + utility + MIA
│   │   ├── fidelity.py             # KS tests, correlation distance, instrument checks
│   │   ├── utility.py              # TSTR (RandomForest) AUC ratio
│   │   └── mia_audit.py            # Nearest-neighbour membership inference audit
│   │
│   ├── generate/                   # ✅ Implemented
│   │   ├── pipeline.py             # generate() + generate_to_csv() (streaming); Track A + B
│   │   ├── track_a.py              # DP statistical generator (noisy marginals) — default
│   │   ├── ctgan.py                # DP-CTGAN (Opacus-wrapped discriminator) — optional
│   │   └── track_b.py              # Free-text vocab sampler (synthetic seeds only)
│   │
│   ├── release/                    # ✅ Implemented
│   │   └── bundler.py              # bundle_release() → CSV + cert HTML + scorecard HTML
│   │
│   ├── sample_data.py              # ✅ sample_dataset() — built-in demo data per instrument
│   ├── cli.py                      # ✅ mirrorbank CLI (instruments/profile/budget/generate/evaluate/release)
│   │
│   └── ui/                         # ✅ Implemented
│       └── app.py                  # Streamlit dashboard (sample data + upload)
│
├── scripts/
│   └── generate_sample_data.py     # Writes data/sample/*.csv (real + synthetic per instrument)
│
├── tests/                          # 129 passing
│   ├── conftest.py                 # Fixtures delegate to sample_dataset()
│   ├── test_instruments.py         # schemas, registry, validate(), detection
│   ├── test_reference.py           # check-digit math, SWIFT, IMAD, MICR
│   ├── test_budget.py              # RDP accounting, presets, calibration
│   ├── test_profiler.py            # column inference, PII detection
│   ├── test_fidelity.py            # KS, correlation, gauntlet orchestrator
│   ├── test_generate_tracks.py     # Track A + Track B
│   ├── test_generate_pipeline.py   # pipeline + streaming to CSV
│   ├── test_ctgan.py               # DP-CTGAN end-to-end (skipped without torch)
│   ├── test_utility.py             # TSTR utility
│   ├── test_mia.py                 # MIA audit
│   ├── test_release.py             # release bundler
│   └── test_cli.py                 # CLI commands
│
└── .github/
    └── workflows/
        └── ci.yml                  # [planned] Lint + tests on every PR
```

---

## Roadmap

| Milestone | Status |
|---|---|
| 6 instrument schemas (ACH, check, Zelle, wire, credit card, debit card) | ✅ Done |
| Reference data — routing numbers (check-digit), SWIFT, IMAD, MICR | ✅ Done |
| `SchemaProfiler` — column type inference + PII detection | ✅ Done |
| `PrivacyBudget` — RDP accountant + `BudgetExhausted` + noise calibration | ✅ Done |
| Fidelity evaluation — KS tests + correlation + instrument business rules | ✅ Done |
| `GauntletReport` orchestrator (fidelity + utility + MIA) | ✅ Done |
| DP statistical generator (Track A default) + streaming to disk | ✅ Done |
| DP-CTGAN baseline — Opacus-wrapped discriminator, end-to-end pipeline | ✅ Done |
| Vocab sampler (Track B) — synthetic seeds, zero privacy budget | ✅ Done |
| TSTR utility evaluation (`utility.py`) + MIA audit (`mia_audit.py`) | ✅ Done |
| Release bundler — CSV + privacy certificate HTML + scorecard HTML | ✅ Done |
| `mirrorbank` CLI (profile / budget / generate / evaluate / release) | ✅ Done |
| Streamlit UI with built-in sample data + upload | ✅ Done |
| 129 passing unit tests (`pipenv run pytest`) | ✅ Done |
| Full shadow-model MIA (Shokri et al. 2017) | 🔲 Next |
| DP-TabDDPM upgrade | 🔲 Next |
| GitHub Actions CI (`.github/workflows/ci.yml`) | 🔲 Next |
| Docker + Hugging Face Spaces deployment | 🔲 Next |

---

## Background Reading

| Topic | Resource |
|---|---|
| Differential Privacy (intro) | [Dwork & Roth, "Algorithmic Foundations of DP" (2014)](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf) |
| Rényi DP & tight composition | [Mironov, "Rényi Differential Privacy" (2017)](https://arxiv.org/abs/1702.07476) |
| DP-SGD | [Abadi et al., "Deep Learning with DP" (2016)](https://arxiv.org/abs/1607.00133) |
| TabDDPM | [Kotelnikov et al., "TabDDPM" (2022)](https://arxiv.org/abs/2209.15421) |
| Membership Inference | [Shokri et al., "MIA against ML models" (2017)](https://arxiv.org/abs/1610.05820) |
| Opacus (DP-SGD library) | [opacus.ai](https://opacus.ai/) |
| CTGAN | [Xu et al., "Modeling Tabular Data using CTGAN" (2019)](https://arxiv.org/abs/1907.00503) |

---

## License

MIT
