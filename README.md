# mirrorbank

> **Differentially private synthetic financial data — generate realistic transaction records across six payment instruments with a mathematical privacy guarantee.**

Banks and fintech teams can't share real customer data with data scientists, vendors, or ML teams without legal and compliance risk. Mirrorbank generates *synthetic* data that is statistically indistinguishable from real data but contains no actual customer records — backed by a formal (ε, δ)-differential privacy certificate and a membership inference audit.

**Hero metric:** Generate 1M synthetic transactions at ε = 3 in under 10 minutes, with a fraud classifier trained on synthetic data reaching ≥ 90% of real-data AUC, and membership inference attack accuracy below 53%.

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
- [Full Dataset Setup](#full-dataset-setup)
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
outputs/
├── synthetic_transactions.csv     # Drop-in replacement for your real dataset
├── privacy_certificate.pdf        # Formal (ε, δ) certificate + MIA audit badge
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
 │  │  Track A (DP-CTGAN → DP-TabDDPM)                   │      │
 │  │  Structured fields: amounts, timestamps,           │      │
 │  │  codes, velocity features                          │      │
 │  │  Trained with DP-SGD via Opacus                    │      │
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
 │  │  privacy_certificate.pdf  (ε, δ, MIA badge)        │      │
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

**v1 baseline — DP-CTGAN:** CTGAN discriminator wrapped with Opacus `PrivacyEngine`. Clips per-sample gradients to norm C, adds Gaussian noise σ, charges the RDP accountant each step.

**v2 upgrade — DP-TabDDPM:** Denoising diffusion model for mixed-type tabular data. Outperforms CTGAN/TVAE on fidelity and is more stable under DP-SGD (single training objective — no adversarial instability).

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
Train XGBoost on synthetic data, evaluate on held-out real data. Target: `AUC_TSTR / AUC_TRTR ≥ 0.90` (instrument-specific targets in `configs/`).

#### Privacy — Shadow Model MIA
10 shadow generative models, binary attack classifier, attack AUC reported. Target: ≤ 0.55. Implemented from scratch — no library dependency.

---

### Stage 5 — Release Bundle

```
release_20240315_ach_eps3_balanced/
├── synthetic_transactions.csv
├── privacy_certificate.pdf    # (ε, δ), noise multiplier σ, MIA badge, CSV hash
└── utility_scorecard.html     # KS results, TSTR chart, correlation heatmaps
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/vamsi-janjanam/mirrorbank.git
cd mirrorbank

# Install dependencies
pipenv install

# Install dev tools (pytest, ruff)
pipenv install --dev

# Run tests
pipenv run pytest

# Launch the UI
pipenv run streamlit run src/mirrorbank/ui/app.py
# Open http://localhost:8501

# List available instruments
make instruments
```

---

## Full Dataset Setup

### IBM TabFormer — Credit Card (default)

24M credit card transactions. See `data/README.md` for download instructions.

```bash
pipenv run python -m mirrorbank.cli generate \
  --input data/raw/tabformer/card_transaction.v1.csv \
  --instrument credit_card \
  --config configs/tabformer.yaml \
  --epsilon 3 \
  --rows 1000000 \
  --output outputs/tabformer_eps3
```

### ACH, Wire, Zelle

```bash
# ACH
pipenv run python -m mirrorbank.cli generate \
  --input data/raw/ach/transactions.csv \
  --instrument ach \
  --config configs/ach.yaml \
  --epsilon 3 \
  --output outputs/ach_eps3

# Wire
pipenv run python -m mirrorbank.cli generate \
  --input data/raw/wire/transactions.csv \
  --instrument wire \
  --config configs/wire.yaml \
  --epsilon 3 \
  --output outputs/wire_eps3

# Zelle
pipenv run python -m mirrorbank.cli generate \
  --input data/raw/zelle/transactions.csv \
  --instrument zelle \
  --config configs/zelle.yaml \
  --epsilon 3 \
  --output outputs/zelle_eps3
```

### Auto-detect instrument from column names

```bash
# Detects type from column fingerprints:
# 'sec_code' → ACH  ·  'imad' → wire  ·  'sender_token' → Zelle  ·  'micr_line' → check
pipenv run python -m mirrorbank.cli generate \
  --input data/raw/unknown/transactions.csv \
  --epsilon 3 \
  --output outputs/auto_eps3
```

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
# Full gauntlet
make gauntlet DATASET=ach SYNTH=dp_ctgan_eps3

# Individual batteries
pipenv run python -m mirrorbank.evaluate.fidelity \
  --real data/raw/ach/transactions.csv \
  --synth outputs/ach_eps3/synthetic_transactions.csv \
  --instrument ach

pipenv run python -m mirrorbank.evaluate.utility \
  --real data/raw/ach/transactions.csv \
  --synth outputs/ach_eps3/synthetic_transactions.csv \
  --instrument ach --target is_fraud

pipenv run python -m mirrorbank.evaluate.mia_audit \
  --real data/raw/ach/transactions.csv \
  --synth outputs/ach_eps3/synthetic_transactions.csv \
  --n-shadow 10

# Run all tests
pipenv run pytest

# Lint / format
pipenv run ruff check src/ tests/
pipenv run ruff format src/ tests/
```

---

## Privacy Dashboard

**Run:** `pipenv run streamlit run src/mirrorbank/ui/app.py`

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
│  [Download privacy_certificate.pdf  ]                        │
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
~200 lines covering shadow models, attack classifier, and AUC reporting. Every design choice (why shadow models? why LightGBM? AUC vs accuracy?) can be whiteboarded in an interview.

### Track B outside the privacy boundary
Text generation is conditioned on synthetic structured fields — never on real records. The text step consumes zero privacy budget regardless of whether a vocab sampler or an LLM is used for Track B.

### Reference data for regulated identifiers
ABA routing numbers and SWIFT BIC codes have rigid check-digit constraints a generative model cannot reliably learn. They are generated post-hoc by `reference/routing_numbers.py` (correct ABA check-digit math) and `reference/swift_codes.py`.

---

## Project Structure

```
mirrorbank/
├── Pipfile                         # Python dependencies
├── Pipfile.lock                    # Pinned dependency graph
├── Makefile                        # install · test · lint · gauntlet · ui
├── pyproject.toml                  # Package metadata, ruff + pytest config
│
├── configs/
│   ├── tabformer.yaml
│   ├── ach.yaml
│   ├── wire.yaml
│   └── zelle.yaml
│
├── data/
│   ├── README.md
│   ├── sample/
│   ├── raw/                        # gitignored
│   └── synthetic/                  # gitignored
│
├── src/mirrorbank/
│   ├── instruments/
│   │   ├── base.py                 # InstrumentSchema, ColumnSpec, ColumnKind
│   │   ├── registry.py             # REGISTRY + detect_instrument()
│   │   ├── ach.py
│   │   ├── check.py
│   │   ├── zelle.py                # validate() enforces $2,500 cap
│   │   ├── wire.py                 # validate() enforces zero weekend volume
│   │   ├── credit_card.py
│   │   └── debit_card.py
│   │
│   ├── reference/
│   │   ├── routing_numbers.py      # ABA check-digit math
│   │   ├── swift_codes.py          # SWIFT BIC format
│   │   └── identifiers.py          # ACH trace, Fedwire IMAD, MICR, check numbers
│   │
│   ├── profile/
│   │   └── schema_profiler.py
│   │
│   ├── generate/
│   │   ├── pipeline.py
│   │   ├── ctgan.py
│   │   ├── tabddpm.py
│   │   ├── dp_trainer.py
│   │   └── vocab_sampler.py
│   │
│   ├── privacy/
│   │   ├── budget.py               # PrivacyBudget, BudgetConfig, calibrate_noise_multiplier
│   │   └── certificate.py
│   │
│   ├── evaluate/
│   │   ├── gauntlet.py
│   │   ├── fidelity.py
│   │   ├── utility.py
│   │   └── mia_audit.py
│   │
│   ├── release/
│   │   ├── bundler.py
│   │   └── scorecard.py
│   │
│   └── ui/
│       └── app.py
│
├── tests/
│   ├── conftest.py                 # Fixtures: one DataFrame per instrument
│   ├── test_instruments.py
│   ├── test_reference.py
│   ├── test_budget.py
│   ├── test_profiler.py
│   └── test_fidelity.py
│
└── .github/
    └── workflows/
        └── ci.yml
```

---

## Roadmap

| Milestone | Status |
|---|---|
| 6 instrument schemas (ACH, check, Zelle, wire, credit card, debit card) | ✅ Done |
| Reference data — routing numbers, SWIFT, IMAD, MICR | ✅ Done |
| `SchemaProfiler` — column type inference + PII detection | ✅ Done |
| `PrivacyBudget` — RDP accountant + noise calibration | ✅ Done |
| Fidelity evaluation — KS + correlation + instrument rules | ✅ Done |
| 48 passing unit tests | ✅ Done |
| DP-CTGAN baseline — Opacus integration, end-to-end pipeline | 🔲 Next |
| Vocab sampler (Track B) | 🔲 Next |
| TSTR utility evaluation + MIA audit | 🔲 Next |
| Streamlit UI with live ε meter | 🔲 Next |
| Privacy certificate PDF + HTML scorecard | 🔲 Next |
| DP-TabDDPM upgrade | 🔲 Next |
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
