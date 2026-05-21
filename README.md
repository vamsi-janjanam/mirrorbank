# mirrorbank

> **Differentially private synthetic financial data — generate realistic transaction records with a mathematical privacy guarantee.**

Banks and fintech teams can't share real customer data with data scientists, vendors, or ML teams without legal and compliance risk. Mirrorbank generates *synthetic* data that is statistically indistinguishable from real data but contains no actual customer records — backed by a formal (ε, δ)-differential privacy certificate and a membership inference audit.

**Hero metric:** Generate 1M synthetic transactions at ε = 3 in under 10 minutes, with a fraud classifier trained on synthetic data reaching ≥ 90% of real-data AUC, and membership inference attack accuracy below 53%.

---

## Table of Contents

- [Core Concept](#core-concept)
- [Architecture](#architecture)
  - [Stage 1 — Ingest & Profile](#stage-1--ingest--profile)
  - [Stage 2 — Dual-Track Generation](#stage-2--dual-track-generation)
  - [Stage 3 — Privacy Budget Accountant](#stage-3--privacy-budget-accountant)
  - [Stage 4 — Evaluation Gauntlet](#stage-4--evaluation-gauntlet)
  - [Stage 5 — Release Bundle](#stage-5--release-bundle)
- [Key Design Decisions](#key-design-decisions)
- [Privacy Dashboard (UI)](#privacy-dashboard-ui)
- [Quick Start — Demo Dataset](#quick-start--demo-dataset)
- [Full Dataset Setup](#full-dataset-setup)
- [Configuration](#configuration)
- [Running Evaluations](#running-evaluations)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [Background Reading](#background-reading)

---

## Core Concept

### The Problem

Real financial transaction data is highly sensitive. A credit card dataset contains:
- Spending patterns that reveal medical conditions, religion, relationships
- Account numbers, names, and locations that are direct PII
- Fraud labels that are legally sensitive

Traditional anonymization (masking, k-anonymity) is broken — re-identification attacks routinely succeed on "anonymized" datasets. Teams either lock data behind slow access controls or work with unrepresentative samples.

### The Solution

Mirrorbank trains a generative model on real data and produces a *synthetic* dataset. The key guarantees:

1. **Statistical fidelity** — distributions, correlations, and rare-event rates match the real data well enough that ML models trained on synthetic data perform almost as well as those trained on real data.
2. **Differential privacy** — the training algorithm (DP-SGD) adds calibrated noise to every gradient update. This provides a mathematical proof that an attacker with unlimited compute cannot determine whether any specific individual's record was in the training set, beyond a probability bounded by ε.
3. **Audited** — a shadow-model membership inference attack (Shokri et al. 2017) is run as a black-box sanity check on top of the DP guarantee.

### What You Get Out

```
outputs/
├── synthetic_transactions.csv     # Drop-in replacement for your real dataset
├── privacy_certificate.pdf        # Formal (ε, δ) certificate + audit result
└── utility_scorecard.html         # KS tests, TSTR AUC, correlation heatmaps
```

---

## Architecture

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │                         MIRRORBANK PIPELINE                         │
 │                                                                     │
 │  Real CSV                                                           │
 │     │                                                               │
 │     ▼                                                               │
 │  ┌──────────────────┐                                               │
 │  │  Stage 1         │  Column types · PII flags · Statistics        │
 │  │  Profiler        │  Correlation graph · Schema contract          │
 │  └────────┬─────────┘                                               │
 │           │  schema + stats                                         │
 │           ▼                                                         │
 │  ┌─────────────────────────────────────────────────────┐           │
 │  │  Stage 2 — Dual-Track Generator                     │           │
 │  │                                                     │           │
 │  │  ┌──────────────────────┐  ┌──────────────────────┐│           │
 │  │  │  Track A             │  │  Track B             ││           │
 │  │  │  DP-CTGAN (baseline) │  │  Merchant name       ││           │
 │  │  │  → DP-TabDDPM (v2)  │  │  vocab sampler       ││           │
 │  │  │                      │  │  (zero budget cost)  ││           │
 │  │  │  Structured fields:  │  │  Free-text fields:   ││           │
 │  │  │  amounts, timestamps,│  │  merchant names,     ││           │
 │  │  │  MCC codes, geo,     │  │  descriptions        ││           │
 │  │  │  account features    │  │                      ││           │
 │  │  │                      │  │  Seeds come from     ││           │
 │  │  │  Trained with DP-SGD │  │  Track A output only ││           │
 │  │  │  via Opacus          │  │  (never real data)   ││           │
 │  │  └──────────┬───────────┘  └─────────┬────────────┘│           │
 │  │             └──────────┬─────────────┘             │           │
 │  └──────────────────────┬─┘─────────────────────────────           │
 │                         │  synthetic rows                          │
 │                         ▼                                          │
 │  ┌──────────────────────────────────────────────────┐              │
 │  │  Stage 3 — Privacy Budget Accountant             │              │
 │  │                                                  │              │
 │  │  RDP (Rényi DP) accounting via Opacus            │              │
 │  │  Charges ε per gradient step                     │              │
 │  │  Halts training when budget exhausted            │              │
 │  │  Live meter streamed to UI                       │              │
 │  │  Presets: ε ∈ {1, 3, 5, 10}, δ = 1e-5           │              │
 │  └──────────────────────┬───────────────────────────┘              │
 │                         │  (ε, δ) + trained model                  │
 │                         ▼                                          │
 │  ┌──────────────────────────────────────────────────┐              │
 │  │  Stage 4 — Evaluation Gauntlet                   │              │
 │  │                                                  │              │
 │  │  Fidelity   KS test per column (p > 0.05         │              │
 │  │             for ≥ 80% of columns)                │              │
 │  │             Correlation Frobenius norm < 0.15    │              │
 │  │                                                  │              │
 │  │  Utility    TSTR: train XGBoost on synthetic,    │              │
 │  │             test on held-out real                │              │
 │  │             Target: AUC_TSTR / AUC_TRTR ≥ 0.90  │              │
 │  │                                                  │              │
 │  │  Privacy    Shadow-model MIA (10 shadow models)  │              │
 │  │             Target: attack AUC ≤ 0.55            │              │
 │  └──────────────────────┬───────────────────────────┘              │
 │                         │  scores + audit result                   │
 │                         ▼                                          │
 │  ┌──────────────────────────────────────────────────┐              │
 │  │  Stage 5 — Release Bundle                        │              │
 │  │                                                  │              │
 │  │  synthetic_transactions.csv                      │              │
 │  │  privacy_certificate.pdf  (ε, δ, MIA badge)      │              │
 │  │  utility_scorecard.html   (charts + metrics)     │              │
 │  └──────────────────────────────────────────────────┘              │
 └─────────────────────────────────────────────────────────────────────┘
```

---

### Stage 1 — Ingest & Profile

**Module:** `src/mirrorbank/profile/`  
**Key class:** `SchemaProfiler`

Reads the input CSV and produces a **schema profile** that is the input contract for all downstream stages. No privacy budget is spent here — profiling only reads aggregate statistics.

| Output | Description |
|--------|-------------|
| Column types | `continuous`, `categorical`, `datetime`, `free-text`, `identifier` |
| PII flags | Columns likely containing personal identifiers (name, email, account number) |
| Per-column stats | mean, std, min, max, cardinality, null rate, top-K values |
| Correlation graph | Pairwise Pearson/Cramér's V matrix — used to verify fidelity post-generation |

**Design note:** PII detection is heuristic (regex + column name matching). Flagged columns are excluded from generation by default, or replaced with synthetic identifiers.

---

### Stage 2 — Dual-Track Generation

**Module:** `src/mirrorbank/generate/`

Two generators run in parallel because no single architecture handles both numerical and text fields well.

#### Track A — Structured Fields (DP-CTGAN → DP-TabDDPM)

**Files:** `ctgan.py`, `tabddpm.py`, `dp_trainer.py`

**v1 baseline — DP-CTGAN:**
CTGAN uses a conditional GAN with mode-specific normalization for mixed-type tabular data. Differential privacy is applied by wrapping the discriminator's optimizer with Opacus's `PrivacyEngine`, which:
1. Clips per-sample gradients to a maximum L2 norm
2. Adds Gaussian noise proportional to the clipping norm and noise multiplier σ
3. Charges the RDP accountant after each step

```
Gradient update with DP-SGD:
  g_i     = ∇ L(θ; x_i)               # per-sample gradient
  g̃_i     = g_i / max(1, ||g_i||₂/C)  # clip to norm C
  g̃       = (1/B) Σ g̃_i + N(0, σ²C²I) # average + Gaussian noise
  θ       = θ - η · g̃                  # update
```

**v2 upgrade — DP-TabDDPM:**
Diffusion model that outperforms CTGAN/TVAE on mixed-type tabular data (especially rare-event columns like fraud labels) and is more stable under DP-SGD because there is no adversarial training dynamic.

#### Track B — Free-Text Fields (Vocabulary Sampler)

**File:** `vocab_sampler.py`

Merchant names and transaction descriptions are sampled from a vocabulary built from the real dataset (merchant names are largely public information tied to MCC codes — not private). Takes synthetic structured rows from Track A as seeds to condition sampling (e.g., a synthetic row with MCC=5411 gets a grocery merchant name).

**Critical invariant:** Track B never receives real records. It operates on synthetic seeds only and consumes zero privacy budget. This is enforced at the pipeline level.

> **Future extension (v3):** Replace the vocab sampler with an LLM narrator that takes synthetic seeds and generates coherent text. The privacy boundary stays the same — the LLM only ever sees synthetic data.

---

### Stage 3 — Privacy Budget Accountant

**Module:** `src/mirrorbank/privacy/budget.py`  
**Key class:** `PrivacyBudget`

Wraps the training loop. Uses **Rényi Differential Privacy (RDP)** accounting, which gives tighter ε bounds than basic composition for iterative mechanisms like DP-SGD.

```
For a mechanism with noise multiplier σ and batch sampling rate q = B/N:
  ε_RDP(α) = α / (2σ²)    (Gaussian mechanism, simplified)

After T steps, compose and convert to (ε, δ)-DP:
  ε(δ) = min_α [ ε_RDP(α) · T + log(1/δ) / (α-1) ]
```

Every gradient step:
1. Charges the RDP accountant (Opacus implementation)
2. Converts running RDP budget to (ε, δ) and emits it as a live metric
3. Halts training if ε exceeds the configured target

**Supported presets:**

| Preset | ε | δ | Privacy level |
|--------|---|---|---------------|
| `tight` | 1 | 1e-5 | High — noticeable utility loss |
| `balanced` | 3 | 1e-5 | Recommended — good utility/privacy tradeoff |
| `loose` | 5 | 1e-5 | Moderate — better utility |
| `demo` | 10 | 1e-5 | Low — for quick demos only |

---

### Stage 4 — Evaluation Gauntlet

**Module:** `src/mirrorbank/evaluate/`  
**Orchestrator:** `gauntlet.py`

Three independent test batteries. All targets must pass for the release to be certified.

#### Fidelity (`fidelity.py`)

Checks that synthetic data looks like real data column by column and jointly.

| Test | Method | Target |
|------|--------|--------|
| Per-column distributions | Two-sample KS test | p > 0.05 for ≥ 80% of columns |
| Joint correlations | Frobenius norm of (C_real − C_synth) | < 0.15 |
| Rare event rate | Absolute difference in fraud prevalence | < 0.5% |

#### Utility (`utility.py`) — TSTR

**Train on Synthetic, Test on Real** — the gold standard for synthetic data utility.

1. Train an XGBoost fraud classifier on the synthetic dataset
2. Evaluate on a held-out real test set (never used in generation)
3. Compare against a baseline trained on real data (TRTR)

```
Utility ratio = AUC_TSTR / AUC_TRTR ≥ 0.90
```

#### Privacy Audit (`mia_audit.py`) — Shadow Model MIA

Implements the Shokri et al. 2017 shadow-model membership inference attack **from scratch** (not a library call — important for interview defensibility).

```
Attack procedure:
1. Train N=10 shadow generative models on disjoint subsets of real data
2. For each shadow model, label synthetic records as "in" or "out"
3. Train a binary attack classifier (LightGBM) on shadow model outputs
4. Evaluate attack classifier on the target model's synthetic data
5. Report attack AUC — random chance = 0.50

Target: attack AUC ≤ 0.55  (5% above chance is the accepted threshold)
```

---

### Stage 5 — Release Bundle

**Module:** `src/mirrorbank/release/`

Packages all artifacts into a versioned, timestamped release:

```
release_20240315_eps3_balanced/
├── synthetic_transactions.csv     # The synthetic dataset
├── privacy_certificate.pdf        # Formal certificate (see below)
└── utility_scorecard.html         # Interactive charts
```

**Privacy certificate contents:**
- Dataset name, row count, generation timestamp
- Final (ε, δ) values with accounting method (RDP)
- DP-SGD hyperparameters (noise multiplier σ, clipping norm C, batch size B)
- MIA audit result — PASS / FAIL badge with attack AUC
- Signed hash of the synthetic CSV

---

## Key Design Decisions

### Why RDP instead of basic composition?
Rényi DP composes tighter than basic DP for sequences of Gaussian mechanisms. For a training run of 10,000 steps, RDP can give ε ≈ 3 where basic composition would give ε ≈ 30. This directly translates to better model utility at the same privacy level.

### Why diffusion (TabDDPM) over GANs for v2?
DP-SGD is applied to the discriminator in GANs, which already receives noisy gradients in adversarial training. The interaction between gradient noise and adversarial instability causes training collapse more often under DP. Diffusion models have a single stable training objective and respond more predictably to DP-SGD noise.

### Why build MIA from scratch?
ML Privacy Meter and similar libraries work, but you cannot whiteboard them in an interview. The shadow-model attack is ~200 lines of readable code. Being able to explain every design choice (why shadow models? why LightGBM for the attack classifier? what does attack AUC vs. accuracy measure?) is more valuable than a library import.

### Why keep Track B simple (vocab sampler, not LLM) for v1?
Merchant names are largely public (tied to MCC codes). An LLM adds API cost, latency, and complexity without improving the privacy story — the key invariant (LLM never sees real data) is the same either way. The vocab sampler ships in a day; the LLM narrator is an extension for v3.

### Why a baseline before the best model?
DP-CTGAN is well-understood, faster to train, and easier to debug than TabDDPM. Having a working baseline means the evaluation gauntlet, UI, and release pipeline can all be built and tested before the harder model work begins.

---

## Privacy Dashboard (UI)

**Module:** `src/mirrorbank/ui/app.py`  
**Run:** `uv run streamlit run src/mirrorbank/ui/app.py`

The Streamlit UI is structured around the privacy certificate — that is the deliverable that matters to a compliance or engineering audience.

```
┌─────────────────────────────────────────────────────────────┐
│  mirrorbank                              [Upload Dataset]   │
├──────────────────────┬──────────────────────────────────────┤
│  PRIVACY BUDGET      │  GENERATION PROGRESS                 │
│                      │                                      │
│  ε target:  3.0      │  ████████████░░░░  62%  Step 6,200   │
│  ε spent:   1.87  ◄──┤── live meter                        │
│  δ:         1e-5     │                                      │
│                      │  ETA: 3m 42s                         │
│  [=========    ]     │                                      │
│   62% of budget      │                                      │
├──────────────────────┴──────────────────────────────────────┤
│  EVALUATION RESULTS (after generation)                      │
│                                                             │
│  Fidelity    KS pass rate:  84%   ✓   Corr norm:  0.11  ✓  │
│  Utility     TSTR AUC:      0.923     TRTR AUC:   0.961     │
│              Ratio: 96.0%   ✓                               │
│  Privacy     MIA AUC: 0.512          [  PASS  ]   ✓        │
├─────────────────────────────────────────────────────────────┤
│  [Download synthetic_transactions.csv]                      │
│  [Download privacy_certificate.pdf  ]                       │
│  [Download utility_scorecard.html   ]                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start — Demo Dataset

A pre-packaged 10,000-row sample is included in the repo so you can run the full pipeline in under 2 minutes without downloading the full IBM TabFormer dataset.

```bash
# Clone and install
git clone https://github.com/your-username/mirrorbank.git
cd mirrorbank
uv sync

# Run on demo dataset (no download required)
uv run python -m mirrorbank.cli generate \
  --input data/sample/transactions_10k.csv \
  --epsilon 10 \
  --preset demo \
  --rows 10000 \
  --output outputs/demo_run

# Or launch the UI
uv run streamlit run src/mirrorbank/ui/app.py

# Or with Docker (recommended)
docker compose up
# Open http://localhost:8501
```

---

## Full Dataset Setup

### IBM TabFormer (default)

24M credit card transactions. See `data/README.md` for download instructions.

```bash
# After downloading and placing at data/raw/tabformer/
uv run python -m mirrorbank.cli generate \
  --input data/raw/tabformer/card_transaction.v1.csv \
  --config configs/tabformer.yaml \
  --epsilon 3 \
  --rows 1000000 \
  --output outputs/tabformer_eps3
```

### Amex Default Prediction (Kaggle)

5.5M rows, 190 features. Available at `kaggle competitions download -c amex-default-prediction`.

```bash
uv run python -m mirrorbank.cli generate \
  --input data/raw/amex/train_data.csv \
  --config configs/amex.yaml \
  --epsilon 3 \
  --output outputs/amex_eps3
```

---

## Configuration

Dataset-specific hyperparameters live in `configs/`. Key knobs:

```yaml
# configs/tabformer.yaml
dataset:
  name: tabformer
  target_column: Is Fraud?
  datetime_columns: [Timestamp]
  categorical_columns: [Use Chip, Merchant Name, MCC]
  exclude_columns: [User, Card]        # PII — excluded from generation

generation:
  model: dp_ctgan                      # dp_ctgan | dp_tabddpm
  n_rows: 1_000_000
  batch_size: 4096
  noise_multiplier: 1.1                # σ — higher = more DP noise = lower utility
  max_grad_norm: 1.0                   # C — gradient clipping norm
  epochs: 300

privacy:
  epsilon: 3.0
  delta: 1.0e-5
  accounting: rdp                      # rdp | gdp

evaluation:
  n_shadow_models: 10                  # MIA shadow model count
  attack_model: lightgbm
  tstr_classifier: xgboost
```

---

## Running Evaluations

```bash
# Run the full evaluation gauntlet
make gauntlet DATASET=tabformer SYNTH=dp_ctgan_eps3

# Run individual batteries
uv run python -m mirrorbank.evaluate.fidelity \
  --real data/raw/tabformer/card_transaction.v1.csv \
  --synth outputs/tabformer_eps3/synthetic_transactions.csv

uv run python -m mirrorbank.evaluate.utility \
  --real data/raw/tabformer/card_transaction.v1.csv \
  --synth outputs/tabformer_eps3/synthetic_transactions.csv \
  --target "Is Fraud?"

uv run python -m mirrorbank.evaluate.mia_audit \
  --real data/raw/tabformer/card_transaction.v1.csv \
  --synth outputs/tabformer_eps3/synthetic_transactions.csv \
  --n-shadow 10

# Run all tests
uv run pytest

# Lint
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

---

## Project Structure

```
mirrorbank/
├── configs/
│   ├── tabformer.yaml          # IBM TabFormer hyperparameters
│   └── amex.yaml               # Amex dataset hyperparameters
│
├── data/
│   ├── README.md               # Download instructions for full datasets
│   ├── sample/
│   │   └── transactions_10k.csv  # Pre-packaged demo dataset (in repo)
│   ├── raw/                    # Full datasets (gitignored)
│   └── synthetic/              # Generated outputs (gitignored)
│
├── src/mirrorbank/
│   ├── cli.py                  # Command-line entry point
│   │
│   ├── profile/
│   │   ├── __init__.py
│   │   └── schema_profiler.py  # Column typing, PII detection, statistics
│   │
│   ├── generate/
│   │   ├── __init__.py
│   │   ├── pipeline.py         # Orchestrates Track A + Track B, merges output
│   │   ├── ctgan.py            # DP-CTGAN (v1 baseline)
│   │   ├── tabddpm.py          # DP-TabDDPM (v2 upgrade)
│   │   ├── dp_trainer.py       # DP-SGD training loop (Opacus wrapper)
│   │   └── vocab_sampler.py    # Free-text field sampler (Track B)
│   │
│   ├── privacy/
│   │   ├── __init__.py
│   │   └── budget.py           # RDP accountant, training halt logic, live meter
│   │
│   ├── evaluate/
│   │   ├── __init__.py
│   │   ├── gauntlet.py         # Runs all three batteries, collects results
│   │   ├── fidelity.py         # KS tests, correlation distance
│   │   ├── utility.py          # TSTR with XGBoost
│   │   └── mia_audit.py        # Shadow-model membership inference attack
│   │
│   ├── release/
│   │   ├── __init__.py
│   │   ├── bundler.py          # Packages CSV + PDF + HTML into versioned release
│   │   ├── certificate.py      # Generates privacy_certificate.pdf
│   │   └── scorecard.py        # Generates utility_scorecard.html
│   │
│   └── ui/
│       ├── __init__.py
│       └── app.py              # Streamlit dashboard with live privacy meter
│
├── notebooks/
│   └── compare_real_vs_synthetic.ipynb  # Visual side-by-side: histograms,
│                                         # correlation heatmaps, t-SNE
│
├── tests/
│   ├── test_profiler.py
│   ├── test_budget.py
│   ├── test_fidelity.py
│   ├── test_utility.py
│   ├── test_mia_audit.py
│   └── conftest.py             # Shared fixtures (tiny synthetic dataset)
│
├── .github/
│   └── workflows/
│       └── ci.yml              # Lint + tests + gauntlet on every PR
│
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── pyproject.toml
├── CLAUDE.md
└── README.md
```

---

## Roadmap

| Week | Milestone | Status |
|------|-----------|--------|
| 1–2 | `SchemaProfiler` + DP-CTGAN baseline + CSV output | Planned |
| 3–4 | MIA audit (scratch implementation) + fidelity metrics (KS test) | Planned |
| 5 | TSTR utility evaluation (XGBoost) + gauntlet runner | Planned |
| 6 | Streamlit UI with live privacy meter + release bundle (PDF cert) | Planned |
| 7 | Upgrade generator to DP-TabDDPM | Planned |
| 8 | Demo dataset, comparison notebook, Docker, CI, README polish | Planned |

---

## Background Reading

| Topic | Resource |
|-------|----------|
| Differential Privacy (intro) | [Dwork & Roth, "The Algorithmic Foundations of DP" (2014)](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf) |
| Rényi DP & tight composition | [Mironov, "Rényi Differential Privacy" (2017)](https://arxiv.org/abs/1702.07476) |
| DP-SGD | [Abadi et al., "Deep Learning with DP" (2016)](https://arxiv.org/abs/1607.00133) |
| TabDDPM | [Kotelnikov et al., "TabDDPM" (2022)](https://arxiv.org/abs/2209.15421) |
| Membership Inference | [Shokri et al., "MIA against ML models" (2017)](https://arxiv.org/abs/1610.05820) |
| Opacus (DP-SGD library) | [pytorch.org/opacus](https://opacus.ai/) |
| CTGAN | [Xu et al., "Modeling Tabular Data using CTGAN" (2019)](https://arxiv.org/abs/1907.00503) |

---

## License

MIT
