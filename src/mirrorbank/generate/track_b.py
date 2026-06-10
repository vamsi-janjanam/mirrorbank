"""Stage 2, Track B — vocab sampler for free-text columns.

PRIVACY INVARIANT: this module must NEVER see real data. `generate_text_columns`
only ever receives `structured`, an already-synthetic (Track A output) frame,
plus a small static built-in vocabulary. It consumes zero privacy budget.
"""

from __future__ import annotations

import random

import polars as pl

from mirrorbank.instruments.base import InstrumentSchema

# ── Static built-in vocab, keyed by keyword found in the column name ────────

_VOCAB: dict[str, list[str]] = {
    "company_name": ["Acme Corp", "Netflix", "Employer Inc", "Globex LLC", "Initech"],
    "merchant_name": [
        "Amazon",
        "Walmart",
        "Starbucks",
        "Target",
        "Shell Gas",
        "Whole Foods",
        "Best Buy",
    ],
    "payee_name": ["John Smith", "Jane Doe", "City Utilities", "Acme Landlord LLC"],
    "originator_name": ["Acme Corp Treasury", "Globex Holdings", "John Smith"],
    "beneficiary_name": ["Globex Supplier Ltd", "Jane Doe", "Initech Vendor"],
    "bank_name": ["First National Bank", "Chase", "Wells Fargo", "Bank of America"],
    "name": ["John Smith", "Jane Doe", "Alex Lee", "Maria Garcia"],
    "memo": [
        "Rent payment",
        "Invoice settlement",
        "Monthly subscription",
        "Loan repayment",
        "Gift",
        "Reimbursement",
    ],
    "desc": [
        "PAYMENT",
        "PAYROLL",
        "SUBSCRIPTION",
        "INVOICE",
        "TRANSFER",
    ],
    "remittance_info": [
        "Invoice #1042 payment",
        "Wire transfer for services rendered",
        "Loan disbursement",
        "Supplier payment",
    ],
    "category": ["groceries", "utilities", "entertainment", "travel", "dining"],
}

_GENERIC_FALLBACK = ["Generic Co", "Sample Inc", "Demo LLC", "Misc Vendor"]


def _vocab_for(col: str) -> list[str]:
    lower = col.lower()
    for keyword, words in _VOCAB.items():
        if keyword in lower:
            return words
    return _GENERIC_FALLBACK


def generate_text_columns(
    structured: pl.DataFrame, schema: InstrumentSchema, *, seed: int = 0
) -> dict[str, list]:
    """Generate free-text column values derived only from `structured` + static vocab.

    For each column in `schema.text_columns()`, returns a list of length
    `structured.height`. Optionally seeds off a structured "category"-like
    column when present (e.g. joining a category value into a memo).
    """
    rng = random.Random(seed)
    n = structured.height
    text_cols = schema.text_columns()
    if not text_cols or n == 0:
        return {col: [] for col in text_cols} if text_cols and n == 0 else {}

    result: dict[str, list] = {}
    for col in text_cols:
        vocab = _vocab_for(col)
        values: list[str] = []
        for i in range(n):
            base = rng.choice(vocab)
            if "memo" in col.lower():
                # Try to enrich the memo with a structured category-like value.
                extra = None
                for cand in ("transaction_type", "category", "sec_code", "mcc_code"):
                    if cand in structured.columns:
                        extra = structured[cand][i]
                        break
                if extra is not None:
                    base = f"{base} - {extra}"
            values.append(base)
        result[col] = values

    return result
