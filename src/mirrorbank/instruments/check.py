"""Paper check instrument schema."""

from __future__ import annotations

from mirrorbank.instruments.base import ColumnKind, ColumnSpec, InstrumentSchema
from mirrorbank.reference.routing_numbers import generate_routing_number
from mirrorbank.reference.identifiers import generate_micr_line


class CheckSchema(InstrumentSchema):
    """
    Paper check schema.

    Statistical fingerprint:
    - Amount: log-normal, right-skewed. Business checks much larger than personal.
    - Days to clear: 1 business day local, 2–5 out-of-state, up to 7 for large amounts.
    - Return rate: ~0.5 % (lower than ACH).
    - Fraud types: check washing is surging — large amounts, usually within 2 days of issue.
    """

    name = "check"
    display_name = "Paper Check"
    fraud_label = "is_fraud"

    columns = [
        # ── Core ─────────────────────────────────────────────────────────────
        ColumnSpec(
            "amount",
            ColumnKind.CONTINUOUS,
            description="Heavy right tail — most checks $50–2 000, some very large business checks",
        ),
        ColumnSpec("date_written", ColumnKind.DATETIME),
        ColumnSpec(
            "date_cleared",
            ColumnKind.DATETIME,
            nullable=True,
            description="Null if not yet cleared or returned before clearing",
        ),
        ColumnSpec(
            "days_to_clear",
            ColumnKind.CONTINUOUS,
            nullable=True,
            description="1–5 business days; longer for large checks or new accounts",
        ),
        ColumnSpec(
            "check_type",
            ColumnKind.CATEGORICAL,
            description="'personal', 'business', 'cashier', 'certified', 'money_order'",
        ),
        # ── Parties ───────────────────────────────────────────────────────────
        ColumnSpec(
            "payee_name",
            ColumnKind.FREE_TEXT,
            description="Who the check is made out to",
        ),
        ColumnSpec("memo", ColumnKind.FREE_TEXT, nullable=True),
        ColumnSpec("bank_name", ColumnKind.FREE_TEXT, description="Drawee bank name"),
        # ── MICR / routing ────────────────────────────────────────────────────
        ColumnSpec(
            "bank_routing",
            ColumnKind.REFERENCE,
            reference_generator=generate_routing_number,
        ),
        ColumnSpec("payor_account", ColumnKind.IDENTIFIER, is_pii=True),
        ColumnSpec("check_number", ColumnKind.IDENTIFIER),
        ColumnSpec(
            "micr_line",
            ColumnKind.REFERENCE,
            description="Full MICR line: routing + account + check number",
            reference_generator=generate_micr_line,
        ),
        # ── Return ────────────────────────────────────────────────────────────
        ColumnSpec(
            "is_returned",
            ColumnKind.CATEGORICAL,
            description="Bounced / returned unpaid",
        ),
        ColumnSpec(
            "return_reason",
            ColumnKind.CATEGORICAL,
            nullable=True,
            description="NSF, Account Closed, Stop Payment, Refer to Maker, Forgery",
        ),
        # ── Fraud ─────────────────────────────────────────────────────────────
        ColumnSpec("is_fraud", ColumnKind.CATEGORICAL),
        ColumnSpec(
            "fraud_type",
            ColumnKind.CATEGORICAL,
            nullable=True,
            description="check_washing, counterfeit, altered_payee, stolen, forgery",
        ),
    ]

    def validate(self, df) -> list[str]:
        errors: list[str] = []
        bad = df.filter(
            (df["is_returned"]) & (df["return_reason"].is_null())
        ).height
        if bad > 0:
            errors.append(f"Check: {bad} returned items are missing return_reason")
        return errors
