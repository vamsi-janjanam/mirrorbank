"""Debit card transaction instrument schema."""

from __future__ import annotations

from mirrorbank.instruments.base import ColumnKind, ColumnSpec, InstrumentSchema


class DebitCardSchema(InstrumentSchema):
    """
    Debit card transaction schema.

    Similar to credit card but key differences:
    - Draws from checking account balance → overdraft is possible.
    - ATM withdrawals are a first-class transaction type.
    - PIN transactions (fully authenticated) vs signature transactions.
    - Higher average fraud rate than credit (fewer consumer protections).
    - No cash_advance or balance_transfer transaction types.
    """

    name = "debit_card"
    display_name = "Debit Card Transaction"
    fraud_label = "is_fraud"

    columns = [
        ColumnSpec("amount", ColumnKind.CONTINUOUS, description="Purchase or withdrawal amount in USD"),
        ColumnSpec("timestamp", ColumnKind.DATETIME),
        ColumnSpec(
            "transaction_type",
            ColumnKind.CATEGORICAL,
            description="'purchase', 'refund', 'atm_withdrawal', 'atm_deposit'",
        ),
        # ── Merchant / ATM ────────────────────────────────────────────────────
        ColumnSpec("mcc_code", ColumnKind.CATEGORICAL, nullable=True, description="Null for ATM transactions"),
        ColumnSpec("merchant_name", ColumnKind.FREE_TEXT, nullable=True),
        ColumnSpec("merchant_city", ColumnKind.CATEGORICAL),
        ColumnSpec("merchant_state", ColumnKind.CATEGORICAL),
        ColumnSpec("merchant_country", ColumnKind.CATEGORICAL),
        # ── Channel ───────────────────────────────────────────────────────────
        ColumnSpec(
            "entry_mode",
            ColumnKind.CATEGORICAL,
            description="'pin', 'signature', 'contactless', 'ecommerce', 'atm'",
        ),
        ColumnSpec(
            "pin_used",
            ColumnKind.CATEGORICAL,
            description="Boolean — PIN transactions have lower fraud rate than signature",
        ),
        ColumnSpec("card_present", ColumnKind.CATEGORICAL),
        ColumnSpec("is_international", ColumnKind.CATEGORICAL),
        # ── Account / balance ─────────────────────────────────────────────────
        ColumnSpec(
            "is_overdraft",
            ColumnKind.CATEGORICAL,
            description="Boolean — transaction caused or increased an overdraft",
        ),
        ColumnSpec(
            "overdraft_fee",
            ColumnKind.CONTINUOUS,
            nullable=True,
            description="Fee charged for overdraft (typically $25–$35)",
        ),
        # ── Authorization ─────────────────────────────────────────────────────
        ColumnSpec(
            "response_code",
            ColumnKind.CATEGORICAL,
            description="'00'=approved, '51'=insufficient funds, '05'=do not honor",
        ),
        # ── Velocity ──────────────────────────────────────────────────────────
        ColumnSpec("txn_count_1h", ColumnKind.CONTINUOUS),
        ColumnSpec("txn_amount_1h", ColumnKind.CONTINUOUS),
        # ── Fraud ─────────────────────────────────────────────────────────────
        ColumnSpec(
            "is_fraud",
            ColumnKind.CATEGORICAL,
            description="Boolean. Higher fraud rate than credit (~0.3 %) due to fewer protections.",
        ),
    ]
