"""ACH (Automated Clearing House) transfer instrument schema."""

from __future__ import annotations

from mirrorbank.instruments.base import ColumnKind, ColumnSpec, InstrumentSchema
from mirrorbank.reference.identifiers import generate_trace_number
from mirrorbank.reference.routing_numbers import generate_routing_number


class ACHSchema(InstrumentSchema):
    """
    ACH transfer schema.

    Statistical fingerprint to preserve in generated data:
    - Amount: bimodal — small recurring ($10–$200) and payroll ($500–$15 000)
    - Settlement day: Friday has ~3× the volume of Monday (payroll cycle)
    - Return rate: ~1–2 % overall; R01 (NSF) is ~40 % of returns
    - SEC code mix: PPD ~60 %, CCD ~25 %, WEB ~10 %, TEL ~3 %, IAT ~2 %
    - Fraud: low volume, high value — business email compromise, account takeover
    """

    name = "ach"
    display_name = "ACH Transfer"
    fraud_label = "is_fraud"

    columns = [
        # ── Core transaction ──────────────────────────────────────────────────
        ColumnSpec(
            "amount",
            ColumnKind.CONTINUOUS,
            description="Transfer amount in USD. Bimodal: small recurring ($10–200) + payroll ($500–15 000). Negative = debit entry.",
        ),
        ColumnSpec(
            "transaction_type",
            ColumnKind.CATEGORICAL,
            description="'credit' or 'debit'",
        ),
        ColumnSpec(
            "sec_code",
            ColumnKind.CATEGORICAL,
            description="Standard Entry Class code: PPD (personal), CCD (corporate), WEB (internet), TEL (telephone), IAT (international)",
        ),
        ColumnSpec(
            "settlement_date",
            ColumnKind.DATETIME,
            description="Business date funds settle. Strong weekday pattern — payroll on Fridays, recurring on 1st/15th.",
        ),
        ColumnSpec("effective_date", ColumnKind.DATETIME),
        ColumnSpec(
            "days_to_settle",
            ColumnKind.CONTINUOUS,
            description="Typically 1–2 business days",
        ),
        # ── Originator ────────────────────────────────────────────────────────
        ColumnSpec(
            "company_name",
            ColumnKind.FREE_TEXT,
            description="Company originating the transfer — employer name, biller name",
        ),
        ColumnSpec(
            "company_entry_desc",
            ColumnKind.FREE_TEXT,
            description="Short memo visible to receiver: 'PAYROLL', 'INSURANCE PMT', 'NETFLIX.COM'",
        ),
        ColumnSpec(
            "odfi_routing",
            ColumnKind.REFERENCE,
            reference_generator=generate_routing_number,
        ),
        ColumnSpec("originator_account", ColumnKind.IDENTIFIER, is_pii=True),
        # ── Receiver ──────────────────────────────────────────────────────────
        ColumnSpec(
            "rdfi_routing",
            ColumnKind.REFERENCE,
            reference_generator=generate_routing_number,
        ),
        ColumnSpec(
            "receiver_account_type",
            ColumnKind.CATEGORICAL,
            description="'checking' or 'savings'",
        ),
        ColumnSpec("receiver_account", ColumnKind.IDENTIFIER, is_pii=True),
        # ── Trace number ──────────────────────────────────────────────────────
        ColumnSpec(
            "trace_number",
            ColumnKind.REFERENCE,
            description="15-digit ACH trace: routing(8) + sequence(7)",
            reference_generator=generate_trace_number,
        ),
        # ── Return / exception ────────────────────────────────────────────────
        ColumnSpec(
            "is_returned",
            ColumnKind.CATEGORICAL,
            description="Boolean. ~1–2 % of ACH transactions are returned.",
        ),
        ColumnSpec(
            "return_code",
            ColumnKind.CATEGORICAL,
            nullable=True,
            description="R01=Insufficient Funds, R02=Account Closed, R03=No Account, R07=Auth Revoked, R10=Unauthorized, R29=Corp Not Authorized",
        ),
        ColumnSpec(
            "return_date",
            ColumnKind.DATETIME,
            nullable=True,
        ),
        # ── Risk / fraud ──────────────────────────────────────────────────────
        ColumnSpec(
            "risk_score",
            ColumnKind.CONTINUOUS,
            description="0–1000. Derived from velocity, amount, account age.",
        ),
        ColumnSpec(
            "is_fraud",
            ColumnKind.CATEGORICAL,
            description="Boolean. ACH fraud is low volume but high value — BEC, account takeover.",
        ),
    ]

    def validate(self, df) -> list[str]:
        errors: list[str] = []
        if "amount" in df.columns and df.filter(df["amount"] == 0).height > 0:
            errors.append("ACH: zero-amount transactions found")
        if "is_returned" in df.columns and "return_code" in df.columns:
            bad_returns = df.filter(
                (df["is_returned"]) & (df["return_code"].is_null())
            ).height
            if bad_returns > 0:
                errors.append(
                    f"ACH: {bad_returns} returned transactions are missing return_code"
                )
        return errors
