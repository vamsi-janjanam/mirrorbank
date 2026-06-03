"""
Payment-network identifier generators.

All identifiers produced here are syntactically valid but entirely fake.
They are assigned post-hoc to synthetic rows and never DP-trained.
"""

from __future__ import annotations

import random
import string
from datetime import date, datetime


# ── ACH trace number ─────────────────────────────────────────────────────────

def generate_trace_number(
    routing: str | None = None,
    sequence: int | None = None,
    rng: random.Random | None = None,
) -> str:
    """
    ACH trace number format: RRRRRRRR SSSSSSS  (15 digits total)
    RRRRRRRR = first 8 digits of ODFI routing number
    SSSSSSS  = 7-digit sequence number assigned by the ODFI

    Example: 021000021000042
    """
    _rng = rng or random
    rt = (routing or "021000021")[:8]
    seq = sequence if sequence is not None else _rng.randint(1, 9_999_999)
    return f"{rt}{seq:07d}"


# ── Fedwire IMAD ──────────────────────────────────────────────────────────────

def generate_imad(
    txn_date: date | None = None,
    sequence: int | None = None,
    rng: random.Random | None = None,
) -> str:
    """
    Fedwire IMAD (Input Message Accountability Data) format:
        YYYYMMDD  BBBBBBBB  SSSSSS
        date      bank-id   sequence

    Example: 20240315B6B7HU3R000042
    """
    _rng = rng or random
    dt = txn_date or date.today()
    date_str = dt.strftime("%Y%m%d")
    bank_id = "".join(_rng.choices(string.ascii_uppercase + string.digits, k=8))
    seq = sequence if sequence is not None else _rng.randint(1, 999_999)
    return f"{date_str}{bank_id}{seq:06d}"


# ── Check identifiers ─────────────────────────────────────────────────────────

def generate_check_number(
    start: int = 1001,
    rng: random.Random | None = None,
) -> str:
    """
    Return a plausible check number (4–6 digits).
    In practice these are sequential per account; this generates a random offset.
    """
    _rng = rng or random
    return str(_rng.randint(start, start + 9999))


def generate_micr_line(
    routing: str | None = None,
    account: str | None = None,
    check_number: str | None = None,
    rng: random.Random | None = None,
) -> str:
    """
    MICR line in human-readable representation.

    Actual MICR uses special E13B font characters; we represent them as:
        |routing|  account  check_number

    Example: |021000021|  1234567890  1042
    """
    _rng = rng or random
    rt = routing or "021000021"
    acct = account or str(_rng.randint(100_000_000, 999_999_999))
    chk = check_number or generate_check_number(rng=_rng)
    return f"|{rt}|  {acct}  {chk}"


# ── Generic helpers ───────────────────────────────────────────────────────────

def generate_authorization_code(rng: random.Random | None = None) -> str:
    """6-character alphanumeric authorization code for card transactions."""
    _rng = rng or random
    return "".join(_rng.choices(string.ascii_uppercase + string.digits, k=6))
