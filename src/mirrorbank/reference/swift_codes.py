"""
SWIFT / BIC code generator.

SWIFT BIC format (ISO 9362):
    BBBB CC LL [DDD]
    BBBB = 4-char bank code (letters)
    CC   = 2-char country code (ISO 3166-1 alpha-2)
    LL   = 2-char location code (letters/digits)
    DDD  = 3-char branch code (optional; 'XXX' means primary office)

Example: CHASUS33XXX = JPMorgan Chase, US, New York City, primary
"""

from __future__ import annotations

import random
import string

# ISO 3166-1 alpha-2 country codes for countries with significant banking activity
_COUNTRY_CODES = [
    "US", "GB", "DE", "FR", "JP", "CA", "AU", "CH", "SG", "HK",
    "NL", "SE", "ES", "IT", "MX", "BR", "IN", "CN", "KR", "AE",
    "ZA", "NG", "KE", "MX", "AR", "CL", "CO",
]


def generate_swift_code(
    country: str | None = None,
    rng: random.Random | None = None,
) -> str:
    """
    Return a structurally valid SWIFT BIC (11 chars, primary office variant).

    country: ISO 3166-1 alpha-2 code, e.g. 'US'. Randomly chosen if None.
    """
    _rng = rng or random
    cc = country or _rng.choice(_COUNTRY_CODES)
    bank_code = "".join(_rng.choices(string.ascii_uppercase, k=4))
    location = "".join(_rng.choices(string.ascii_uppercase + string.digits, k=2))
    return f"{bank_code}{cc}{location}XXX"


def is_valid_swift_code(bic: str) -> bool:
    """Return True if bic matches the SWIFT BIC format (8 or 11 chars)."""
    bic = bic.strip().upper()
    if len(bic) not in (8, 11):
        return False
    # Bank code: 4 letters
    if not bic[:4].isalpha():
        return False
    # Country code: 2 letters
    if not bic[4:6].isalpha():
        return False
    # Location: 2 alphanumeric
    if not bic[6:8].isalnum():
        return False
    # Branch (if 11 chars): 3 alphanumeric
    if len(bic) == 11 and not bic[8:11].isalnum():
        return False
    return True
