"""
ABA routing number generator.

ABA routing numbers are 9 digits. The 9th digit is a check digit computed as:

    checksum = 3*(d1+d4+d7) + 7*(d2+d5+d8) + 1*(d3+d6)
    check_digit = (10 - (checksum % 10)) % 10

The first two digits encode the Federal Reserve district and institution type.
Valid prefix examples: 021 (NY Fed commercial), 061 (Atlanta), 121 (SF), etc.
"""

from __future__ import annotations

import random

# Real-world ABA routing prefixes grouped by Fed district.
# These are public information (published by the ABA).
_VALID_PREFIXES = [
    "011",
    "021",
    "022",
    "026",  # Boston / New York
    "031",
    "036",  # Philadelphia
    "041",
    "042",
    "044",  # Cleveland
    "051",
    "052",
    "053",
    "054",
    "055",  # Richmond
    "061",
    "062",
    "063",
    "064",
    "065",  # Atlanta
    "071",
    "072",
    "073",
    "074",
    "075",  # Chicago
    "081",
    "082",
    "083",
    "084",
    "085",  # St. Louis
    "091",
    "092",
    "096",  # Minneapolis
    "101",
    "102",
    "103",  # Kansas City
    "111",
    "112",
    "113",
    "114",  # Dallas
    "121",
    "122",
    "123",
    "124",
    "125",  # San Francisco
]


def _compute_check_digit(eight_digits: str) -> int:
    d = [int(x) for x in eight_digits]
    total = 3 * (d[0] + d[3] + d[6]) + 7 * (d[1] + d[4] + d[7]) + (d[2] + d[5])
    return (10 - (total % 10)) % 10


def generate_routing_number(rng: random.Random | None = None) -> str:
    """
    Return a syntactically valid 9-digit ABA routing number.

    The check digit is computed correctly, so any bank validation
    library will accept these as structurally valid.  They are NOT
    mapped to real banks.
    """
    _rng = rng or random
    prefix = _rng.choice(_VALID_PREFIXES)
    middle = f"{_rng.randint(0, 99999):05d}"
    eight = prefix + middle
    check = _compute_check_digit(eight)
    return eight + str(check)


def is_valid_routing_number(routing: str) -> bool:
    """Return True if routing passes the ABA check-digit algorithm."""
    if len(routing) != 9 or not routing.isdigit():
        return False
    return _compute_check_digit(routing[:8]) == int(routing[8])
