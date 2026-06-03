"""Base class for all payment instrument schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class ColumnKind(str, Enum):
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    FREE_TEXT = "free_text"
    # Excluded from DP training; regenerated post-hoc from reference/ module
    IDENTIFIER = "identifier"
    # Has rigid check-digit or format rules (routing numbers, SWIFT, IMAD)
    REFERENCE = "reference"


@dataclass
class ColumnSpec:
    name: str
    kind: ColumnKind
    is_pii: bool = False
    nullable: bool = False
    description: str = ""
    # For REFERENCE columns: callable that produces a valid fake value
    reference_generator: Callable[[], str] | None = None


class InstrumentSchema:
    """
    Base class for every payment instrument.

    Subclasses declare `name`, `display_name`, `fraud_label`, and `columns`.
    The pipeline (profiler, generator, gauntlet) reads these attributes to
    route data correctly — no instrument-specific logic leaks into generic code.
    """

    name: str = ""
    display_name: str = ""
    fraud_label: str | None = None  # column that flags fraudulent/disputed txns
    columns: list[ColumnSpec] = []

    # ── Column routing ─────────────────────────────────────────────────────────

    def training_columns(self) -> list[str]:
        """Structured columns that go to the DP generator (Track A)."""
        return [
            c.name
            for c in self.columns
            if c.kind in (ColumnKind.CONTINUOUS, ColumnKind.CATEGORICAL, ColumnKind.DATETIME)
            and not c.is_pii
        ]

    def text_columns(self) -> list[str]:
        """Free-text columns handled by the vocab sampler / LLM narrator (Track B)."""
        return [c.name for c in self.columns if c.kind == ColumnKind.FREE_TEXT]

    def reference_columns(self) -> list[ColumnSpec]:
        """Columns generated post-hoc from reference/ module, never DP-trained."""
        return [c for c in self.columns if c.kind == ColumnKind.REFERENCE]

    def identifier_columns(self) -> list[str]:
        """High-cardinality identifiers excluded from training; regenerated post-hoc."""
        return [c.name for c in self.columns if c.kind == ColumnKind.IDENTIFIER]

    def pii_columns(self) -> list[str]:
        """Columns flagged as PII — excluded from training."""
        return [c.name for c in self.columns if c.is_pii]

    # ── Instrument-specific validation ─────────────────────────────────────────

    def validate(self, df) -> list[str]:  # df: polars.DataFrame
        """
        Instrument-specific business-rule checks run after generation.
        Returns a list of error strings; empty list = all checks passed.

        Override in each subclass to add instrument-aware validation.
        """
        return []

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} columns={len(self.columns)}>"
