"""Utility evaluation — TSTR (train-synthetic, test-real) AUC comparison."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from mirrorbank.instruments.base import InstrumentSchema

_NUMERIC_DTYPES = (
    pl.Float32,
    pl.Float64,
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt32,
    pl.UInt64,
    pl.Boolean,
)


@dataclass
class UtilityReport:
    auc_tstr: float  # classifier trained on SYNTH, evaluated on held-out REAL
    auc_trtr: float  # trained on REAL, evaluated on held-out REAL (baseline)
    ratio: float  # auc_tstr / auc_trtr (0 if trtr == 0)
    target_column: str
    pass_utility: bool  # ratio >= ratio_target


def run_utility(
    real: pl.DataFrame,
    synth: pl.DataFrame,
    schema: InstrumentSchema | None = None,
    *,
    target_column: str | None = None,
    ratio_target: float = 0.90,
    seed: int = 0,
    min_minority: int = 5,
) -> UtilityReport | None:
    """Train-synthetic-test-real utility check.

    Trains a RandomForest classifier on synthetic data and on real data
    (baseline), then evaluates both on a held-out split of the real data.
    Returns ``None`` when there is no usable fraud label, the label is
    missing from either frame, or there are too few examples of either
    class to train/evaluate a classifier.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    target = target_column or (schema.fraud_label if schema else None)
    if not target:
        return None
    if target not in real.columns or target not in synth.columns:
        return None

    feature_cols = _numeric_feature_columns(real, synth, target, schema)
    if not feature_cols:
        return None

    real_pd = real.select([*feature_cols, target]).to_pandas()
    synth_pd = synth.select([*feature_cols, target]).to_pandas()

    y_real = real_pd[target].astype(int)
    if y_real.nunique() < 2:
        return None

    value_counts = y_real.value_counts()
    if value_counts.min() < min_minority:
        return None

    X_real = real_pd[feature_cols].astype(float)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_real,
            y_real,
            test_size=0.3,
            random_state=seed,
            stratify=y_real,
        )
    except ValueError:
        return None

    if y_test.nunique() < 2:
        return None

    X_synth = synth_pd[feature_cols].astype(float)
    y_synth = synth_pd[target].astype(int)
    if y_synth.nunique() < 2:
        return None

    clf_trtr = RandomForestClassifier(n_estimators=100, random_state=seed)
    clf_trtr.fit(X_train, y_train)
    auc_trtr = roc_auc_score(y_test, clf_trtr.predict_proba(X_test)[:, 1])

    clf_tstr = RandomForestClassifier(n_estimators=100, random_state=seed)
    clf_tstr.fit(X_synth, y_synth)
    auc_tstr = roc_auc_score(y_test, clf_tstr.predict_proba(X_test)[:, 1])

    ratio = (auc_tstr / auc_trtr) if auc_trtr != 0 else 0.0

    return UtilityReport(
        auc_tstr=float(auc_tstr),
        auc_trtr=float(auc_trtr),
        ratio=float(ratio),
        target_column=target,
        pass_utility=ratio >= ratio_target,
    )


def _numeric_feature_columns(
    real: pl.DataFrame,
    synth: pl.DataFrame,
    target: str,
    schema: InstrumentSchema | None,
) -> list[str]:
    candidates = (
        schema.training_columns()
        if schema
        else [c for c in real.columns if c != target]
    )
    return [
        c
        for c in candidates
        if c != target
        and c in real.columns
        and c in synth.columns
        and real[c].dtype in _NUMERIC_DTYPES
        and synth[c].dtype in _NUMERIC_DTYPES
    ]
