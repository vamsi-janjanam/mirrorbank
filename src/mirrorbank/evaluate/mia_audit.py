"""Privacy evaluation — membership inference attack (MIA) audit.

This is a v1 black-box heuristic audit implemented from scratch (Shokri et al.
2017-style intuition, simplified): a nearest-neighbor distance attack. The
hypothesis is that synthetic rows generated from "member" real rows tend to sit
closer to those members than to held-out "non-member" real rows. If an attacker
can distinguish members from non-members using only nearest-neighbor distance to
the synthetic dataset, the synthetic data is leaking information about the
training set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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
class MIAReport:
    attack_auc: float  # ~0.5 = good privacy; higher = leakage
    pass_privacy: bool  # attack_auc <= auc_target


def run_mia(
    real: pl.DataFrame,
    synth: pl.DataFrame,
    schema: InstrumentSchema | None = None,
    *,
    holdout_frac: float = 0.5,
    auc_target: float = 0.55,
    seed: int = 0,
) -> MIAReport | None:
    """Nearest-neighbor distance membership inference audit.

    Splits ``real`` into "members" (A) and "non-members" (B) by
    ``holdout_frac``. For every real row in A ∪ B, computes the distance to
    its nearest neighbor in ``synth`` (after standardizing numeric features
    using a scaler fit on ``synth``). The attack score is the negative
    nearest-neighbor distance (closer => more likely a training member).
    ``attack_auc`` is the ROC-AUC of using that score to predict membership.

    Returns ``None`` if there are fewer than 2 usable numeric features or too
    few rows to form both a member and non-member group.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    feature_cols = _numeric_feature_columns(real, synth, schema)
    if len(feature_cols) < 2:
        return None

    real_pd = real.select(feature_cols).fill_null(0).to_pandas().astype(float)
    synth_pd = synth.select(feature_cols).fill_null(0).to_pandas().astype(float)

    n = len(real_pd)
    if n < 4 or len(synth_pd) < 1:
        return None

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    n_members = int(round(n * (1 - holdout_frac)))
    member_idx = indices[:n_members]
    nonmember_idx = indices[n_members:]

    if len(member_idx) == 0 or len(nonmember_idx) == 0:
        return None

    scaler = StandardScaler()
    synth_scaled = scaler.fit_transform(synth_pd.to_numpy())
    real_scaled = scaler.transform(real_pd.to_numpy())

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(synth_scaled)

    eval_idx = np.concatenate([member_idx, nonmember_idx])
    distances, _ = nn.kneighbors(real_scaled[eval_idx])
    nearest_dist = distances[:, 0]

    membership_label = np.concatenate(
        [np.ones(len(member_idx)), np.zeros(len(nonmember_idx))]
    )
    score = -nearest_dist  # closer => higher score => more likely member

    if len(np.unique(membership_label)) < 2:
        return None

    attack_auc = float(roc_auc_score(membership_label, score))

    return MIAReport(
        attack_auc=attack_auc,
        pass_privacy=attack_auc <= auc_target,
    )


def _numeric_feature_columns(
    real: pl.DataFrame,
    synth: pl.DataFrame,
    schema: InstrumentSchema | None,
) -> list[str]:
    candidates = (
        schema.training_columns() if schema else list(real.columns)
    )
    return [
        c
        for c in candidates
        if c in real.columns
        and c in synth.columns
        and real[c].dtype in _NUMERIC_DTYPES
        and synth[c].dtype in _NUMERIC_DTYPES
    ]
