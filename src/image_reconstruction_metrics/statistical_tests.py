from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon


@dataclass(frozen=True)
class WilcoxonResult:
    statistic: float
    p_value: float
    alpha: float
    reject_null: bool


@dataclass(frozen=True)
class FriedmanResult:
    statistic: float
    p_value: float
    alpha: float
    reject_null: bool


def wilcoxon_signed_rank_test(
    sample_a,
    sample_b,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> WilcoxonResult:
    """Run Wilcoxon signed-rank test for paired non-parametric comparison."""
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)

    if a.shape != b.shape:
        raise ValueError("sample_a and sample_b must have the same shape")

    if a.ndim != 1:
        raise ValueError("sample_a and sample_b must be 1D sequences")

    test = wilcoxon(a, b, alternative=alternative)

    return WilcoxonResult(
        statistic=float(test.statistic),
        p_value=float(test.pvalue),
        alpha=float(alpha),
        reject_null=bool(test.pvalue < alpha),
    )


def friedman_rank_test(
    *samples,
    alpha: float = 0.05,
) -> FriedmanResult:
    """Run Friedman rank test across >= 3 related samples."""
    if len(samples) < 3:
        raise ValueError("Friedman test requires at least three related samples")

    vectors = [np.asarray(s, dtype=float) for s in samples]

    if any(v.ndim != 1 for v in vectors):
        raise ValueError("All samples must be 1D sequences")

    n = vectors[0].shape[0]
    if any(v.shape[0] != n for v in vectors):
        raise ValueError("All samples must have equal length")

    test = friedmanchisquare(*vectors)

    return FriedmanResult(
        statistic=float(test.statistic),
        p_value=float(test.pvalue),
        alpha=float(alpha),
        reject_null=bool(test.pvalue < alpha),
    )
