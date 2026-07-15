from collections.abc import Callable
from dataclasses import dataclass
from math import ceil

import numpy as np
import numpy.typing as npt

from judgegate.errors import AnalysisError
from judgegate.stats.bootstrap import kappa_interval
from judgegate.stats.intervals import z_quantile
from judgegate.stats.kappa import (
    Weighting,
    kappa_from_counts_batch,
    kappa_standard_error_batch,
)

FloatArray = npt.NDArray[np.float64]

_EPS = 1e-9


@dataclass(frozen=True)
class LabelBudget:
    """How many human labels a trustworthy verdict requires."""

    required: int | None
    kappa_true: float
    kappa_threshold: float
    alpha: float
    power: float
    max_labels: int
    curve: tuple[tuple[int, float], ...]


def joint_distribution(marginals: npt.ArrayLike, kappa: float) -> FloatArray:
    """Judge-human joint distribution with the given kappa and marginals.

    A mixture of perfect agreement and independence: the mixture weight
    equals kappa exactly, both marginals equal the supplied ones, and the
    construction is valid for kappa in [0, 1]. The unweighted and
    weighted kappa of this joint coincide, because its expected matrix
    under independence matches the mixture's off-diagonal structure.
    """
    m = np.asarray(marginals, dtype=np.float64)
    if m.ndim != 1 or m.size < 2:
        raise AnalysisError("marginals must be a vector of at least two class proportions")
    if not np.all(np.isfinite(m)):
        raise AnalysisError("marginals contain non-finite values")
    if np.any(m < 0.0) or abs(float(m.sum()) - 1.0) > 1e-6:
        raise AnalysisError("marginals must be non-negative and sum to 1")
    if not 0.0 <= kappa <= 1.0:
        raise AnalysisError(f"the joint construction needs kappa in [0, 1], got {kappa}")
    return kappa * np.diag(m) + (1.0 - kappa) * np.outer(m, m)


def simulate_decision_rate(
    marginals: npt.ArrayLike,
    kappa_true: float,
    kappa_threshold: float,
    n_labels: int,
    alpha: float = 0.05,
    sims: int = 1500,
    seed: int | None = None,
) -> float:
    """Fraction of simulated label sets whose kappa lower bound clears the threshold.

    Uses the large-sample Wald decision on unweighted kappa, which is
    fast but mildly optimistic relative to the real bootstrap gate at
    small samples. ``required_labels`` therefore validates its final
    answer against the actual gate; treat this function as the cheap
    screening estimate it is.
    """
    if n_labels < 10:
        raise AnalysisError(f"n_labels must be at least 10, got {n_labels}")
    if sims < 200:
        raise AnalysisError(f"sims must be at least 200, got {sims}")
    if not 0.0 < alpha < 1.0:
        raise AnalysisError(f"alpha must be in (0, 1), got {alpha}")
    joint = joint_distribution(marginals, kappa_true)
    k = joint.shape[0]
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n_labels, joint.flatten(), size=sims).astype(np.float64)
    counts = counts.reshape(sims, k, k)
    kappa_hat = kappa_from_counts_batch(counts)
    se = kappa_standard_error_batch(counts)
    z = z_quantile(1.0 - alpha / 2.0)
    decided = kappa_hat - z * se > kappa_threshold
    return float(np.mean(decided))


def bootstrap_decision_rate(
    joint: FloatArray,
    n_labels: int,
    kappa_threshold: float,
    alpha: float,
    weighting: Weighting = "none",
    sims: int = 200,
    resamples: int = 600,
    seed: int | None = None,
) -> float:
    """Certification rate of the real bootstrap gate on simulated label sets.

    Slower than the Wald screen but it measures the decision procedure
    the verify command actually runs, so label budgets validated with it
    deliver the stated power.
    """
    k = joint.shape[0]
    rng = np.random.default_rng(seed)
    certified = 0
    for index in range(sims):
        counts = rng.multinomial(n_labels, joint.flatten()).reshape(k, k)
        try:
            interval = kappa_interval(
                counts,
                weighting=weighting,
                confidence=1.0 - alpha,
                resamples=resamples,
                seed=None if seed is None else seed + index + 1,
            )
        except AnalysisError:
            continue
        if interval.low > kappa_threshold:
            certified += 1
    return certified / sims


def required_labels(
    marginals: npt.ArrayLike,
    kappa_true: float,
    kappa_threshold: float,
    alpha: float = 0.05,
    power: float = 0.8,
    sims: int = 1500,
    max_labels: int = 20_000,
    seed: int | None = None,
    weighting: Weighting = "none",
) -> LabelBudget:
    """Smallest label count that certifies a judge of the assumed quality.

    Answers: if the judge's true agreement is ``kappa_true``, how many
    human labels are needed so the verify gate passes it with the target
    power? The search screens candidate sizes with the fast Wald
    approximation (unweighted gates) or the real bootstrap gate
    (weighted gates), then validates the final answer against the real
    gate and enlarges it until the measured certification rate meets the
    target. Returns ``required=None`` when the assumed kappa does not
    exceed the threshold (no amount of labeling can certify such a
    judge) or when the budget exceeds ``max_labels``.
    """
    if not 0.0 < power < 1.0:
        raise AnalysisError(f"power must be in (0, 1), got {power}")
    curve: list[tuple[int, float]] = []

    def unachievable() -> LabelBudget:
        return LabelBudget(
            required=None,
            kappa_true=kappa_true,
            kappa_threshold=kappa_threshold,
            alpha=alpha,
            power=power,
            max_labels=max_labels,
            curve=tuple(sorted(set(curve))),
        )

    if kappa_true <= kappa_threshold + _EPS:
        return unachievable()

    joint = joint_distribution(marginals, kappa_true)

    def wald_rate(n: int, offset: int) -> float:
        value = simulate_decision_rate(
            marginals,
            kappa_true,
            kappa_threshold,
            n,
            alpha,
            sims,
            None if seed is None else seed + offset,
        )
        curve.append((n, value))
        return value

    def gate_rate(n: int, offset: int) -> float:
        value = bootstrap_decision_rate(
            joint,
            n,
            kappa_threshold,
            alpha,
            weighting,
            sims=150,
            resamples=500,
            seed=None if seed is None else seed + 1000 * (offset + 1),
        )
        curve.append((n, value))
        return value

    rate: Callable[[int, int], float] = wald_rate if weighting == "none" else gate_rate

    low, high = 10, 20
    offset = 0
    while high <= max_labels:
        if rate(high, offset) >= power:
            break
        low, high = high, high * 2
        offset += 1
    else:
        return unachievable()

    while high - low > max(1, low // 20):
        mid = (low + high) // 2
        offset += 1
        if rate(mid, offset) >= power:
            high = mid
        else:
            low = mid

    candidate = high
    for bump in range(8):
        measured = bootstrap_decision_rate(
            joint,
            candidate,
            kappa_threshold,
            alpha,
            weighting,
            sims=200,
            resamples=600,
            seed=None if seed is None else seed + 5000 + bump,
        )
        curve.append((candidate, measured))
        if measured >= power:
            break
        candidate = ceil(candidate * 1.12)
        if candidate > max_labels:
            return unachievable()
    else:
        return unachievable()

    return LabelBudget(
        required=candidate,
        kappa_true=kappa_true,
        kappa_threshold=kappa_threshold,
        alpha=alpha,
        power=power,
        max_labels=max_labels,
        curve=tuple(sorted(set(curve))),
    )


def detectable_kappa(
    n_labels: int,
    marginals: npt.ArrayLike,
    kappa_threshold: float,
    alpha: float = 0.05,
    power: float = 0.8,
    sims: int = 1500,
    seed: int | None = None,
) -> float | None:
    """Smallest true kappa this label count can reliably certify.

    Answers: with the labels you already have, how good does the judge
    actually have to be before the gate can prove it? Uses the Wald
    screen, so treat the answer as a planning estimate rather than a
    certified bound. Returns None when even a near-perfect judge cannot
    be certified at this sample size.
    """
    low = kappa_threshold
    high = 0.99
    if (
        simulate_decision_rate(marginals, high, kappa_threshold, n_labels, alpha, sims, seed)
        < power
    ):
        return None
    for offset in range(1, 12):
        mid = (low + high) / 2.0
        rate = simulate_decision_rate(
            marginals,
            mid,
            kappa_threshold,
            n_labels,
            alpha,
            sims,
            None if seed is None else seed + offset,
        )
        if rate >= power:
            high = mid
        else:
            low = mid
    return round(high, 3)
