import numpy as np
import numpy.typing as npt

from judgegate.errors import AnalysisError
from judgegate.stats.intervals import (
    Interval,
    normal_cdf,
    wilson_proportion_interval,
    z_quantile,
)
from judgegate.stats.kappa import Weighting, kappa_from_counts, kappa_from_counts_batch

_EPS = 1e-12


def _perfect_agreement_interval(
    counts: npt.NDArray[np.int64], confidence: float
) -> Interval:
    """Lower bound for kappa when observed agreement is perfect.

    The items bootstrap is blind at this boundary: every resample of an
    all-diagonal matrix is also all-diagonal. The honest bound comes from
    the binomial uncertainty of the observed agreement rate itself,
    mapped through the kappa formula with the plug-in chance agreement.
    """
    n = int(counts.sum())
    proportions = counts.astype(np.float64) / n
    row = proportions.sum(axis=1)
    col = proportions.sum(axis=0)
    chance = float(np.sum(row * col))
    po_low = wilson_proportion_interval(1.0, n, confidence).low
    if 1.0 - chance < _EPS:
        return Interval(1.0, 1.0, confidence)
    kappa_low = (po_low - chance) / (1.0 - chance)
    return Interval(max(-1.0, min(kappa_low, 1.0)), 1.0, confidence)


def kappa_interval(
    matrix: npt.ArrayLike,
    weighting: Weighting = "none",
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int | None = None,
) -> Interval:
    """Bias-corrected and accelerated bootstrap interval for kappa.

    Items are the resampling unit. Because kappa depends on the data only
    through the confusion matrix, resampling n items with replacement is
    equivalent to drawing multinomial counts over the confusion cells,
    which allows the whole bootstrap to run vectorized. The jackknife for
    the acceleration term is exact and grouped: removing any item from
    the same cell produces the same leave-one-out kappa, so only one
    computation per occupied cell is needed, weighted by the cell count.
    """
    counts = np.asarray(matrix, dtype=np.int64)
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise AnalysisError("expected a square confusion count matrix")
    if resamples < 100:
        raise AnalysisError(f"resamples must be at least 100, got {resamples}")
    n = int(counts.sum())
    if n < 2:
        raise AnalysisError("interval estimation needs at least two labeled items")

    observed = kappa_from_counts(counts, weighting)
    off_diagonal = int(counts.sum() - np.trace(counts))
    if off_diagonal == 0:
        return _perfect_agreement_interval(counts, confidence)
    flat = counts.flatten().astype(np.float64)
    occupied = int(np.count_nonzero(flat))
    if occupied <= 1:
        return Interval(observed, observed, confidence)

    rng = np.random.default_rng(seed)
    shape = counts.shape
    boot_flat = rng.multinomial(n, flat / n, size=resamples).astype(np.float64)
    boot = kappa_from_counts_batch(boot_flat.reshape(resamples, *shape), weighting)

    if float(boot.max() - boot.min()) < _EPS:
        low = float(boot[0])
        return Interval(min(low, observed), max(low, observed), confidence)

    below = float(np.count_nonzero(boot < observed))
    ties = float(np.count_nonzero(boot == observed))
    p0 = (below + 0.5 * ties) / resamples
    p0 = min(max(p0, 1.0 / (resamples + 1.0)), resamples / (resamples + 1.0))
    z0 = z_quantile(p0)

    jack_values = []
    jack_weights = []
    for index in np.flatnonzero(flat):
        reduced = flat.copy()
        reduced[index] -= 1.0
        jack_values.append(
            float(kappa_from_counts_batch(reduced.reshape(1, *shape), weighting)[0])
        )
        jack_weights.append(flat[index])
    values = np.asarray(jack_values)
    weights = np.asarray(jack_weights)
    mean_jack = float(np.sum(weights * values) / n)
    centered = mean_jack - values
    denom = 6.0 * float(np.sum(weights * centered**2)) ** 1.5
    accel = float(np.sum(weights * centered**3)) / denom if denom > _EPS else 0.0

    alpha = 1.0 - confidence
    quantiles = []
    for z_alpha in (z_quantile(alpha / 2.0), z_quantile(1.0 - alpha / 2.0)):
        correction = 1.0 - accel * (z0 + z_alpha)
        if correction <= _EPS:
            quantiles.append(normal_cdf(z_alpha))
        else:
            quantiles.append(normal_cdf(z0 + (z0 + z_alpha) / correction))

    low = float(np.quantile(boot, quantiles[0]))
    high = float(np.quantile(boot, quantiles[1]))
    if low > high:
        low, high = high, low
    return Interval(low, high, confidence)
