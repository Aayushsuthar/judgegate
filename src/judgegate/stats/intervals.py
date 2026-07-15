from dataclasses import dataclass
from statistics import NormalDist

_NORMAL = NormalDist()


@dataclass(frozen=True)
class Interval:
    """A two-sided confidence interval."""

    low: float
    high: float
    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence < 1.0:
            raise ValueError(f"confidence must be in (0, 1), got {self.confidence}")
        if self.low > self.high:
            raise ValueError(f"interval low {self.low} exceeds high {self.high}")

    @property
    def width(self) -> float:
        return self.high - self.low

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high


def z_quantile(p: float) -> float:
    """Standard normal quantile."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"quantile probability must be in (0, 1), got {p}")
    return _NORMAL.inv_cdf(p)


def normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return _NORMAL.cdf(x)


def wilson_proportion_interval(p_hat: float, total: int, confidence: float = 0.95) -> Interval:
    """Wilson score interval for an observed proportion over ``total`` units.

    Behaves well for small samples and proportions near 0 or 1, which is
    the common regime for probe flip rates.
    """
    if total <= 0:
        raise ValueError("total must be a positive integer")
    if not 0.0 <= p_hat <= 1.0:
        raise ValueError(f"p_hat must be in [0, 1], got {p_hat}")
    z = z_quantile(0.5 + confidence / 2.0)
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p_hat + z2 / (2.0 * total)) / denom
    half = z * ((p_hat * (1.0 - p_hat) / total + z2 / (4.0 * total * total)) ** 0.5) / denom
    low = min(max(0.0, center - half), p_hat)
    high = max(min(1.0, center + half), p_hat)
    return Interval(low, high, confidence)


def proportion_diff_interval(
    p1: float, n1: int, p2: float, n2: int, confidence: float = 0.95
) -> Interval:
    """Normal-approximation interval for the difference of two proportions."""
    if n1 <= 0 or n2 <= 0:
        raise ValueError("both sample sizes must be positive")
    for name, p in (("p1", p1), ("p2", p2)):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {p}")
    z = z_quantile(0.5 + confidence / 2.0)
    diff = p1 - p2
    half = z * ((p1 * (1.0 - p1) / n1 + p2 * (1.0 - p2) / n2) ** 0.5)
    return Interval(diff - half, diff + half, confidence)
