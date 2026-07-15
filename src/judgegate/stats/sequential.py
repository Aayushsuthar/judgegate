import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

_EPS = 1e-12
_MIN_VARIANCE = 1e-12


def agreement_threshold(chance_agreement: float, kappa_threshold: float) -> float:
    """Observed-agreement rate equivalent to a kappa threshold.

    A judge with kappa exactly at the threshold agrees with humans at
    this rate, given the current chance agreement between the two label
    distributions. Sequential mode tests the per-item agreement
    indicators against this rate.
    """
    if not 0.0 <= chance_agreement < 1.0:
        raise ValueError(f"chance agreement must be in [0, 1), got {chance_agreement}")
    return chance_agreement + kappa_threshold * (1.0 - chance_agreement)


@dataclass(frozen=True)
class SequentialDecision:
    """Snapshot of the sequential test after an update."""

    n: int
    mean: float
    p_value: float
    decided: bool
    direction: int

    @property
    def keeps_collecting(self) -> bool:
        return not self.decided


class MixtureSPRT:
    """Mixture sequential probability ratio test for a mean.

    Tests the null hypothesis that the mean of the observed values
    equals ``theta0`` against a normal mixture of alternatives with
    prior scale ``tau``. The p-value is designed to be inspected after
    every batch: the running minimum of the inverse likelihood ratio
    keeps the false positive rate near alpha under optional stopping.

    With ``variance`` given, the information scale is known under the
    null hypothesis (the Bernoulli case: agreement indicators against a
    known rate have variance ``theta0 * (1 - theta0)``), which keeps the
    test monotone in the evidence: perfect agreement is the strongest
    possible signal, not a degenerate one. Without it, the variance is
    re-estimated from the data at every look and, when ``tau`` is not
    given, it is chosen from the same data, so the guarantee is
    approximate; batches with no spread then contribute no evidence,
    because a sample variance of zero carries no information scale.
    """

    def __init__(
        self,
        theta0: float = 0.0,
        alpha: float = 0.05,
        tau: float | None = None,
        min_samples: int = 10,
        variance: float | None = None,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if tau is not None and tau <= 0.0:
            raise ValueError(f"tau must be positive, got {tau}")
        if min_samples < 2:
            raise ValueError(f"min_samples must be at least 2, got {min_samples}")
        if variance is not None and variance <= 0.0:
            raise ValueError(f"variance must be positive, got {variance}")
        self.theta0 = theta0
        self.alpha = alpha
        self.min_samples = min_samples
        self._tau = tau
        self._variance = variance
        self._values: list[float] = []
        self._p_value = 1.0

    @property
    def n(self) -> int:
        return len(self._values)

    @property
    def p_value(self) -> float:
        return self._p_value

    @property
    def tau(self) -> float | None:
        return self._tau

    def update(self, new_values: npt.ArrayLike) -> SequentialDecision:
        """Fold a batch of observations into the test and return the state."""
        batch = np.asarray(new_values, dtype=np.float64).ravel()
        if batch.size == 0:
            return self._decision()
        if not np.all(np.isfinite(batch)):
            raise ValueError("batch contains non-finite values")
        self._values.extend(float(v) for v in batch)

        n = len(self._values)
        if n < self.min_samples:
            return self._decision()

        arr = np.asarray(self._values, dtype=np.float64)
        if self._variance is not None:
            variance = max(self._variance, _MIN_VARIANCE)
        else:
            if float(arr.max() - arr.min()) < _EPS:
                return self._decision()
            variance = max(float(arr.var(ddof=1)), _MIN_VARIANCE)
        mean = float(arr.mean())
        if self._tau is None:
            self._tau = max(math.sqrt(variance), 1e-6)
        tau2 = self._tau * self._tau

        denom = variance + n * tau2
        log_lambda = 0.5 * math.log(variance / denom)
        log_lambda += (n * n * tau2 * (mean - self.theta0) ** 2) / (2.0 * variance * denom)
        p_now = math.exp(-log_lambda) if log_lambda < 700.0 else 0.0
        self._p_value = min(self._p_value, 1.0, p_now)
        return self._decision()

    def _decision(self) -> SequentialDecision:
        n = len(self._values)
        mean = float(np.mean(self._values)) if n else 0.0
        decided = n >= self.min_samples and self._p_value <= self.alpha
        direction = 0
        if decided:
            direction = 1 if mean > self.theta0 else -1
        return SequentialDecision(
            n=n,
            mean=mean,
            p_value=self._p_value,
            decided=decided,
            direction=direction,
        )
