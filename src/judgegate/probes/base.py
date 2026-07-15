from dataclasses import dataclass

from judgegate.stats.intervals import Interval


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of one bias or stability probe.

    A probe fails only when its confidence interval proves the effect
    exceeds the tolerance, mirroring the discipline of the kappa gate.
    A point estimate above tolerance that is not statistically confirmed
    surfaces as WARN, which never fails the gate but is always reported.
    """

    name: str
    applicable: bool
    passed: bool | None
    effect: float | None
    effect_label: str
    tolerance: float | None
    interval: Interval | None
    detail: str

    @property
    def verdict(self) -> str:
        if not self.applicable:
            return "SKIPPED"
        if self.passed is False:
            return "FAIL"
        if (
            self.effect is not None
            and self.tolerance is not None
            and abs(self.effect) > self.tolerance
        ):
            return "WARN"
        return "PASS"


def skipped(name: str, effect_label: str, reason: str) -> ProbeResult:
    return ProbeResult(
        name=name,
        applicable=False,
        passed=None,
        effect=None,
        effect_label=effect_label,
        tolerance=None,
        interval=None,
        detail=reason,
    )


def proven_beyond(interval: Interval, tolerance: float) -> bool:
    """True when the interval proves the effect magnitude exceeds tolerance."""
    return interval.low > tolerance or interval.high < -tolerance
