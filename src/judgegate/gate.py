from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from judgegate.config import Config
from judgegate.data import DatasetMode, LabeledExample
from judgegate.errors import AnalysisError
from judgegate.probes.base import ProbeResult
from judgegate.stats.bootstrap import kappa_interval
from judgegate.stats.intervals import Interval
from judgegate.stats.kappa import AgreementResult, measure_agreement
from judgegate.stats.power import required_labels

_EPS = 1e-9


class Verdict(StrEnum):
    TRUSTED = "TRUSTED"
    UNTRUSTED = "UNTRUSTED"
    INCONCLUSIVE = "INCONCLUSIVE"


EXIT_CODES: dict[Verdict, int] = {
    Verdict.TRUSTED: 0,
    Verdict.UNTRUSTED: 1,
    Verdict.INCONCLUSIVE: 2,
}


@dataclass(frozen=True)
class GateReport:
    """Everything the renderers need to explain a trust decision."""

    verdict: Verdict
    mode: DatasetMode
    agreement: AgreementResult
    interval: Interval
    threshold: float
    alpha: float
    n_labels: int
    probes: tuple[ProbeResult, ...]
    labels_needed: int | None
    notes: tuple[str, ...]

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.verdict]

    @property
    def failed_probes(self) -> tuple[ProbeResult, ...]:
        return tuple(p for p in self.probes if p.applicable and p.passed is False)


def _labels_needed(
    agreement: AgreementResult, config: Config
) -> tuple[int | None, str | None]:
    kappa_hat = agreement.kappa
    threshold = config.gate.min_kappa
    if kappa_hat <= threshold + _EPS:
        return None, (
            "the observed agreement sits at or below the bar; more labels "
            "cannot certify this judge unless it is genuinely better than it looks"
        )
    marginals = agreement.matrix.sum(axis=1) / agreement.n_items
    if np.count_nonzero(marginals) < 2:
        return None, "human labels cover a single class, so agreement is unmeasurable"
    budget = required_labels(
        marginals,
        kappa_true=min(kappa_hat, 0.99),
        kappa_threshold=threshold,
        alpha=config.gate.alpha,
        power=config.gate.power,
        sims=800,
        seed=config.gate.seed,
        weighting=config.gate.weighting,
    )
    if budget.required is None:
        return None, (
            "certifying this judge would need more labels than the search budget; "
            "either the judge is barely above the bar or the bar is too high"
        )
    return budget.required, None


def _warn_notes(probes: Sequence[ProbeResult]) -> list[str]:
    notes = []
    for probe in probes:
        if probe.verdict == "WARN" and probe.effect is not None and probe.tolerance is not None:
            notes.append(
                f"{probe.name} probe: the effect estimate {probe.effect:+.1%} exceeds "
                f"the {probe.tolerance:.1%} tolerance but is not statistically "
                "confirmed; more items would settle it"
            )
    return notes


def _failure_notes(failed: Sequence[ProbeResult]) -> list[str]:
    notes = []
    for probe in failed:
        if probe.effect is not None and probe.tolerance is not None:
            notes.append(
                f"{probe.name} probe failed: {probe.effect_label} "
                f"{probe.effect:.1%} exceeds the {probe.tolerance:.1%} tolerance"
            )
        else:
            notes.append(f"{probe.name} probe failed")
    return notes


def evaluate_gate(
    examples: Sequence[LabeledExample],
    judge_labels: dict[str, str],
    config: Config,
    probes: Sequence[ProbeResult] = (),
) -> GateReport:
    """Combine agreement statistics and probe results into a trust verdict.

    TRUSTED requires the whole kappa confidence interval above the
    threshold and no failed probes. UNTRUSTED means the interval sits
    entirely below the threshold or a probe failed outright. Everything
    else is INCONCLUSIVE, reported with the label count that could
    resolve it.
    """
    if not examples:
        raise AnalysisError("no labeled examples to analyze")
    missing = [e.id for e in examples if e.id not in judge_labels]
    if missing:
        raise AnalysisError(
            f"{len(missing)} example(s) have no judge label, first: {missing[0]!r}"
        )

    mode: DatasetMode = examples[0].mode
    human = [e.human_label for e in examples]
    judge = [judge_labels[e.id] for e in examples]
    agreement = measure_agreement(
        human, judge, config.labels.values, config.gate.weighting
    )
    interval = kappa_interval(
        agreement.matrix,
        weighting=config.gate.weighting,
        confidence=config.gate.confidence,
        resamples=config.gate.resamples,
        seed=config.gate.seed,
    )

    notes: list[str] = []
    threshold = config.gate.min_kappa
    failed = [p for p in probes if p.applicable and p.passed is False]
    too_few = agreement.n_items < config.gate.min_labels
    if too_few:
        notes.append(
            f"only {agreement.n_items} labels; the gate requires at least "
            f"{config.gate.min_labels} before certifying a judge"
        )
    occupied_human = int(np.count_nonzero(agreement.matrix.sum(axis=1)))
    unmeasurable = occupied_human < 2 or agreement.chance_agreement >= 1.0 - _EPS
    if unmeasurable:
        notes.append(
            "human labels cover a single class, so chance-corrected agreement "
            "is unmeasurable; label a mix of classes before trusting any judge"
        )
    notes.extend(_warn_notes(probes))

    labels_needed: int | None = None
    if failed:
        verdict = Verdict.UNTRUSTED
        notes.extend(_failure_notes(failed))
    elif not unmeasurable and interval.high < threshold:
        verdict = Verdict.UNTRUSTED
        notes.append(
            "the entire confidence interval sits below the kappa threshold; "
            "this judge disagrees with your humans too often to gate anything"
        )
    elif not unmeasurable and interval.low > threshold and not too_few:
        verdict = Verdict.TRUSTED
    else:
        verdict = Verdict.INCONCLUSIVE
        if not unmeasurable:
            labels_needed, note = _labels_needed(agreement, config)
            if note:
                notes.append(note)

    return GateReport(
        verdict=verdict,
        mode=mode,
        agreement=agreement,
        interval=interval,
        threshold=threshold,
        alpha=config.gate.alpha,
        n_labels=agreement.n_items,
        probes=tuple(probes),
        labels_needed=labels_needed,
        notes=tuple(notes),
    )
