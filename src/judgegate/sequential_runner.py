from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from judgegate.config import Config
from judgegate.data import LabeledExample
from judgegate.errors import AnalysisError
from judgegate.gate import EXIT_CODES, Verdict
from judgegate.stats.kappa import measure_agreement
from judgegate.stats.sequential import MixtureSPRT, agreement_threshold


@dataclass(frozen=True)
class BatchSnapshot:
    """State of the sequential test after one labeling batch."""

    batch: int
    n_labels: int
    agreement_rate: float
    p_value: float
    decided: bool


@dataclass(frozen=True)
class SequentialOutcome:
    """Where the labeling effort could have stopped."""

    verdict: Verdict
    snapshots: tuple[BatchSnapshot, ...]
    labels_used: int
    labels_available: int
    theta0: float
    chance_agreement: float
    notes: tuple[str, ...]

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.verdict]

    @property
    def saved_fraction(self) -> float:
        if self.labels_available <= 0:
            return 0.0
        return max(0.0, 1.0 - self.labels_used / self.labels_available)


def run_replay(
    examples: Sequence[LabeledExample],
    judge_labels: dict[str, str],
    config: Config,
    batch_size: int = 25,
) -> SequentialOutcome:
    """Replay labeled items through always-valid sequential boundaries.

    Shows where the human labeling effort could have stopped. Each item
    contributes an agreement indicator, tested against the agreement
    rate a threshold-kappa judge would achieve. Chance agreement is
    estimated from the full dataset, which a replay legitimately knows;
    the report states this assumption.
    """
    if batch_size < 1:
        raise AnalysisError(f"batch_size must be at least 1, got {batch_size}")
    if len(examples) < config.gate.min_labels:
        raise AnalysisError(
            f"sequential replay needs at least {config.gate.min_labels} labels, "
            f"found {len(examples)}"
        )

    human = [e.human_label for e in examples]
    judge = [judge_labels[e.id] for e in examples]
    agreement = measure_agreement(human, judge, config.labels.values)
    chance = agreement.chance_agreement
    theta0 = agreement_threshold(chance, config.gate.min_kappa)
    indicators = np.asarray(
        [1.0 if h == j else 0.0 for h, j in zip(human, judge, strict=True)]
    )

    sprt = MixtureSPRT(
        theta0=theta0,
        alpha=config.gate.alpha,
        min_samples=config.gate.min_labels,
        variance=theta0 * (1.0 - theta0),
    )
    snapshots: list[BatchSnapshot] = []
    consumed = 0
    batch_number = 0
    decided = False
    direction = 0
    while consumed < len(examples) and not decided:
        batch_number += 1
        stop = min(consumed + batch_size, len(examples))
        decision = sprt.update(indicators[consumed:stop])
        consumed = stop
        decided = decision.decided
        direction = decision.direction
        snapshots.append(
            BatchSnapshot(
                batch=batch_number,
                n_labels=decision.n,
                agreement_rate=decision.mean,
                p_value=decision.p_value,
                decided=decision.decided,
            )
        )

    if not decided:
        verdict = Verdict.INCONCLUSIVE
    elif direction > 0:
        verdict = Verdict.TRUSTED
    else:
        verdict = Verdict.UNTRUSTED

    notes = [
        (
            f"chance agreement {chance:.1%} was estimated from the full dataset; "
            f"the test stops when observed agreement is provably away from "
            f"{theta0:.1%}, the rate a threshold-kappa judge would achieve"
        )
    ]
    if verdict is Verdict.TRUSTED:
        notes.append(
            "the sequential verdict covers agreement only; run the full verify "
            "gate with probes before promoting a judge"
        )
    if config.gate.weighting != "none":
        notes.append(
            "sequential mode interprets the kappa bar as unweighted agreement; "
            "the verify gate applies the configured weighting"
        )
    return SequentialOutcome(
        verdict=verdict,
        snapshots=tuple(snapshots),
        labels_used=consumed,
        labels_available=len(examples),
        theta0=theta0,
        chance_agreement=chance,
        notes=tuple(notes),
    )
