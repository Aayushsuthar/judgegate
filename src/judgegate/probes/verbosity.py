from collections.abc import Sequence
from statistics import median

from judgegate.config import Config
from judgegate.data import LabeledExample
from judgegate.probes.base import ProbeResult, proven_beyond, skipped
from judgegate.stats.intervals import proportion_diff_interval

NAME = "verbosity"
EFFECT = "judge bias gap, long minus short outputs"


def run(
    examples: Sequence[LabeledExample],
    config: Config,
    base_labels: dict[str, str],
    mode: str,
) -> ProbeResult:
    """Check whether output length sways the judge relative to humans.

    Fully offline: items are split at the median output length, and the
    judge's deviation from the human verdict is compared between the
    halves in both directions (rating above the human verdict, and
    rating below it). The direction with the larger gap is reported, so
    the result does not depend on which way the label list is written.
    Needs an ordinal label order, which binary labels have by
    construction.
    """
    settings = config.probes.verbosity
    if not settings.enabled:
        return skipped(NAME, EFFECT, "disabled in config")
    if mode != "grader":
        return skipped(NAME, EFFECT, "applies to grader judges only")
    if not config.labels.ordinal and len(config.labels.values) != 2:
        return skipped(
            NAME,
            EFFECT,
            "needs ordinal labels (or binary labels) to define bias direction",
        )
    if len(examples) < 20:
        return skipped(NAME, EFFECT, "needs at least 20 labeled items to split by length")

    rank = {label: i for i, label in enumerate(config.labels.values)}
    lengths = [example.output_length for example in examples]
    cut = median(lengths)
    short = [e for e in examples if e.output_length <= cut]
    long_ = [e for e in examples if e.output_length > cut]
    if len(short) < 5 or len(long_) < 5:
        return skipped(NAME, EFFECT, "output lengths are too uniform to split")

    def deviation_rate(group: Sequence[LabeledExample], upward: bool) -> float:
        if upward:
            hits = sum(1 for e in group if rank[base_labels[e.id]] > rank[e.human_label])
        else:
            hits = sum(1 for e in group if rank[base_labels[e.id]] < rank[e.human_label])
        return hits / len(group)

    candidates = []
    for upward, direction in ((True, "above"), (False, "below")):
        gap = deviation_rate(long_, upward) - deviation_rate(short, upward)
        interval = proportion_diff_interval(
            deviation_rate(long_, upward),
            len(long_),
            deviation_rate(short, upward),
            len(short),
            config.gate.confidence,
        )
        candidates.append((abs(gap), gap, interval, direction))
    _, gap, interval, direction = max(candidates, key=lambda c: c[0])

    passed = not proven_beyond(interval, settings.max_leniency_gap)
    return ProbeResult(
        name=NAME,
        applicable=True,
        passed=passed,
        effect=gap,
        effect_label=EFFECT,
        tolerance=settings.max_leniency_gap,
        interval=interval,
        detail=(
            f"on {len(long_)} long vs {len(short)} short items (split at {cut:.0f} "
            f"characters), the judge rates {direction} the human verdict "
            f"{abs(gap):.1%} more often on long outputs"
        ),
    )
