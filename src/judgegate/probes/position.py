from collections.abc import Sequence

from judgegate.config import Config
from judgegate.data import LabeledExample
from judgegate.judge.client import JudgeClient
from judgegate.judge.runner import judge_examples
from judgegate.probes.base import ProbeResult, proven_beyond, skipped
from judgegate.stats.intervals import wilson_proportion_interval

NAME = "position"
EFFECT = "verdict flips when outputs swap position"

_SWAP = {"a": "b", "b": "a"}
_MIN_ITEMS = 10


def run(
    examples: Sequence[LabeledExample],
    config: Config,
    client: JudgeClient | None,
    base_labels: dict[str, str],
    mode: str,
) -> ProbeResult:
    """Swap the two candidate outputs and check the verdict follows the content.

    A position-consistent pairwise judge that picked output_a must pick
    the same content after the swap, which now sits in the b slot. Items
    where the verdict follows the slot instead of the content count as
    flips.
    """
    settings = config.probes.position
    if not settings.enabled:
        return skipped(NAME, EFFECT, "disabled in config")
    if mode != "pairwise":
        return skipped(NAME, EFFECT, "applies to pairwise judges only")
    if client is None:
        return skipped(NAME, EFFECT, "requires a judge endpoint; offline labels only")
    if len(examples) < _MIN_ITEMS:
        return skipped(NAME, EFFECT, f"needs at least {_MIN_ITEMS} items")
    lowered = {label.lower() for label in config.labels.values}
    if not {"a", "b"} <= lowered:
        return skipped(
            NAME, EFFECT, "pairwise position probe needs labels named a and b"
        )

    swapped = judge_examples(examples, config, client, swap_pairwise=True)
    flips = 0
    for example in examples:
        base = base_labels[example.id].lower()
        after = swapped[example.id].lower()
        expected = _SWAP.get(base, base)
        if after != expected:
            flips += 1

    rate = flips / len(examples)
    interval = wilson_proportion_interval(rate, len(examples), config.gate.confidence)
    passed = not proven_beyond(interval, settings.max_flip_rate)
    return ProbeResult(
        name=NAME,
        applicable=True,
        passed=passed,
        effect=rate,
        effect_label=EFFECT,
        tolerance=settings.max_flip_rate,
        interval=interval,
        detail=f"{flips} of {len(examples)} verdicts followed the slot, not the content",
    )
