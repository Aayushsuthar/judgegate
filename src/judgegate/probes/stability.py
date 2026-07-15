from collections.abc import Sequence

from judgegate.config import Config
from judgegate.data import LabeledExample
from judgegate.judge.client import JudgeClient
from judgegate.judge.runner import judge_examples
from judgegate.probes.base import ProbeResult, proven_beyond, skipped
from judgegate.stats.intervals import wilson_proportion_interval

NAME = "stability"
EFFECT = "mean per-rerun flip rate"

_MIN_ITEMS = 10


def run(
    examples: Sequence[LabeledExample],
    config: Config,
    client: JudgeClient | None,
    base_labels: dict[str, str],
) -> ProbeResult:
    """Rerun the judge on identical inputs and measure verdict churn.

    A judge that changes its mind between identical calls adds pure
    noise to every downstream decision. The effect is the mean per-rerun
    flip rate: across all reruns, the fraction of item verdicts that
    differed from the base run. Averaging per rerun keeps the number
    comparable when the runs setting changes.
    """
    settings = config.probes.stability
    if not settings.enabled:
        return skipped(NAME, EFFECT, "disabled in config")
    if client is None:
        return skipped(NAME, EFFECT, "requires a judge endpoint; offline labels only")
    if len(examples) < _MIN_ITEMS:
        return skipped(NAME, EFFECT, f"needs at least {_MIN_ITEMS} items")

    flips = 0
    trials = 0
    for rerun in range(1, settings.runs):
        labels = judge_examples(examples, config, client, salt=f"stability-{rerun}")
        for example in examples:
            trials += 1
            if labels.get(example.id) != base_labels.get(example.id):
                flips += 1

    rate = flips / trials
    interval = wilson_proportion_interval(rate, trials, config.gate.confidence)
    passed = not proven_beyond(interval, settings.max_flip_rate)
    return ProbeResult(
        name=NAME,
        applicable=True,
        passed=passed,
        effect=rate,
        effect_label=EFFECT,
        tolerance=settings.max_flip_rate,
        interval=interval,
        detail=(
            f"{flips} of {trials} rerun verdicts differed from the base run "
            f"across {settings.runs} runs at temperature {config.judge.temperature}"
        ),
    )
