from collections.abc import Sequence

from judgegate.config import Config
from judgegate.data import DatasetMode, LabeledExample
from judgegate.judge.client import JudgeClient
from judgegate.probes import format as format_probe
from judgegate.probes import position, stability, verbosity
from judgegate.probes.base import ProbeResult

__all__ = ["ProbeResult", "run_probes"]


def run_probes(
    examples: Sequence[LabeledExample],
    config: Config,
    client: JudgeClient | None,
    base_labels: dict[str, str],
    mode: DatasetMode,
) -> list[ProbeResult]:
    """Run the full probe battery and collect the results."""
    return [
        stability.run(examples, config, client, base_labels),
        position.run(examples, config, client, base_labels, mode),
        verbosity.run(examples, config, base_labels, mode),
        format_probe.run(examples, config, client, base_labels),
    ]
