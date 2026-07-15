import contextlib
import sys
from pathlib import Path
from typing import Any

import click
import numpy as np
from rich.console import Console

from judgegate.__about__ import __version__
from judgegate.config import Config, apply_overrides, load_config
from judgegate.data import LabeledExample, dataset_mode, load_examples
from judgegate.errors import DataError, JudgegateError
from judgegate.gate import GateReport, evaluate_gate
from judgegate.judge.cache import ResponseCache
from judgegate.judge.client import JudgeClient
from judgegate.judge.runner import resolve_judge_labels
from judgegate.probes import run_probes
from judgegate.report import render_json, render_markdown, render_terminal
from judgegate.report.power_report import (
    PowerReport,
    render_power_json,
    render_power_markdown,
    render_power_terminal,
)
from judgegate.report.sequential_report import (
    render_sequential_json,
    render_sequential_markdown,
    render_sequential_terminal,
)
from judgegate.report.terminal import render_probes
from judgegate.sequential_runner import run_replay
from judgegate.stats.kappa import measure_agreement
from judgegate.stats.power import detectable_kappa, required_labels

ERROR_EXIT_CODE = 3

_config_option = click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to the judge.yaml configuration.",
)
_format_option = click.option(
    "--format",
    "output_format",
    type=click.Choice(["terminal", "markdown", "json"]),
    default="terminal",
    show_default=True,
    help="Output format.",
)


class _Session:
    """Owns the client and cache lifecycle for one command."""

    def __init__(self, config: Config, examples: list[LabeledExample]) -> None:
        self.config = config
        with_judge = sum(1 for example in examples if example.judge_label is not None)
        if 0 < with_judge < len(examples):
            raise DataError(
                f"{len(examples) - with_judge} example(s) lack a judge_label while "
                "others have one; either precompute all judge labels or none"
            )
        self.offline = with_judge == len(examples)
        self.cache: ResponseCache | None = None
        self.client: JudgeClient | None = None
        if not self.offline:
            if config.cache.enabled:
                self.cache = ResponseCache(Path(config.cache.path))
            self.client = JudgeClient(config.judge, self.cache)

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
        if self.cache is not None:
            self.cache.close()


def _gate_overrides(**values: Any) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _power_inputs_from_labels(
    labels_path: Path, config: Config | None, kappa_true: float | None
) -> tuple[Any, float | None, int, bool]:
    examples = load_examples(labels_path)
    counts: dict[str, int] = {}
    for example in examples:
        counts[example.human_label] = counts.get(example.human_label, 0) + 1
    label_order = list(config.labels.values) if config is not None else sorted(counts)
    marginal_values = np.asarray(
        [counts.get(label, 0) for label in label_order], dtype=np.float64
    )
    if marginal_values.sum() <= 0:
        raise click.UsageError(
            f"none of the human labels {sorted(counts)} are in the configured "
            f"label set {label_order}; check the config's labels.values"
        )
    dropped = sorted(set(counts) - set(label_order))
    if dropped:
        click.echo(
            f"warning: ignoring human labels outside the configured label set: {dropped}",
            err=True,
        )
    marginal_values = marginal_values / marginal_values.sum()
    fully_judge_labeled = all(e.judge_label is not None for e in examples)
    if kappa_true is None and fully_judge_labeled and config is not None:
        agreement = measure_agreement(
            [e.human_label for e in examples],
            [str(e.judge_label) for e in examples],
            config.labels.values,
        )
        kappa_true = max(0.0, min(agreement.kappa, 0.99))
    return marginal_values, kappa_true, len(examples), fully_judge_labeled


def _emit(content: str, output: Path | None) -> None:
    if output is None:
        click.echo(content)
    else:
        try:
            output.write_text(content, encoding="utf-8", newline="\n")
        except OSError as exc:
            raise JudgegateError(f"could not write report to {output}: {exc}") from exc


def _render_gate_report(report: GateReport, output_format: str, output: Path | None) -> None:
    if output_format == "terminal":
        if output is None:
            render_terminal(report)
        else:
            try:
                with output.open("w", encoding="utf-8") as handle:
                    render_terminal(report, Console(file=handle, no_color=True, width=100))
            except OSError as exc:
                raise JudgegateError(f"could not write report to {output}: {exc}") from exc
    elif output_format == "markdown":
        _emit(render_markdown(report), output)
    else:
        _emit(render_json(report), output)
    if output is not None:
        click.echo(
            f"verdict: {report.verdict.value} "
            f"(kappa {report.agreement.kappa:.3f}, "
            f"CI [{report.interval.low:.3f}, {report.interval.high:.3f}]); "
            f"report written to {output}"
        )


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="judgegate")
def cli() -> None:
    """A CI trust gate for LLM judges.

    Exit codes for verify and sequential: 0 TRUSTED, 1 UNTRUSTED,
    2 INCONCLUSIVE, 3 operational error.
    """


@cli.command()
@click.argument("labels", type=click.Path(path_type=Path))
@_config_option
@_format_option
@click.option(
    "--min-kappa",
    type=click.FloatRange(-1.0, 1.0, min_open=True, max_open=True),
    default=None,
    help="Trust threshold, overriding the config file.",
)
@click.option(
    "--alpha",
    type=click.FloatRange(0.0, 1.0, min_open=True, max_open=True),
    default=None,
    help="Significance level, overriding the config file.",
)
@click.option("--seed", type=click.IntRange(min=0), default=None)
@click.option("--resamples", type=click.IntRange(min=100), default=None)
@click.option("--no-probes", is_flag=True, help="Skip the probe battery.")
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Write the report to a file instead of stdout.",
)
def verify(
    labels: Path,
    config_path: Path,
    output_format: str,
    min_kappa: float | None,
    alpha: float | None,
    seed: int | None,
    resamples: int | None,
    no_probes: bool,
    output: Path | None,
) -> None:
    """Measure the judge against human labels and decide if it can be trusted."""
    config = apply_overrides(
        load_config(config_path),
        gate=_gate_overrides(
            min_kappa=min_kappa, alpha=alpha, seed=seed, resamples=resamples
        ),
    )
    examples = load_examples(labels)
    mode = dataset_mode(examples)
    session = _Session(config, examples)
    try:
        judge_labels = resolve_judge_labels(examples, config, session.client)
        probe_results = (
            []
            if no_probes
            else run_probes(examples, config, session.client, judge_labels, mode)
        )
        report = evaluate_gate(examples, judge_labels, config, probe_results)
    finally:
        session.close()
    _render_gate_report(report, output_format, output)
    sys.exit(report.exit_code)


@cli.command()
@click.argument("labels", type=click.Path(path_type=Path))
@_config_option
def probe(labels: Path, config_path: Path) -> None:
    """Run only the bias and stability probes against the judge."""
    config = load_config(config_path)
    examples = load_examples(labels)
    mode = dataset_mode(examples)
    session = _Session(config, examples)
    try:
        judge_labels = resolve_judge_labels(examples, config, session.client)
        results = run_probes(examples, config, session.client, judge_labels, mode)
    finally:
        session.close()
    render_probes(results)
    failed = any(p.applicable and p.passed is False for p in results)
    sys.exit(1 if failed else 0)


@cli.command()
@click.option(
    "--labels",
    "labels_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Labels file used to estimate marginals and the current kappa.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=None,
    help="judge.yaml providing the label set and threshold.",
)
@click.option(
    "--marginals",
    default=None,
    help="Comma separated class proportions when no labels file is given.",
)
@click.option(
    "--kappa",
    "kappa_true",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help="Assumed true kappa of the judge.",
)
@click.option(
    "--threshold",
    type=click.FloatRange(-1.0, 1.0, min_open=True, max_open=True),
    default=None,
    help="Trust threshold to certify against.",
)
@click.option(
    "--alpha",
    type=click.FloatRange(0.0, 1.0, min_open=True, max_open=True),
    default=None,
)
@click.option(
    "--power",
    "power_target",
    type=click.FloatRange(0.0, 1.0, min_open=True, max_open=True),
    default=None,
)
@click.option("--max-labels", type=click.IntRange(min=100), default=20_000, show_default=True)
@click.option("--seed", type=click.IntRange(min=0), default=None)
@_format_option
def power(
    labels_path: Path | None,
    config_path: Path | None,
    marginals: str | None,
    kappa_true: float | None,
    threshold: float | None,
    alpha: float | None,
    power_target: float | None,
    max_labels: int,
    seed: int | None,
    output_format: str,
) -> None:
    """Answer the label budget question: how many human labels do you need?"""
    config = load_config(config_path) if config_path is not None else None
    gate_threshold = threshold if threshold is not None else (
        config.gate.min_kappa if config is not None else 0.6
    )
    gate_alpha = alpha if alpha is not None else (
        config.gate.alpha if config is not None else 0.05
    )
    gate_power = power_target if power_target is not None else (
        config.gate.power if config is not None else 0.8
    )
    gate_seed = seed if seed is not None else (
        config.gate.seed if config is not None else None
    )

    current_labels: int | None = None
    fully_judge_labeled = False
    if labels_path is not None:
        marginal_values, kappa_true, current_labels, fully_judge_labeled = (
            _power_inputs_from_labels(labels_path, config, kappa_true)
        )
    elif marginals is not None:
        try:
            marginal_values = np.asarray(
                [float(part) for part in marginals.split(",") if part.strip()],
                dtype=np.float64,
            )
        except ValueError as exc:
            raise click.UsageError(f"cannot parse --marginals {marginals!r}") from exc
        if not np.all(np.isfinite(marginal_values)):
            raise click.UsageError("--marginals must contain finite numbers")
    else:
        raise click.UsageError("provide either --labels or --marginals")

    if kappa_true is None:
        if fully_judge_labeled and config_path is None:
            raise click.UsageError(
                "estimating kappa from the file's judge labels needs --config "
                "for the label set; either pass --config or give --kappa directly"
            )
        raise click.UsageError(
            "provide --kappa, or a labels file with judge_label fields plus "
            "--config to estimate it"
        )

    budget = required_labels(
        marginal_values,
        kappa_true=kappa_true,
        kappa_threshold=gate_threshold,
        alpha=gate_alpha,
        power=gate_power,
        max_labels=max_labels,
        seed=gate_seed,
    )
    detectable = (
        detectable_kappa(
            current_labels,
            marginal_values,
            gate_threshold,
            gate_alpha,
            gate_power,
            seed=gate_seed,
        )
        if current_labels is not None and current_labels >= 10
        else None
    )
    report = PowerReport(
        budget=budget,
        detectable=detectable,
        current_labels=current_labels,
        marginals=tuple(float(v) for v in marginal_values),
    )
    if output_format == "terminal":
        render_power_terminal(report)
    elif output_format == "markdown":
        click.echo(render_power_markdown(report))
    else:
        click.echo(render_power_json(report))


@cli.command()
@click.argument("labels", type=click.Path(path_type=Path))
@_config_option
@_format_option
@click.option("--batch-size", type=click.IntRange(min=1), default=25, show_default=True)
def sequential(
    labels: Path, config_path: Path, output_format: str, batch_size: int
) -> None:
    """Replay labels through sequential boundaries to find the early stop point."""
    config = load_config(config_path)
    examples = load_examples(labels)
    session = _Session(config, examples)
    try:
        judge_labels = resolve_judge_labels(examples, config, session.client)
    finally:
        session.close()
    outcome = run_replay(examples, judge_labels, config, batch_size)
    if output_format == "terminal":
        render_sequential_terminal(outcome)
    elif output_format == "markdown":
        click.echo(render_sequential_markdown(outcome))
    else:
        click.echo(render_sequential_json(outcome))
    sys.exit(outcome.exit_code)


@cli.command()
@click.argument("labels", type=click.Path(path_type=Path))
def validate(labels: Path) -> None:
    """Check that a labels file parses and summarize its contents."""
    examples = load_examples(labels)
    mode = dataset_mode(examples)
    label_counts: dict[str, int] = {}
    for example in examples:
        label_counts[example.human_label] = label_counts.get(example.human_label, 0) + 1
    with_judge = sum(1 for e in examples if e.judge_label is not None)
    lengths = sorted(e.output_length for e in examples)

    click.echo(f"{labels}: OK")
    click.echo(f"  examples: {len(examples)}")
    click.echo(f"  mode: {mode}")
    click.echo(
        "  human labels: "
        + ", ".join(f"{label} x{count}" for label, count in sorted(label_counts.items()))
    )
    click.echo(f"  with judge_label: {with_judge} of {len(examples)}")
    click.echo(f"  output length: {lengths[0]} to {lengths[-1]} characters")
    if with_judge == len(examples):
        click.echo("  verify can run fully offline with this file")


def main() -> None:
    """Console entry point with distinct exit codes for errors."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            with contextlib.suppress(Exception):
                reconfigure(encoding="utf-8", errors="replace")
    try:
        result = cli.main(standalone_mode=False)
    except SystemExit:
        raise
    except click.ClickException as exc:
        exc.show()
        sys.exit(ERROR_EXIT_CODE)
    except click.exceptions.Abort:
        click.echo("aborted", err=True)
        sys.exit(ERROR_EXIT_CODE)
    except JudgegateError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(ERROR_EXIT_CODE)
    except Exception as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(ERROR_EXIT_CODE)
    sys.exit(result if isinstance(result, int) else 0)
