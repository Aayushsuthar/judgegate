from collections.abc import Sequence

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from judgegate.gate import GateReport, Verdict
from judgegate.probes.base import ProbeResult
from judgegate.report.errorbar import render_error_bar

_STYLES = {
    Verdict.TRUSTED: "bold green",
    Verdict.UNTRUSTED: "bold red",
    Verdict.INCONCLUSIVE: "bold yellow",
}

_SUBTITLES = {
    Verdict.TRUSTED: "this judge agrees with your humans well enough to gate",
    Verdict.UNTRUSTED: "do not let this judge gate anything yet",
    Verdict.INCONCLUSIVE: "not enough labels to certify or reject this judge",
}


def _stats_table(report: GateReport) -> Table:
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim")
    table.add_column()
    weighting = report.agreement.weighting
    metric = "kappa" if weighting == "none" else f"{weighting}-weighted kappa"
    table.add_row("metric", f"{metric} ({report.mode})")
    table.add_row("kappa", f"{report.agreement.kappa:.3f}")
    table.add_row(
        f"{report.interval.confidence:.0%} confidence interval",
        f"[{report.interval.low:.3f}, {report.interval.high:.3f}]",
    )
    table.add_row("trust threshold", f"{report.threshold:.2f}")
    table.add_row("observed agreement", f"{report.agreement.observed_agreement:.1%}")
    table.add_row("chance agreement", f"{report.agreement.chance_agreement:.1%}")
    table.add_row("human labels", str(report.n_labels))
    if report.labels_needed is not None:
        table.add_row("labels needed to decide", f"~{report.labels_needed}")
    return table


def _class_table(report: GateReport) -> Table:
    table = Table(box=None, pad_edge=False)
    table.add_column("label", style="dim")
    table.add_column("human uses", justify="right")
    table.add_column("judge uses", justify="right")
    table.add_column("judge recall", justify="right")
    table.add_column("judge precision", justify="right")
    for row in report.agreement.per_class:
        table.add_row(
            row.label,
            str(row.human_count),
            str(row.judge_count),
            f"{row.recall:.1%}",
            f"{row.precision:.1%}",
        )
    return table


def _probe_table(probes: Sequence[ProbeResult]) -> Table:
    table = Table(box=None, pad_edge=False)
    table.add_column("probe", style="dim")
    table.add_column("verdict")
    table.add_column("effect", justify="right")
    table.add_column("tolerance", justify="right")
    table.add_column("detail")
    for probe in probes:
        style = {"PASS": "green", "FAIL": "red", "SKIPPED": "dim"}[probe.verdict]
        effect = f"{probe.effect:+.1%}" if probe.effect is not None else ""
        tolerance = f"{probe.tolerance:.1%}" if probe.tolerance is not None else ""
        table.add_row(
            probe.name, f"[{style}]{probe.verdict}[/]", effect, tolerance, probe.detail
        )
    return table


def render_probes(probes: Sequence[ProbeResult], console: Console | None = None) -> None:
    """Print probe results on their own, outside a full verify run."""
    console = console or Console()
    failed = [p for p in probes if p.applicable and p.passed is False]
    style = "red" if failed else "green"
    title = "PROBES FAILED" if failed else "PROBES PASSED"
    console.print(
        Panel(
            _probe_table(probes),
            title=f"[bold {style}]{title}[/]",
            border_style=style,
        )
    )


def render_terminal(report: GateReport, console: Console | None = None) -> None:
    """Print the trust report to the terminal."""
    console = console or Console()
    style = _STYLES[report.verdict]
    body: list[RenderableType] = [
        _stats_table(report),
        Text(""),
        Text(render_error_bar(report.interval, report.agreement.kappa, report.threshold)),
        Text(""),
        _class_table(report),
    ]
    if report.probes:
        body += [Text(""), _probe_table(report.probes)]
    if report.notes:
        body.append(Text(""))
        for note in report.notes:
            body.append(Text(f"note: {note}", style="dim"))
    console.print(
        Panel(
            Group(*body),
            title=f"[{style}]{report.verdict.value}[/]",
            subtitle=_SUBTITLES[report.verdict],
            border_style=style.replace("bold ", ""),
        )
    )
