import json
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from judgegate.stats.power import LabelBudget


@dataclass(frozen=True)
class PowerReport:
    """The label-budget answers for one judge and threshold."""

    budget: LabelBudget
    detectable: float | None
    current_labels: int | None
    marginals: tuple[float, ...]


def power_report_to_dict(report: PowerReport) -> dict[str, Any]:
    budget = report.budget
    return {
        "required_labels": budget.required,
        "kappa_true": budget.kappa_true,
        "kappa_threshold": budget.kappa_threshold,
        "alpha": budget.alpha,
        "power": budget.power,
        "max_labels": budget.max_labels,
        "detectable_kappa_now": report.detectable,
        "current_labels": report.current_labels,
        "marginals": list(report.marginals),
        "curve": [{"n": n, "decision_rate": rate} for n, rate in budget.curve],
    }


def render_power_json(report: PowerReport) -> str:
    return json.dumps(power_report_to_dict(report), indent=2)


def render_power_markdown(report: PowerReport) -> str:
    budget = report.budget
    lines = ["## Label budget", ""]
    if budget.required is not None:
        lines += [
            f"Certifying a judge with true kappa **{budget.kappa_true:.2f}** against a "
            f"threshold of **{budget.kappa_threshold:.2f}** needs about "
            f"**{budget.required}** human labels "
            f"(power {budget.power:.0%}, alpha {budget.alpha}).",
            "",
        ]
    else:
        lines += [
            f"A judge with true kappa {budget.kappa_true:.2f} cannot be certified "
            f"against a threshold of {budget.kappa_threshold:.2f} within "
            f"{budget.max_labels} labels.",
            "",
        ]
    if report.current_labels is not None and report.detectable is not None:
        lines += [
            f"With the current **{report.current_labels}** labels, only a judge with "
            f"true kappa of at least **{report.detectable:.2f}** can be certified.",
            "",
        ]
    if budget.curve:
        lines += ["| Labels | Certification rate |", "|---|---|"]
        lines += [f"| {n} | {rate:.0%} |" for n, rate in budget.curve]
        lines.append("")
    return "\n".join(lines)


def render_power_terminal(report: PowerReport, console: Console | None = None) -> None:
    console = console or Console()
    budget = report.budget
    lines = []
    if budget.required is not None:
        lines.append(
            f"certifying a true-kappa-{budget.kappa_true:.2f} judge at threshold "
            f"{budget.kappa_threshold:.2f} needs about {budget.required} human labels"
        )
    else:
        lines.append(
            f"a judge with true kappa {budget.kappa_true:.2f} cannot be certified at "
            f"threshold {budget.kappa_threshold:.2f} within {budget.max_labels} labels"
        )
    lines.append(f"alpha: {budget.alpha}   power: {budget.power:.0%}")
    if report.current_labels is not None and report.detectable is not None:
        lines.append(
            f"your current {report.current_labels} labels can only certify a judge "
            f"with true kappa {report.detectable:.2f} or better"
        )
    console.print(Panel.fit("\n".join(lines), title="label budget"))
    if budget.curve:
        table = Table(box=None, pad_edge=False)
        table.add_column("labels", justify="right")
        table.add_column("certification rate", justify="right")
        for n, rate in budget.curve:
            table.add_row(str(n), f"{rate:.0%}")
        console.print(table)
