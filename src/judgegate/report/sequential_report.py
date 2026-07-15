import json
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from judgegate.gate import Verdict
from judgegate.sequential_runner import SequentialOutcome

_STYLES = {
    Verdict.TRUSTED: "green",
    Verdict.UNTRUSTED: "red",
    Verdict.INCONCLUSIVE: "yellow",
}


def sequential_to_dict(outcome: SequentialOutcome) -> dict[str, Any]:
    return {
        "verdict": outcome.verdict.value,
        "exit_code": outcome.exit_code,
        "labels_used": outcome.labels_used,
        "labels_available": outcome.labels_available,
        "saved_fraction": outcome.saved_fraction,
        "theta0": outcome.theta0,
        "chance_agreement": outcome.chance_agreement,
        "batches": [
            {
                "batch": snap.batch,
                "n_labels": snap.n_labels,
                "agreement_rate": snap.agreement_rate,
                "p_value": snap.p_value,
                "decided": snap.decided,
            }
            for snap in outcome.snapshots
        ],
        "notes": list(outcome.notes),
    }


def render_sequential_json(outcome: SequentialOutcome) -> str:
    return json.dumps(sequential_to_dict(outcome), indent=2)


def render_sequential_markdown(outcome: SequentialOutcome) -> str:
    lines = [
        f"## Sequential verdict: **{outcome.verdict.value}**",
        "",
        f"Decided after **{outcome.labels_used}** of **{outcome.labels_available}** "
        f"labels ({outcome.saved_fraction:.0%} of the labeling effort saved).",
        "",
        "| Batch | Labels | Agreement | Always-valid p |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {snap.batch} | {snap.n_labels} | {snap.agreement_rate:.1%} "
        f"| {snap.p_value:.4f} |"
        for snap in outcome.snapshots
    ]
    if outcome.notes:
        lines.append("")
        lines += [f"- {note}" for note in outcome.notes]
    lines.append("")
    return "\n".join(lines)


def render_sequential_terminal(
    outcome: SequentialOutcome, console: Console | None = None
) -> None:
    console = console or Console()
    table = Table(box=None, pad_edge=False)
    table.add_column("batch", justify="right")
    table.add_column("labels", justify="right")
    table.add_column("agreement", justify="right")
    table.add_column("always-valid p", justify="right")
    for snap in outcome.snapshots:
        style = "bold" if snap.decided else ""
        table.add_row(
            str(snap.batch),
            str(snap.n_labels),
            f"{snap.agreement_rate:.1%}",
            f"{snap.p_value:.4f}",
            style=style,
        )
    summary = (
        f"stopped after {outcome.labels_used} of {outcome.labels_available} labels "
        f"(saved {outcome.saved_fraction:.0%} of the labeling effort)"
    )
    body = [summary] + [f"note: {note}" for note in outcome.notes]
    console.print(
        Panel.fit(
            "\n".join(body),
            title=f"[bold {_STYLES[outcome.verdict]}]{outcome.verdict.value}[/]",
            border_style=_STYLES[outcome.verdict],
        )
    )
    console.print(table)
