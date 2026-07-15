from judgegate.__about__ import __version__
from judgegate.gate import GateReport, Verdict
from judgegate.report.errorbar import render_error_bar

MARKER = "<!-- judgegate-report -->"

_HEADLINES = {
    Verdict.TRUSTED: ":white_check_mark: **TRUSTED**",
    Verdict.UNTRUSTED: ":no_entry: **UNTRUSTED**",
    Verdict.INCONCLUSIVE: ":warning: **INCONCLUSIVE**",
}

_EXPLANATIONS = {
    Verdict.TRUSTED: (
        "The judge's agreement with human labels clears the threshold with the "
        "entire confidence interval, and no probe failed."
    ),
    Verdict.UNTRUSTED: (
        "This judge should not gate anything yet. The evidence is in the table "
        "and probe results below."
    ),
    Verdict.INCONCLUSIVE: (
        "There are not enough human labels to certify or reject this judge."
    ),
}


def render_markdown(report: GateReport) -> str:
    """Render the trust report as GitHub flavored markdown."""
    weighting = report.agreement.weighting
    metric = "kappa" if weighting == "none" else f"{weighting}-weighted kappa"
    lines = [
        MARKER,
        f"## {_HEADLINES[report.verdict]}",
        "",
        _EXPLANATIONS[report.verdict],
        "",
        "| | |",
        "|---|---|",
        f"| Metric | {metric} ({report.mode}) |",
        f"| Kappa | {report.agreement.kappa:.3f} |",
        (
            f"| {report.interval.confidence:.0%} confidence interval "
            f"| [{report.interval.low:.3f}, {report.interval.high:.3f}] |"
        ),
        f"| Trust threshold | {report.threshold:.2f} |",
        f"| Observed agreement | {report.agreement.observed_agreement:.1%} |",
        f"| Chance agreement | {report.agreement.chance_agreement:.1%} |",
        f"| Human labels | {report.n_labels} |",
    ]
    if report.labels_needed is not None:
        lines.append(f"| Labels needed to decide | ~{report.labels_needed} |")
    lines += [
        "",
        "```",
        render_error_bar(report.interval, report.agreement.kappa, report.threshold),
        "```",
        "",
    ]
    if report.probes:
        lines += [
            "| Probe | Verdict | Effect | Tolerance |",
            "|---|---|---|---|",
        ]
        for probe in report.probes:
            effect = f"{probe.effect:+.1%}" if probe.effect is not None else ""
            tolerance = f"{probe.tolerance:.1%}" if probe.tolerance is not None else ""
            lines.append(f"| {probe.name} | {probe.verdict} | {effect} | {tolerance} |")
        lines.append("")
    lines += [
        "<details>",
        "<summary>Per-class agreement</summary>",
        "",
        "| Label | Human uses | Judge uses | Judge recall | Judge precision |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {row.label} | {row.human_count} | {row.judge_count} "
        f"| {row.recall:.1%} | {row.precision:.1%} |"
        for row in report.agreement.per_class
    ]
    lines.append("")
    if report.notes:
        lines += [f"- {note}" for note in report.notes]
        lines.append("")
    lines += [
        "</details>",
        "",
        f"<sub>judgegate v{__version__}</sub>",
        "",
    ]
    return "\n".join(lines)
