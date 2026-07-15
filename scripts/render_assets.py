"""Regenerate the SVG terminal screenshots embedded in the README.

Run from the repository root with the project installed:

    python scripts/render_assets.py

Every image is produced from real judgegate output on the data in
examples/, so the assets never drift from actual behavior.
"""

from pathlib import Path

import numpy as np
from rich.console import Console

from judgegate.config import load_config
from judgegate.data import dataset_mode, load_examples
from judgegate.gate import evaluate_gate
from judgegate.probes import run_probes
from judgegate.report.power_report import PowerReport, render_power_terminal
from judgegate.report.sequential_report import render_sequential_terminal
from judgegate.report.terminal import render_terminal
from judgegate.sequential_runner import run_replay
from judgegate.stats.kappa import measure_agreement
from judgegate.stats.power import detectable_kappa, required_labels

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"


def fresh_console() -> Console:
    return Console(record=True, width=90, force_terminal=True, color_system="truecolor")


def save(console: Console, name: str) -> None:
    console.save_svg(str(ASSETS / name), title="judgegate")
    print(f"wrote assets/{name}")


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    config = load_config(ROOT / "examples" / "judge.yaml")

    good = load_examples(ROOT / "examples" / "labels.jsonl")
    good_labels = {e.id: str(e.judge_label) for e in good}
    console = fresh_console()
    console.print(
        "[bold green]$[/] judgegate verify labels.jsonl --config judge.yaml",
        highlight=False,
    )
    probes = run_probes(good, config, None, good_labels, dataset_mode(good))
    render_terminal(evaluate_gate(good, good_labels, config, probes), console)
    save(console, "verify-trusted.svg")

    noisy = load_examples(ROOT / "examples" / "labels-noisy-judge.jsonl")
    noisy_labels = {e.id: str(e.judge_label) for e in noisy}
    console = fresh_console()
    console.print(
        "[bold green]$[/] judgegate verify cheap-judge-labels.jsonl --config judge.yaml",
        highlight=False,
    )
    noisy_probes = run_probes(noisy, config, None, noisy_labels, dataset_mode(noisy))
    render_terminal(evaluate_gate(noisy, noisy_labels, config, noisy_probes), console)
    save(console, "verify-untrusted.svg")

    console = fresh_console()
    console.print(
        "[bold green]$[/] judgegate power --labels labels.jsonl --config judge.yaml",
        highlight=False,
    )
    counts: dict[str, int] = {}
    for example in good:
        counts[example.human_label] = counts.get(example.human_label, 0) + 1
    marginals = np.asarray(
        [counts.get(label, 0) for label in config.labels.values], dtype=np.float64
    )
    marginals = marginals / marginals.sum()
    agreement = measure_agreement(
        [e.human_label for e in good],
        [good_labels[e.id] for e in good],
        config.labels.values,
    )
    budget = required_labels(
        marginals,
        kappa_true=min(agreement.kappa, 0.99),
        kappa_threshold=config.gate.min_kappa,
        alpha=config.gate.alpha,
        power=config.gate.power,
        seed=config.gate.seed,
    )
    detectable = detectable_kappa(
        len(good), marginals, config.gate.min_kappa, seed=config.gate.seed
    )
    render_power_terminal(
        PowerReport(
            budget=budget,
            detectable=detectable,
            current_labels=len(good),
            marginals=tuple(float(v) for v in marginals),
        ),
        console,
    )
    save(console, "power.svg")

    console = fresh_console()
    console.print(
        "[bold green]$[/] judgegate sequential cheap-judge-labels.jsonl --config judge.yaml",
        highlight=False,
    )
    outcome = run_replay(noisy, noisy_labels, config, batch_size=25)
    render_sequential_terminal(outcome, console)
    save(console, "sequential.svg")


if __name__ == "__main__":
    main()
