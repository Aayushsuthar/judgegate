import json

from rich.console import Console

from judgegate.data import load_examples
from judgegate.gate import evaluate_gate
from judgegate.probes.base import ProbeResult, skipped
from judgegate.report import MARKER, render_error_bar, render_json, render_markdown
from judgegate.report.jsonout import report_to_dict
from judgegate.report.terminal import render_probes, render_terminal
from judgegate.stats.intervals import Interval
from tests.conftest import grader_rows, make_config, write_labels


def sample_report(tmp_path, flip_every=25):
    examples = load_examples(
        write_labels(tmp_path / "labels.jsonl", grader_rows(120, flip_every=flip_every))
    )
    labels = {e.id: e.judge_label for e in examples}
    probes = [
        skipped("stability", "flip rate", "requires a judge endpoint"),
        ProbeResult(
            name="verbosity",
            applicable=True,
            passed=True,
            effect=0.03,
            effect_label="leniency gap",
            tolerance=0.15,
            interval=Interval(-0.05, 0.11, 0.95),
            detail="balanced across lengths",
        ),
    ]
    return evaluate_gate(examples, labels, make_config(), probes)


class TestErrorBar:
    def test_contains_markers_and_legend(self):
        art = render_error_bar(Interval(0.55, 0.85, 0.95), 0.7, 0.6)
        lines = art.splitlines()
        assert "[" in lines[0]
        assert "]" in lines[0]
        assert "o" in lines[0]
        assert "|" in lines[1]
        assert "threshold" in lines[2]
        assert "95% CI [" in lines[3]

    def test_point_sits_inside_brackets(self):
        art = render_error_bar(Interval(0.2, 0.9, 0.95), 0.5, 0.6)
        bar = art.splitlines()[0]
        assert bar.index("[") < bar.index("o") < bar.index("]")

    def test_ascii_only(self):
        art = render_error_bar(Interval(-0.4, 0.95, 0.99), 0.3, 0.6)
        assert art == art.encode("ascii", errors="replace").decode("ascii")


class TestMarkdown:
    def test_contains_marker_verdict_tables_and_probes(self, tmp_path):
        report = sample_report(tmp_path)
        text = render_markdown(report)
        assert text.startswith(MARKER)
        assert report.verdict.value in text
        assert "| Kappa |" in text
        assert "| verbosity |" in text
        assert "Per-class agreement" in text
        assert "```" in text


class TestJson:
    def test_round_trip_and_stable_keys(self, tmp_path):
        report = sample_report(tmp_path)
        payload = json.loads(render_json(report))
        assert payload["verdict"] == report.verdict.value
        assert payload["exit_code"] == report.exit_code
        assert payload["confusion_matrix"] == report.agreement.matrix.tolist()
        for key in (
            "judgegate_version",
            "kappa",
            "interval",
            "threshold",
            "per_class",
            "probes",
            "notes",
        ):
            assert key in payload

    def test_dict_probe_shape(self, tmp_path):
        payload = report_to_dict(sample_report(tmp_path))
        skipped_probe = payload["probes"][0]
        assert skipped_probe["verdict"] == "SKIPPED"
        assert skipped_probe["interval"] is None


class TestTerminal:
    def test_renders_report_without_crashing(self, tmp_path):
        report = sample_report(tmp_path)
        console = Console(record=True, width=100, no_color=True)
        render_terminal(report, console)
        text = console.export_text()
        assert report.verdict.value in text
        assert "kappa" in text

    def test_renders_probes_standalone(self):
        console = Console(record=True, width=100, no_color=True)
        render_probes(
            [skipped("format", "flip rate", "no markdown items")], console
        )
        assert "PROBES PASSED" in console.export_text()
