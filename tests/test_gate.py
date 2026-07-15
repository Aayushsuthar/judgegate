import pytest

from judgegate.data import load_examples
from judgegate.errors import AnalysisError
from judgegate.gate import Verdict, evaluate_gate
from judgegate.probes.base import ProbeResult
from judgegate.stats.intervals import Interval
from tests.conftest import grader_rows, make_config, write_labels


def examples_from(tmp_path, rows):
    return load_examples(write_labels(tmp_path / "labels.jsonl", rows))


def labels_of(examples):
    return {e.id: e.judge_label for e in examples}


def failing_probe() -> ProbeResult:
    return ProbeResult(
        name="stability",
        applicable=True,
        passed=False,
        effect=0.4,
        effect_label="flip rate across reruns",
        tolerance=0.1,
        interval=Interval(0.3, 0.5, 0.95),
        detail="4 of 10 items changed verdict",
    )


class TestVerdicts:
    def test_agreeing_judge_is_trusted(self, tmp_path):
        examples = examples_from(tmp_path, grader_rows(120, flip_every=25))
        report = evaluate_gate(examples, labels_of(examples), make_config())
        assert report.verdict is Verdict.TRUSTED
        assert report.exit_code == 0
        assert report.interval.low > 0.6

    def test_random_judge_is_untrusted(self, tmp_path):
        rows = grader_rows(120, flip_every=2)
        examples = examples_from(tmp_path, rows)
        report = evaluate_gate(examples, labels_of(examples), make_config())
        assert report.verdict is Verdict.UNTRUSTED
        assert report.exit_code == 1

    def test_small_sample_is_inconclusive_with_budget(self, tmp_path):
        examples = examples_from(tmp_path, grader_rows(24, flip_every=6))
        report = evaluate_gate(examples, labels_of(examples), make_config())
        assert report.verdict is Verdict.INCONCLUSIVE
        assert report.exit_code == 2

    def test_failed_probe_forces_untrusted_despite_high_kappa(self, tmp_path):
        examples = examples_from(tmp_path, grader_rows(120, flip_every=25))
        report = evaluate_gate(
            examples, labels_of(examples), make_config(), [failing_probe()]
        )
        assert report.verdict is Verdict.UNTRUSTED
        assert any("stability probe failed" in note for note in report.notes)

    def test_below_minimum_labels_cannot_be_trusted(self, tmp_path):
        examples = examples_from(tmp_path, grader_rows(12))
        report = evaluate_gate(examples, labels_of(examples), make_config())
        assert report.verdict is not Verdict.TRUSTED
        assert any("at least" in note for note in report.notes)

    def test_missing_judge_labels_raise(self, tmp_path):
        examples = examples_from(tmp_path, grader_rows(30))
        labels = labels_of(examples)
        labels.pop("ex-000")
        with pytest.raises(AnalysisError, match="no judge label"):
            evaluate_gate(examples, labels, make_config())

    def test_report_carries_probe_results(self, tmp_path):
        examples = examples_from(tmp_path, grader_rows(120, flip_every=25))
        probe = failing_probe()
        report = evaluate_gate(examples, labels_of(examples), make_config(), [probe])
        assert report.probes == (probe,)
        assert report.failed_probes == (probe,)
