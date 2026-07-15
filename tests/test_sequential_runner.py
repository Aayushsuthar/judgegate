import pytest

from judgegate.data import load_examples
from judgegate.errors import AnalysisError
from judgegate.gate import Verdict
from judgegate.sequential_runner import run_replay
from tests.conftest import grader_rows, make_config, write_labels


def examples_from(tmp_path, rows):
    return load_examples(write_labels(tmp_path / "labels.jsonl", rows))


def labels_of(examples):
    return {e.id: e.judge_label for e in examples}


class TestReplay:
    def test_excellent_judge_certifies_early(self, tmp_path):
        examples = examples_from(tmp_path, grader_rows(400, flip_every=50))
        outcome = run_replay(examples, labels_of(examples), make_config(), batch_size=25)
        assert outcome.verdict is Verdict.TRUSTED
        assert outcome.labels_used < 400
        assert outcome.saved_fraction > 0.2

    def test_bad_judge_is_rejected_early(self, tmp_path):
        examples = examples_from(tmp_path, grader_rows(400, flip_every=2))
        outcome = run_replay(examples, labels_of(examples), make_config(), batch_size=25)
        assert outcome.verdict is Verdict.UNTRUSTED
        assert outcome.labels_used < 400

    def test_borderline_judge_exhausts_labels(self, tmp_path):
        examples = examples_from(tmp_path, grader_rows(60, flip_every=7))
        outcome = run_replay(examples, labels_of(examples), make_config(), batch_size=20)
        if outcome.verdict is Verdict.INCONCLUSIVE:
            assert outcome.labels_used == 60
        assert outcome.snapshots[-1].n_labels == outcome.labels_used

    def test_notes_state_the_assumption(self, tmp_path):
        examples = examples_from(tmp_path, grader_rows(100, flip_every=25))
        outcome = run_replay(examples, labels_of(examples), make_config())
        assert any("chance agreement" in note for note in outcome.notes)

    def test_too_few_labels_raise(self, tmp_path):
        examples = examples_from(tmp_path, grader_rows(10))
        with pytest.raises(AnalysisError, match="at least"):
            run_replay(examples, labels_of(examples), make_config())
