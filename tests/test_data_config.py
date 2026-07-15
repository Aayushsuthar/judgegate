import pytest

from judgegate.config import load_config, render_prompt
from judgegate.data import dataset_mode, load_examples, observed_labels
from judgegate.errors import ConfigError, DataError
from tests.conftest import write_labels


class TestLoadExamples:
    def test_parses_grader_rows_with_extras_into_metadata(self, tmp_path):
        path = write_labels(
            tmp_path / "labels.jsonl",
            [
                {
                    "id": "a",
                    "input": "q",
                    "output": "ans",
                    "human_label": "pass",
                    "source": "batch-7",
                }
            ],
        )
        examples = load_examples(path)
        assert examples[0].metadata == {"source": "batch-7"}
        assert examples[0].mode == "grader"
        assert examples[0].judge_label is None

    def test_parses_pairwise_rows(self, tmp_path):
        path = write_labels(
            tmp_path / "labels.jsonl",
            [
                {
                    "id": "a",
                    "input": "q",
                    "output_a": "first",
                    "output_b": "second",
                    "human_label": "a",
                }
            ],
        )
        examples = load_examples(path)
        assert examples[0].mode == "pairwise"

    def test_line_numbers_in_errors(self, tmp_path):
        path = tmp_path / "labels.jsonl"
        path.write_text(
            '{"id": "a", "input": "q", "output": "x", "human_label": "pass"}\nnot json\n',
            encoding="utf-8",
        )
        with pytest.raises(DataError, match=":2:"):
            load_examples(path)

    def test_missing_output_is_rejected(self, tmp_path):
        path = write_labels(
            tmp_path / "labels.jsonl", [{"id": "a", "input": "q", "human_label": "pass"}]
        )
        with pytest.raises(DataError, match="output"):
            load_examples(path)

    def test_mixed_output_modes_on_one_row_rejected(self, tmp_path):
        path = write_labels(
            tmp_path / "labels.jsonl",
            [
                {
                    "id": "a",
                    "output": "x",
                    "output_a": "y",
                    "output_b": "z",
                    "human_label": "pass",
                }
            ],
        )
        with pytest.raises(DataError, match="not both"):
            load_examples(path)

    def test_duplicate_ids_rejected(self, tmp_path):
        rows = [
            {"id": "a", "output": "x", "human_label": "pass"},
            {"id": "a", "output": "y", "human_label": "fail"},
        ]
        with pytest.raises(DataError, match="duplicate"):
            load_examples(write_labels(tmp_path / "labels.jsonl", rows))

    def test_mixed_dataset_modes_rejected(self, tmp_path):
        rows = [
            {"id": "a", "output": "x", "human_label": "pass"},
            {"id": "b", "output_a": "x", "output_b": "y", "human_label": "a"},
        ]
        examples = load_examples(write_labels(tmp_path / "labels.jsonl", rows))
        with pytest.raises(DataError, match="mixes"):
            dataset_mode(examples)

    def test_observed_labels_sorted(self, tmp_path):
        rows = [
            {"id": "a", "output": "x", "human_label": "pass"},
            {"id": "b", "output": "y", "human_label": "fail"},
        ]
        examples = load_examples(write_labels(tmp_path / "labels.jsonl", rows))
        assert observed_labels(examples) == ["fail", "pass"]


class TestConfig:
    def test_loads_valid_yaml(self, tmp_path):
        path = tmp_path / "judge.yaml"
        path.write_text(
            "judge:\n"
            "  model: gpt-test\n"
            "  prompt: 'Grade {output}'\n"
            "labels:\n"
            "  values: [fail, pass]\n",
            encoding="utf-8",
        )
        config = load_config(path)
        assert config.judge.model == "gpt-test"
        assert config.gate.min_kappa == 0.6

    def test_unknown_keys_rejected_with_location(self, tmp_path):
        path = tmp_path / "judge.yaml"
        path.write_text(
            "judge:\n  model: m\n  prompt: p\n  tempratur: 0\n"
            "labels:\n  values: [a, b]\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="tempratur"):
            load_config(path)

    def test_duplicate_labels_rejected(self, tmp_path):
        path = tmp_path / "judge.yaml"
        path.write_text(
            "judge:\n  model: m\n  prompt: p\nlabels:\n  values: [pass, pass]\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="distinct"):
            load_config(path)

    def test_regex_extraction_requires_pattern(self, tmp_path):
        path = tmp_path / "judge.yaml"
        path.write_text(
            "judge:\n  model: m\n  prompt: p\n  extraction:\n    method: regex\n"
            "labels:\n  values: [a, b]\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="pattern"):
            load_config(path)

    def test_missing_file(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "nope.yaml")


class TestRenderPrompt:
    def test_fills_placeholders_and_leaves_other_braces(self):
        template = 'Reply {"label": "..."} for {input} and {output}'
        rendered = render_prompt(template, {"input": "Q", "output": "A"})
        assert rendered == 'Reply {"label": "..."} for Q and A'
