import pytest

from judgegate.config import Config
from judgegate.data import load_examples
from judgegate.judge.client import JudgeClient
from judgegate.probes import format as format_probe
from judgegate.probes import position, stability, verbosity
from tests.conftest import (
    PAIRWISE_CONFIG,
    json_label_transport,
    make_config,
    write_labels,
)


def grader_examples(tmp_path, rows):
    return load_examples(write_labels(tmp_path / "labels.jsonl", rows))


class TestVerbosityOffline:
    def _rows(self, lenient_on_long: bool):
        rows = []
        for i in range(30):
            long_output = i % 2 == 0
            text = ("word " * 60) if long_output else "word"
            human = "fail"
            judge = "pass" if (long_output and lenient_on_long) else "fail"
            rows.append(
                {
                    "id": f"v-{i}",
                    "input": "q",
                    "output": text,
                    "human_label": human,
                    "judge_label": judge,
                }
            )
        return rows

    def test_length_biased_judge_fails(self, tmp_path):
        examples = grader_examples(tmp_path, self._rows(lenient_on_long=True))
        labels = {e.id: e.judge_label for e in examples}
        result = verbosity.run(examples, make_config(), labels, "grader")
        assert result.applicable
        assert result.passed is False
        assert result.effect is not None
        assert result.effect > 0.5

    def test_unbiased_judge_passes(self, tmp_path):
        examples = grader_examples(tmp_path, self._rows(lenient_on_long=False))
        labels = {e.id: e.judge_label for e in examples}
        result = verbosity.run(examples, make_config(), labels, "grader")
        assert result.applicable
        assert result.passed is True

    def test_skips_for_pairwise_and_small_sets(self, tmp_path):
        examples = grader_examples(tmp_path, self._rows(True)[:10])
        labels = {e.id: e.judge_label for e in examples}
        assert not verbosity.run(examples, make_config(), labels, "pairwise").applicable
        assert not verbosity.run(examples, make_config(), labels, "grader").applicable


class TestStabilityLive:
    def test_flaky_judge_fails(self, tmp_path):
        rows = [
            {"id": f"s-{i}", "input": f"q{i}", "output": f"a{i}", "human_label": "pass"}
            for i in range(10)
        ]
        examples = grader_examples(tmp_path, rows)
        base = {e.id: "pass" for e in examples}

        def decide_by_marker(prompt: str) -> str:
            for marker in ("q1\n", "q2\n", "q3\n"):
                if marker in prompt or prompt.rstrip().endswith(marker.strip()):
                    return "fail"
            return "pass"

        client = JudgeClient(
            make_config().judge, transport=json_label_transport(decide_by_marker)
        )
        result = stability.run(examples, make_config(), client, base)
        client.close()
        assert result.applicable
        assert result.passed is False
        assert result.effect == pytest.approx(0.3)

    def test_stable_judge_passes(self, tmp_path):
        rows = [
            {"id": f"s-{i}", "input": f"q{i}", "output": f"a{i}", "human_label": "pass"}
            for i in range(10)
        ]
        examples = grader_examples(tmp_path, rows)
        base = {e.id: "pass" for e in examples}
        client = JudgeClient(
            make_config().judge, transport=json_label_transport(lambda _p: "pass")
        )
        result = stability.run(examples, make_config(), client, base)
        client.close()
        assert result.passed is True
        assert result.effect == 0.0

    def test_skips_offline(self, tmp_path):
        rows = [
            {"id": "s-0", "input": "q", "output": "a", "human_label": "pass"}
        ]
        examples = grader_examples(tmp_path, rows)
        result = stability.run(examples, make_config(), None, {"s-0": "pass"})
        assert not result.applicable


class TestFormatLive:
    def _rows(self):
        return [
            {
                "id": f"f-{i}",
                "input": "q",
                "output": f"**bold answer {i}** with `code`",
                "human_label": "pass",
            }
            for i in range(8)
        ]

    def test_typography_grader_fails(self, tmp_path):
        examples = grader_examples(tmp_path, self._rows())
        base = {e.id: "pass" for e in examples}

        def decide(prompt: str) -> str:
            return "pass" if "**" in prompt else "fail"

        client = JudgeClient(make_config().judge, transport=json_label_transport(decide))
        result = format_probe.run(examples, make_config(), client, base)
        client.close()
        assert result.applicable
        assert result.passed is False
        assert result.effect == pytest.approx(1.0)

    def test_content_grader_passes(self, tmp_path):
        examples = grader_examples(tmp_path, self._rows())
        base = {e.id: "pass" for e in examples}
        client = JudgeClient(
            make_config().judge, transport=json_label_transport(lambda _p: "pass")
        )
        result = format_probe.run(examples, make_config(), client, base)
        client.close()
        assert result.passed is True

    def test_skips_without_markdown(self, tmp_path):
        rows = [
            {"id": f"f-{i}", "input": "q", "output": "plain", "human_label": "pass"}
            for i in range(8)
        ]
        examples = grader_examples(tmp_path, rows)
        client = JudgeClient(
            make_config().judge, transport=json_label_transport(lambda _p: "pass")
        )
        result = format_probe.run(examples, make_config(), client, {e.id: "pass" for e in examples})
        client.close()
        assert not result.applicable

    def test_strip_markdown(self):
        assert format_probe.strip_markdown("**bold** and `code`") == "bold and code"
        assert format_probe.strip_markdown("# Title\ntext") == "Title\ntext"


class TestPositionLive:
    def _examples(self, tmp_path):
        rows = [
            {
                "id": f"p-{i}",
                "input": "q",
                "output_a": f"GOOD answer {i}",
                "output_b": f"weak answer {i}",
                "human_label": "a",
            }
            for i in range(10)
        ]
        return grader_examples(tmp_path, rows)

    def test_slot_following_judge_fails(self, tmp_path):
        config = Config.model_validate(PAIRWISE_CONFIG)
        examples = self._examples(tmp_path)
        base = {e.id: "a" for e in examples}
        client = JudgeClient(config.judge, transport=json_label_transport(lambda _p: "a"))
        result = position.run(examples, config, client, base, "pairwise")
        client.close()
        assert result.applicable
        assert result.passed is False
        assert result.effect == pytest.approx(1.0)

    def test_content_following_judge_passes(self, tmp_path):
        config = Config.model_validate(PAIRWISE_CONFIG)
        examples = self._examples(tmp_path)
        base = {e.id: "a" for e in examples}

        def decide(prompt: str) -> str:
            a_section = prompt.split("A: ")[1].split("B: ", maxsplit=1)[0]
            return "a" if "GOOD" in a_section else "b"

        client = JudgeClient(config.judge, transport=json_label_transport(decide))
        result = position.run(examples, config, client, base, "pairwise")
        client.close()
        assert result.passed is True

    def test_skips_for_grader_mode(self, tmp_path):
        config = Config.model_validate(PAIRWISE_CONFIG)
        examples = self._examples(tmp_path)
        result = position.run(examples, config, None, {}, "grader")
        assert not result.applicable
