import json

import httpx
import numpy as np
import pytest

from judgegate.config import Config, ExtractionSettings
from judgegate.data import load_examples
from judgegate.errors import ConfigError
from judgegate.gate import Verdict, evaluate_gate
from judgegate.judge.cache import ResponseCache
from judgegate.judge.client import JudgeClient
from judgegate.judge.extraction import extract_label
from judgegate.probes import format as format_probe
from judgegate.probes import stability, verbosity
from judgegate.probes.base import ProbeResult
from judgegate.sequential_runner import run_replay
from judgegate.stats.bootstrap import kappa_interval
from judgegate.stats.intervals import Interval
from judgegate.stats.kappa import kappa_from_counts, kappa_standard_error
from judgegate.stats.sequential import MixtureSPRT
from tests.conftest import (
    BASE_CONFIG,
    grader_rows,
    json_label_transport,
    make_config,
    write_labels,
)

LABELS = ["fail", "pass"]


class TestSingleClassGuard:
    def test_constant_judge_on_single_class_data_is_not_certified(self, tmp_path):
        rows = [
            {"id": f"s-{i}", "input": "q", "output": "a", "human_label": "pass",
             "judge_label": "pass"}
            for i in range(50)
        ]
        examples = load_examples(write_labels(tmp_path / "one.jsonl", rows))
        labels = {e.id: "pass" for e in examples}
        report = evaluate_gate(examples, labels, make_config())
        assert report.verdict is Verdict.INCONCLUSIVE
        assert any("single class" in note for note in report.notes)
        assert report.exit_code == 2


class TestStandardErrorFormula:
    def test_matches_monte_carlo_on_asymmetric_marginals(self):
        joint = np.array([[0.45, 0.30], [0.02, 0.23]])
        rng = np.random.default_rng(5)
        matrix = rng.multinomial(4000, joint.flatten()).reshape(2, 2)
        se = kappa_standard_error(matrix)
        empirical = []
        p = matrix.flatten() / matrix.sum()
        for i in range(400):
            resample = np.random.default_rng(i).multinomial(4000, p).reshape(2, 2)
            empirical.append(kappa_from_counts(resample))
        assert se == pytest.approx(float(np.std(empirical)), rel=0.15)


class TestPerfectAgreementInterval:
    def test_few_perfect_labels_do_not_certify_a_high_bar(self):
        interval = kappa_interval(np.diag([14, 6]), seed=1)
        assert interval.high == pytest.approx(1.0)
        assert interval.low < 0.9

    def test_many_perfect_labels_earn_a_tight_bound(self):
        small = kappa_interval(np.diag([14, 6]), seed=1)
        large = kappa_interval(np.diag([350, 150]), seed=1)
        assert large.low > small.low
        assert large.low > 0.9


class TestSequentialKnownVariance:
    def test_perfect_agreement_certifies_instead_of_starving(self):
        sprt = MixtureSPRT(theta0=0.8, alpha=0.05, min_samples=10, variance=0.16)
        decision = sprt.update(np.ones(120))
        assert decision.decided
        assert decision.direction == 1

    def test_verdict_is_monotone_in_agreement(self):
        def labels_used(disagreements: int) -> int:
            sprt = MixtureSPRT(theta0=0.8, alpha=0.05, min_samples=10, variance=0.16)
            values = np.ones(200)
            values[:disagreements] = 0.0
            rng = np.random.default_rng(3)
            rng.shuffle(values)
            for start in range(0, 200, 25):
                decision = sprt.update(values[start : start + 25])
                if decision.decided:
                    return decision.n
            return 999

        assert labels_used(0) <= labels_used(8)

    def test_replay_certifies_flawless_judge_early(self, tmp_path):
        examples = load_examples(
            write_labels(tmp_path / "perfect.jsonl", grader_rows(400, flip_every=0))
        )
        labels = {e.id: e.judge_label for e in examples}
        outcome = run_replay(examples, labels, make_config(), batch_size=25)
        assert outcome.verdict is Verdict.TRUSTED
        assert outcome.labels_used < 400


class TestExtractionRegressions:
    def test_nested_json_objects_are_parsed(self):
        settings = ExtractionSettings(method="json_field", field="label")
        text = '{"label": "pass", "confidence": {"score": 0.9}}'
        assert extract_label(text, settings, LABELS) == "pass"

    def test_hyphenated_longer_label_wins_over_its_prefix(self):
        settings = ExtractionSettings(method="label_search")
        allowed = ["pass", "pass-with-warnings", "fail"]
        assert (
            extract_label("Verdict: pass-with-warnings", settings, allowed)
            == "pass-with-warnings"
        )

    def test_cjk_labels_fall_back_to_substring_search(self):
        settings = ExtractionSettings(method="label_search")
        allowed = ["合格", "不合格"]
        assert extract_label("判定は合格です", settings, allowed) == "合格"
        assert extract_label("判定は不合格です", settings, allowed) == "不合格"


class TestClientRegressions:
    def test_retry_after_is_capped(self, monkeypatch):
        sleeps: list[float] = []
        monkeypatch.setattr("judgegate.judge.client.time.sleep", sleeps.append)
        attempts = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["count"] += 1
            if attempts["count"] == 1:
                return httpx.Response(429, headers={"Retry-After": "86400"}, text="wait")
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        client = JudgeClient(make_config().judge, transport=httpx.MockTransport(handler))
        assert client.complete("p") == "ok"
        assert max(sleeps) <= 60.0
        client.close()

    def test_no_sleep_after_final_network_failure(self, monkeypatch):
        sleeps: list[float] = []
        monkeypatch.setattr("judgegate.judge.client.time.sleep", sleeps.append)

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("nope")

        client = JudgeClient(make_config().judge, transport=httpx.MockTransport(handler))
        with pytest.raises(Exception, match="network error"):
            client.complete("p")
        assert len(sleeps) == make_config().judge.retries
        client.close()


class TestCacheRegressions:
    def test_missing_parent_directory_is_created(self, tmp_path):
        cache = ResponseCache(tmp_path / "deep" / "nested" / "cache.sqlite")
        cache.put("k", "v")
        assert cache.get("k") == "v"
        cache.close()

    def test_unopenable_path_raises_config_error_naming_the_setting(self, tmp_path):
        blocker = tmp_path / "blocker"
        blocker.write_text("file, not a directory", encoding="utf-8")
        with pytest.raises(ConfigError, match=r"cache\.path"):
            ResponseCache(blocker / "cache.sqlite")


class TestStripMarkdownSafety:
    def test_math_and_code_survive(self):
        assert format_probe.strip_markdown("x = 2 ** 3 and y = 4 ** 5") == (
            "x = 2 ** 3 and y = 4 ** 5"
        )
        assert format_probe.strip_markdown("area = w * h * d") == "area = w * h * d"
        assert format_probe.strip_markdown("def f(*args, **kwargs)") == (
            "def f(*args, **kwargs)"
        )
        assert format_probe.strip_markdown("delete *.pyc and *.log") == (
            "delete *.pyc and *.log"
        )

    def test_fenced_code_content_is_preserved_without_fences(self):
        text = "```python\nx = a * b * c\n```"
        assert format_probe.strip_markdown(text) == "x = a * b * c\n"

    def test_decoration_still_goes(self):
        assert format_probe.strip_markdown("**bold** and `code`") == "bold and code"
        assert format_probe.strip_markdown("# Title\ntext") == "Title\ntext"


class TestProbeWarnZone:
    def test_unproven_excess_is_warn_not_fail(self, tmp_path):
        rows = [
            {"id": f"s-{i}", "input": f"q{i}", "output": f"a{i}", "human_label": "pass"}
            for i in range(10)
        ]
        examples = load_examples(write_labels(tmp_path / "w.jsonl", rows))
        base = {e.id: "pass" for e in examples}

        def decide(prompt: str) -> str:
            return "fail" if "q1\n" in prompt else "pass"

        payload = json.loads(json.dumps(BASE_CONFIG))
        payload["probes"] = {"stability": {"runs": 2, "max_flip_rate": 0.05}}
        config = Config.model_validate(payload)
        client = JudgeClient(config.judge, transport=json_label_transport(decide))
        result = stability.run(examples, config, client, base)
        client.close()
        assert result.passed is True
        assert result.verdict == "WARN"
        assert result.effect == pytest.approx(0.1)

    def test_warn_probe_does_not_fail_the_gate(self, tmp_path):
        examples = load_examples(
            write_labels(tmp_path / "g.jsonl", grader_rows(120, flip_every=25))
        )
        labels = {e.id: e.judge_label for e in examples}
        warn_probe = ProbeResult(
            name="stability",
            applicable=True,
            passed=True,
            effect=0.12,
            effect_label="mean per-rerun flip rate",
            tolerance=0.10,
            interval=Interval(0.04, 0.24, 0.95),
            detail="slightly above tolerance, unproven",
        )
        report = evaluate_gate(examples, labels, make_config(), [warn_probe])
        assert report.verdict is Verdict.TRUSTED
        assert any("not statistically confirmed" in note for note in report.notes)


class TestVerbosityOrderIndependence:
    def _rows(self):
        rows = []
        for i in range(30):
            long_output = i % 2 == 0
            text = ("word " * 60) if long_output else "word"
            judge = "pass" if long_output else "fail"
            rows.append(
                {
                    "id": f"v-{i}",
                    "input": "q",
                    "output": text,
                    "human_label": "fail",
                    "judge_label": judge,
                }
            )
        return rows

    def test_flags_length_bias_under_both_label_orders(self, tmp_path):
        examples = load_examples(write_labels(tmp_path / "v.jsonl", self._rows()))
        labels = {e.id: str(e.judge_label) for e in examples}
        for order in (["fail", "pass"], ["pass", "fail"]):
            payload = json.loads(json.dumps(BASE_CONFIG))
            payload["labels"] = {"values": order}
            config = Config.model_validate(payload)
            result = verbosity.run(examples, config, labels, "grader")
            assert result.applicable
            assert result.passed is False
