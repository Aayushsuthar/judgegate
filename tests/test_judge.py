import json

import httpx
import pytest

from judgegate.config import ExtractionSettings
from judgegate.data import load_examples
from judgegate.errors import ConfigError, DataError, JudgeError
from judgegate.judge.cache import ResponseCache, response_key
from judgegate.judge.client import JudgeClient
from judgegate.judge.extraction import extract_label
from judgegate.judge.runner import build_prompt, resolve_judge_labels
from tests.conftest import grader_rows, json_label_transport, make_config, write_labels

LABELS = ["fail", "pass"]


class TestExtraction:
    def test_json_field_with_surrounding_prose(self):
        text = 'Let me think.\n{"label": "pass"}\nHope that helps.'
        settings = ExtractionSettings(method="json_field", field="label")
        assert extract_label(text, settings, LABELS) == "pass"

    def test_json_field_uses_last_object(self):
        text = '{"label": "fail"} wait, reconsidering: {"label": "pass"}'
        settings = ExtractionSettings(method="json_field", field="label")
        assert extract_label(text, settings, LABELS) == "pass"

    def test_case_is_normalized(self):
        settings = ExtractionSettings(method="json_field", field="label")
        assert extract_label('{"label": "PASS"}', settings, LABELS) == "pass"

    def test_unknown_label_is_rejected(self):
        settings = ExtractionSettings(method="json_field", field="label")
        with pytest.raises(JudgeError, match="not one of the labels"):
            extract_label('{"label": "excellent"}', settings, LABELS)

    def test_regex_extraction(self):
        settings = ExtractionSettings(method="regex", pattern=r"verdict:\s*(\w+)")
        assert extract_label("verdict: pass", settings, LABELS) == "pass"

    def test_label_search_takes_last_mention(self):
        settings = ExtractionSettings(method="label_search")
        assert extract_label("could pass, but ultimately fail", settings, LABELS) == "fail"

    def test_empty_response(self):
        settings = ExtractionSettings(method="label_search")
        with pytest.raises(JudgeError, match="empty"):
            extract_label("   ", settings, LABELS)


class TestCache:
    def test_round_trip_and_salt_separation(self, tmp_path):
        with ResponseCache(tmp_path / "cache.sqlite") as cache:
            key = response_key("e", "m", 0.0, 100, "prompt")
            salted = response_key("e", "m", 0.0, 100, "prompt", salt="rerun-1")
            assert key != salted
            assert cache.get(key) is None
            cache.put(key, "value")
            assert cache.get(key) == "value"
            assert cache.get(salted) is None


class TestClient:
    def test_missing_api_key_is_a_config_error(self, monkeypatch):
        monkeypatch.delenv("JUDGEGATE_TEST_KEY", raising=False)
        with pytest.raises(ConfigError, match="JUDGEGATE_TEST_KEY"):
            JudgeClient(make_config().judge)

    def test_successful_call_and_cache_hit(self, tmp_path):
        calls = {"count": 0}

        def decide(prompt: str) -> str:
            calls["count"] += 1
            return "pass"

        cache = ResponseCache(tmp_path / "cache.sqlite")
        client = JudgeClient(
            make_config().judge, cache, transport=json_label_transport(decide)
        )
        first = client.complete("same prompt")
        second = client.complete("same prompt")
        assert first == second
        assert calls["count"] == 1
        client.close()
        cache.close()

    def test_retries_on_429_then_succeeds(self, monkeypatch):
        monkeypatch.setattr("judgegate.judge.client.time.sleep", lambda _s: None)
        attempts = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["count"] += 1
            if attempts["count"] == 1:
                return httpx.Response(429, headers={"Retry-After": "1"}, text="slow down")
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "ok"}}]}
            )

        client = JudgeClient(
            make_config().judge, transport=httpx.MockTransport(handler)
        )
        assert client.complete("p") == "ok"
        assert attempts["count"] == 2
        client.close()

    def test_exhausted_retries_raise_judge_error(self, monkeypatch):
        monkeypatch.setattr("judgegate.judge.client.time.sleep", lambda _s: None)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        client = JudgeClient(
            make_config().judge, transport=httpx.MockTransport(handler)
        )
        with pytest.raises(JudgeError, match="HTTP 500"):
            client.complete("p")
        client.close()

    def test_unexpected_shape_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"unexpected": True})

        client = JudgeClient(
            make_config().judge, transport=httpx.MockTransport(handler)
        )
        with pytest.raises(JudgeError, match="unexpected response shape"):
            client.complete("p")
        client.close()

    def test_non_retryable_status_fails_fast(self):
        attempts = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["count"] += 1
            return httpx.Response(401, text="bad key")

        client = JudgeClient(
            make_config().judge, transport=httpx.MockTransport(handler)
        )
        with pytest.raises(JudgeError, match="HTTP 401"):
            client.complete("p")
        assert attempts["count"] == 1
        client.close()


class TestRunner:
    def test_build_prompt_grader_and_pairwise_swap(self, tmp_path):
        rows = [
            {
                "id": "p1",
                "input": "q",
                "output_a": "first",
                "output_b": "second",
                "human_label": "a",
            }
        ]
        examples = load_examples(write_labels(tmp_path / "p.jsonl", rows))
        template = "A: {output_a} B: {output_b}"
        assert build_prompt(template, examples[0]) == "A: first B: second"
        assert build_prompt(template, examples[0], swap_pairwise=True) == "A: second B: first"

    def test_resolve_prefers_precomputed_labels(self, tmp_path):
        examples = load_examples(write_labels(tmp_path / "l.jsonl", grader_rows(12)))
        labels = resolve_judge_labels(examples, make_config(), client=None)
        assert len(labels) == 12

    def test_resolve_rejects_partial_precomputed(self, tmp_path):
        rows = grader_rows(4)
        del rows[0]["judge_label"]
        examples = load_examples(write_labels(tmp_path / "l.jsonl", rows))
        with pytest.raises(DataError, match="lack a judge_label"):
            resolve_judge_labels(examples, make_config(), client=None)

    def test_resolve_rejects_unknown_precomputed_label(self, tmp_path):
        rows = grader_rows(4)
        rows[1]["judge_label"] = "excellent"
        examples = load_examples(write_labels(tmp_path / "l.jsonl", rows))
        with pytest.raises(DataError, match="excellent"):
            resolve_judge_labels(examples, make_config(), client=None)

    def test_resolve_calls_judge_when_no_precomputed(self, tmp_path):
        rows = grader_rows(6)
        for row in rows:
            del row["judge_label"]
        examples = load_examples(write_labels(tmp_path / "l.jsonl", rows))

        def decide(prompt: str) -> str:
            payload = json.loads('{"x": 1}')
            assert payload
            return "pass"

        client = JudgeClient(
            make_config().judge, transport=json_label_transport(decide)
        )
        labels = resolve_judge_labels(examples, make_config(), client)
        assert set(labels.values()) == {"pass"}
        client.close()

    def test_offline_without_client_is_explained(self, tmp_path):
        rows = grader_rows(4)
        for row in rows:
            del row["judge_label"]
        examples = load_examples(write_labels(tmp_path / "l.jsonl", rows))
        with pytest.raises(DataError, match="no judge endpoint"):
            resolve_judge_labels(examples, make_config(), client=None)
