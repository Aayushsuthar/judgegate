import json
from pathlib import Path

import httpx
import pytest

from judgegate.config import Config

BASE_CONFIG: dict = {
    "judge": {
        "model": "test-judge",
        "api_key_env": "JUDGEGATE_TEST_KEY",
        "prompt": (
            "Grade the answer.\nQuestion: {input}\nAnswer: {output}\n"
            'Respond with JSON: {"label": "pass"} or {"label": "fail"}.'
        ),
        "retries": 1,
    },
    "labels": {"values": ["fail", "pass"]},
    "gate": {"min_kappa": 0.6, "seed": 7, "resamples": 1500},
}

PAIRWISE_CONFIG: dict = {
    "judge": {
        "model": "test-judge",
        "api_key_env": "JUDGEGATE_TEST_KEY",
        "prompt": (
            "Pick the better answer.\nQuestion: {input}\n"
            "A: {output_a}\nB: {output_b}\n"
            'Respond with JSON: {"label": "a"} or {"label": "b"}.'
        ),
        "retries": 1,
    },
    "labels": {"values": ["a", "b"]},
    "gate": {"min_kappa": 0.6, "seed": 7, "resamples": 1500},
}


def make_config(payload: dict | None = None) -> Config:
    return Config.model_validate(payload or BASE_CONFIG)


def write_labels(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def grader_rows(n: int, flip_every: int = 0) -> list[dict]:
    rows = []
    for i in range(n):
        human = "pass" if i % 3 else "fail"
        judge = human
        if flip_every and i % flip_every == 0:
            judge = "fail" if human == "pass" else "pass"
        rows.append(
            {
                "id": f"ex-{i:03d}",
                "input": f"question {i}",
                "output": f"answer {i} " + ("with extra detail " * (i % 4)),
                "human_label": human,
                "judge_label": judge,
            }
        )
    return rows


def json_label_transport(decide) -> httpx.MockTransport:
    """Transport whose judge answers with a JSON label chosen by ``decide``."""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        prompt = payload["messages"][0]["content"]
        label = decide(prompt)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps({"label": label})}}]},
        )

    return httpx.MockTransport(handler)


@pytest.fixture(autouse=True)
def _test_api_key(monkeypatch):
    monkeypatch.setenv("JUDGEGATE_TEST_KEY", "test-key")
