import json
import subprocess
import sys

import pytest
from click.testing import CliRunner

from judgegate.cli import cli, main
from judgegate.report import MARKER
from tests.conftest import grader_rows, write_labels

CONFIG_YAML = """\
judge:
  model: test-judge
  api_key_env: JUDGEGATE_TEST_KEY
  prompt: |
    Grade the answer.
    Question: {input}
    Answer: {output}
    Respond with JSON: {"label": "pass"} or {"label": "fail"}.
labels:
  values: [fail, pass]
gate:
  min_kappa: 0.6
  seed: 7
  resamples: 1500
"""


@pytest.fixture
def workspace(tmp_path):
    config = tmp_path / "judge.yaml"
    config.write_text(CONFIG_YAML, encoding="utf-8")
    write_labels(tmp_path / "trusted.jsonl", grader_rows(120, flip_every=25))
    write_labels(tmp_path / "untrusted.jsonl", grader_rows(120, flip_every=2))
    write_labels(tmp_path / "small.jsonl", grader_rows(24, flip_every=6))
    return tmp_path


@pytest.fixture
def runner():
    return CliRunner()


class TestVerify:
    def test_trusted_exits_zero(self, runner, workspace):
        result = runner.invoke(
            cli,
            ["verify", str(workspace / "trusted.jsonl"), "--config",
             str(workspace / "judge.yaml")],
        )
        assert result.exit_code == 0
        assert "TRUSTED" in result.output

    def test_untrusted_exits_one(self, runner, workspace):
        result = runner.invoke(
            cli,
            ["verify", str(workspace / "untrusted.jsonl"), "--config",
             str(workspace / "judge.yaml")],
        )
        assert result.exit_code == 1
        assert "UNTRUSTED" in result.output

    def test_inconclusive_exits_two(self, runner, workspace):
        result = runner.invoke(
            cli,
            ["verify", str(workspace / "small.jsonl"), "--config",
             str(workspace / "judge.yaml")],
        )
        assert result.exit_code == 2
        assert "INCONCLUSIVE" in result.output

    def test_json_format_round_trips(self, runner, workspace):
        result = runner.invoke(
            cli,
            ["verify", str(workspace / "trusted.jsonl"), "--config",
             str(workspace / "judge.yaml"), "--format", "json"],
        )
        payload = json.loads(result.output)
        assert payload["verdict"] == "TRUSTED"
        assert payload["exit_code"] == 0
        assert payload["n_labels"] == 120
        assert len(payload["probes"]) == 4

    def test_markdown_format(self, runner, workspace):
        result = runner.invoke(
            cli,
            ["verify", str(workspace / "trusted.jsonl"), "--config",
             str(workspace / "judge.yaml"), "--format", "markdown"],
        )
        assert MARKER in result.output

    @pytest.mark.parametrize("output_format", ["terminal", "markdown", "json"])
    def test_output_file(self, runner, workspace, tmp_path, output_format):
        out = tmp_path / "report.txt"
        result = runner.invoke(
            cli,
            ["verify", str(workspace / "trusted.jsonl"), "--config",
             str(workspace / "judge.yaml"), "--format", output_format,
             "--output", str(out)],
        )
        assert result.exit_code == 0
        assert "report written to" in result.output
        content = out.read_text(encoding="utf-8")
        if output_format == "json":
            assert json.loads(content)["verdict"] == "TRUSTED"
        else:
            assert "TRUSTED" in content

    def test_min_kappa_override_changes_verdict(self, runner, workspace):
        result = runner.invoke(
            cli,
            ["verify", str(workspace / "trusted.jsonl"), "--config",
             str(workspace / "judge.yaml"), "--min-kappa", "0.95"],
        )
        assert result.exit_code in (1, 2)

    def test_no_probes_flag(self, runner, workspace):
        result = runner.invoke(
            cli,
            ["verify", str(workspace / "trusted.jsonl"), "--config",
             str(workspace / "judge.yaml"), "--no-probes", "--format", "json"],
        )
        payload = json.loads(result.output)
        assert payload["probes"] == []


class TestPower:
    def test_with_labels_file(self, runner, workspace):
        result = runner.invoke(
            cli,
            ["power", "--labels", str(workspace / "trusted.jsonl"), "--config",
             str(workspace / "judge.yaml"), "--format", "json"],
        )
        payload = json.loads(result.output)
        assert payload["required_labels"] is not None
        assert payload["current_labels"] == 120
        assert result.exit_code == 0

    def test_with_manual_marginals(self, runner):
        result = runner.invoke(
            cli,
            ["power", "--marginals", "0.7,0.3", "--kappa", "0.85",
             "--threshold", "0.6", "--seed", "3", "--format", "json"],
        )
        payload = json.loads(result.output)
        assert payload["required_labels"] is not None

    def test_requires_labels_or_marginals(self, runner):
        result = runner.invoke(cli, ["power", "--kappa", "0.8"])
        assert result.exit_code != 0
        assert "either --labels or --marginals" in result.output

    def test_requires_kappa_when_not_inferable(self, runner):
        result = runner.invoke(cli, ["power", "--marginals", "0.5,0.5"])
        assert result.exit_code != 0
        assert "--kappa" in result.output


class TestSequential:
    def test_replay_json(self, runner, workspace):
        result = runner.invoke(
            cli,
            ["sequential", str(workspace / "trusted.jsonl"), "--config",
             str(workspace / "judge.yaml"), "--format", "json"],
        )
        payload = json.loads(result.output)
        assert payload["verdict"] in ("TRUSTED", "INCONCLUSIVE")
        assert result.exit_code == payload["exit_code"]

    def test_untrusted_replay(self, runner, workspace):
        result = runner.invoke(
            cli,
            ["sequential", str(workspace / "untrusted.jsonl"), "--config",
             str(workspace / "judge.yaml")],
        )
        assert result.exit_code == 1


class TestValidate:
    def test_valid_file_summary(self, runner, workspace):
        result = runner.invoke(cli, ["validate", str(workspace / "trusted.jsonl")])
        assert result.exit_code == 0
        assert "OK" in result.output
        assert "fully offline" in result.output


class TestMainEntry:
    def test_version(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["judgegate", "--version"])
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 0
        assert "judgegate" in capsys.readouterr().out

    def test_missing_labels_file_exits_three(self, monkeypatch, capsys, workspace):
        monkeypatch.setattr(
            sys,
            "argv",
            ["judgegate", "verify", str(workspace / "nope.jsonl"), "--config",
             str(workspace / "judge.yaml")],
        )
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 3

    def test_usage_error_exits_three(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["judgegate", "verify"])
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 3

    def test_module_execution_works(self):
        result = subprocess.run(
            [sys.executable, "-m", "judgegate", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "judgegate" in result.stdout
