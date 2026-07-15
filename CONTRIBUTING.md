# Contributing

Thanks for your interest in improving judgegate.

## Setup

```bash
git clone https://github.com/yashchimata/judgegate
cd judgegate
python -m venv .venv
. .venv/bin/activate          # .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

## Before opening a pull request

```bash
pytest
ruff check src tests scripts
mypy src
```

All three must pass. CI runs the test suite on Linux across Python 3.11 to
3.13 and on Windows with Python 3.12, plus ruff and mypy on Linux.

## What makes a good contribution here

- **Statistical changes need evidence.** Anything in `judgegate/stats/`
  must include or extend a calibration test that demonstrates coverage or
  error rates on synthetic data with known ground truth. A method that
  cannot be validated that way does not belong in the core.
- **New probes** are welcome when they isolate a real, documented judge
  failure mode, report an effect size with an interval, and skip
  gracefully when not applicable. Look at `probes/format.py` for the
  shape.
- **Client improvements** must keep the offline path untouched: verify
  with precomputed judge labels must never require a network or a key.
- **Report changes** should keep all three renderers (terminal, markdown,
  JSON) consistent with each other.

## How changes land

The `main` branch is protected. Every change, including from the
maintainer, is expected to arrive as a pull request that:

1. passes all required CI checks (lint, the full test matrix, and the
   action smoke test),
2. is approved by a code owner (currently @yashchimata), and
3. lands as a squash merge, keeping history linear.

Force pushes and branch deletion on `main` are disabled. Release tags
(`v*`) cannot be deleted or moved. Workflows run with read-only tokens
unless a job explicitly requests more, and publishing to PyPI requires a
manual approval on the `pypi` environment on top of CI.

First-time contributors will see "workflow awaiting approval" on their
pull request; a maintainer approves the run after a quick look at the
diff. This is a standard defense for public repositories, not a judgment
of your change.

## License of contributions

judgegate is MIT licensed. By submitting a pull request you agree that
your contribution is provided under the same MIT license as the project
(the usual inbound equals outbound norm, also reflected in GitHub's
Terms of Service). You keep the copyright to your work; no copyright
assignment or CLA is required.

## Design rules the project holds to

- The offline path is sacred: no network, no telemetry, no surprises.
- Runtime dependencies stay minimal (numpy, click, rich, pydantic, httpx,
  pyyaml).
- Every public function is typed; `mypy` runs in strict mode.
- Exit codes are a contract: 0 TRUSTED, 1 UNTRUSTED, 2 INCONCLUSIVE,
  3 error. Nothing may repurpose them.
- API keys are read from environment variables only, never from files.

## Reporting bugs

Include the judgegate version (`judgegate --version`), the command you
ran, and if possible a minimal labels file that reproduces the issue. If
the data is sensitive, `judgegate validate` output plus the config (which
never contains keys) is usually enough to start.
