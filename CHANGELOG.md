# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-07-15

### Added

- `judgegate verify`: trust gate for LLM judges with TRUSTED, UNTRUSTED,
  and INCONCLUSIVE verdicts, built on Cohen's kappa (and weighted kappa
  for ordinal labels) with bias-corrected bootstrap confidence intervals
  against human labels.
- `judgegate power`: label budget analysis answering how many human labels
  a trustworthy verdict requires, and the smallest true kappa the current
  label count can certify.
- `judgegate sequential`: always-valid sequential boundaries over
  agreement indicators, showing where the labeling effort could stop.
- `judgegate probe`: bias battery covering rerun stability, position swap
  for pairwise judges, an offline verbosity diagnostic, and markdown
  format perturbation, each with effect sizes and tolerances.
- `judgegate validate`: pre-flight check for labels files.
- Offline mode: labels files with precomputed judge verdicts run with no
  network and no API key.
- Judge client for any OpenAI-compatible endpoint with retries, bounded
  concurrency, and a SQLite response cache.
- Renderers: rich terminal reports with kappa error bars, GitHub flavored
  markdown, and JSON.
- Composite GitHub Action with sticky pull request comments, job summary
  output, and verdict-based check enforcement.
- Exit code contract: 0 TRUSTED, 1 UNTRUSTED, 2 INCONCLUSIVE,
  3 operational error.
