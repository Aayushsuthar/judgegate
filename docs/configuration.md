# Configuration

Everything judgegate needs lives in one YAML file, passed with
`--config judge.yaml`. Unknown keys are rejected with an error naming the
key, so typos fail loudly instead of silently using defaults.

## Full reference

```yaml
judge:
  endpoint: https://api.openai.com/v1
  model: gpt-4o-mini
  api_key_env: OPENAI_API_KEY
  temperature: 0.0
  max_tokens: 512
  timeout_seconds: 60
  retries: 4
  concurrency: 4
  prompt: |
    Question: {input}
    Answer: {output}
    Respond with JSON only: {"label": "pass"} or {"label": "fail"}.
  extraction:
    method: json_field
    field: label

labels:
  values: [fail, pass]
  ordinal: false

gate:
  min_kappa: 0.6
  alpha: 0.05
  power: 0.8
  weighting: none
  resamples: 10000
  seed: 42
  min_labels: 20

probes:
  stability:
    enabled: true
    runs: 3
    max_flip_rate: 0.10
  position:
    enabled: true
    max_flip_rate: 0.15
  verbosity:
    enabled: true
    max_leniency_gap: 0.15
  format:
    enabled: true
    max_flip_rate: 0.10

cache:
  enabled: true
  path: .judgegate-cache.sqlite
```

## [judge]

| Key | Default | Meaning |
|---|---|---|
| `endpoint` | OpenAI | Any OpenAI-compatible chat completions base URL, including local servers. |
| `model` | required | Model name sent in the request. |
| `api_key_env` | `OPENAI_API_KEY` | Environment variable holding the key. judgegate never reads the key from config files, so configs are safe to commit. |
| `temperature` | `0.0` | Judge sampling temperature. The stability probe measures churn at whatever you set here. |
| `prompt` | required | The judge prompt. Placeholders `{input}` and `{output}` (grader) or `{output_a}` and `{output_b}` (pairwise) are substituted; other braces are left alone, so JSON examples in prompts are safe. |
| `extraction.method` | `json_field` | How to pull the label from the response: `json_field` (last JSON object's field), `regex` (last match, first capture group), or `label_search` (last whole-word label mention). |
| `retries` / `timeout_seconds` / `concurrency` | 4 / 60 / 4 | Transient failures retry with backoff and honor Retry-After headers. |

## [labels]

| Key | Default | Meaning |
|---|---|---|
| `values` | required | The label vocabulary, at least two distinct values, listed from worst to best. The order defines the leniency direction used by the verbosity probe and, for ordinal sets, the weighted kappa distances. |
| `ordinal` | `false` | Declares the order meaningful, enabling weighted kappa and the verbosity probe for multi-class labels. |

## [gate]

| Key | Default | Meaning |
|---|---|---|
| `min_kappa` | `0.6` | Trust threshold. TRUSTED needs the entire confidence interval above it. 0.6 is a common bar for substantial agreement; raise it when the judge gates releases, lower it for advisory dashboards. |
| `alpha` | `0.05` | Significance level; the interval covers `1 - alpha`. |
| `power` | `0.8` | Target power for label budget answers. |
| `weighting` | `none` | `linear` or `quadratic` weighted kappa for ordinal labels. |
| `resamples` | `10000` | Bootstrap resamples. |
| `seed` | unset | Set in CI for reproducible reports. |
| `min_labels` | `20` | The gate never certifies a judge on fewer labels than this, regardless of the interval. |

## [probes]

Each probe has `enabled` plus a tolerance. A probe fails only when its
confidence interval proves the effect exceeds the tolerance; a point
estimate above tolerance without statistical confirmation is reported as
WARN. Probe failures force UNTRUSTED; skipped probes are reported but
never fail the gate. Tolerances are effect-size bounds: for stability,
`max_flip_rate: 0.10` bounds the mean per-rerun flip rate, so it stays
comparable when `runs` changes.

## [cache]

Judge responses are cached in SQLite keyed by a content hash of the
endpoint, model, parameters, and prompt; the response text is stored,
the prompt is not. Delete the file to force fresh calls, and add it to
your repository's .gitignore in live mode. Stability probe reruns use
distinct cache keys on purpose, so they measure the judge, not the
cache.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | TRUSTED (verify, sequential); success (power, validate, probe with no failures) |
| 1 | UNTRUSTED, or any probe failure from `judgegate probe` |
| 2 | INCONCLUSIVE |
| 3 | Operational error (bad file, bad flags, endpoint failure) |
