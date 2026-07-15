# GitHub Action

The repository doubles as a composite GitHub Action that verifies the
judge and posts the report to the pull request.

## Minimal setup

```yaml
name: Judge gate
on: pull_request

jobs:
  judge-gate:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
    steps:
      - uses: actions/checkout@v7
      - uses: yashchimata/judgegate@v0.1.0
        with:
          labels: evals/judge-labels.jsonl
          config: evals/judge.yaml
```

The action:

1. Installs judgegate into the runner's Python.
2. Runs `judgegate verify` with markdown output.
3. Appends the report to the job summary.
4. Posts the report as a sticky comment on the pull request, updating the
   same comment on subsequent pushes.
5. Fails the check only when the verdict is UNTRUSTED, or when the verdict
   is INCONCLUSIVE and `fail-on-inconclusive` is set.

`permissions: pull-requests: write` is required for the comment. Without
it, set `comment: "false"` and rely on the job summary.

## Offline or live

The clean CI pattern is offline: commit or artifact a labels file whose
rows include `judge_label` (exported from your eval runs), and the gate
needs no API key at all.

For live judging, provide the key as a secret through the environment
variable named in your config's `api_key_env`:

```yaml
      - uses: yashchimata/judgegate@v0.1.0
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        with:
          labels: evals/judge-labels.jsonl
          config: evals/judge.yaml
```

## Inputs

| Input | Default | Meaning |
|---|---|---|
| `labels` | required | Path to the labels file. |
| `config` | required | Path to `judge.yaml`. |
| `min-kappa` | from config | Trust threshold override. |
| `alpha` | from config | Significance level override. |
| `no-probes` | `false` | Skip the probe battery. |
| `comment` | `true` | Post the sticky pull request comment. |
| `fail-on-inconclusive` | `false` | Treat INCONCLUSIVE as a failure. |
| `github-token` | `github.token` | Token used for the comment. |
| `python-version` | `3.12` | Python used to run judgegate. |

## Outputs

| Output | Meaning |
|---|---|
| `verdict` | `TRUSTED`, `UNTRUSTED`, `INCONCLUSIVE`, or `ERROR`. |
| `exit-code` | The raw judgegate exit code. |

## When should this gate run?

Any change that touches the judge: its prompt, its model, its temperature,
its extraction format, or the label set. Teams that version `judge.yaml`
in the repository typically scope the workflow with a `paths:` filter so
the gate runs exactly when the judge changes:

```yaml
on:
  pull_request:
    paths:
      - "evals/judge.yaml"
      - "evals/judge-labels.jsonl"
```
