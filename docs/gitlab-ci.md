# GitLab CI

judgegate does not ship a native GitLab CI/CD component, but the CLI is a single `pip install` away, so wiring it into a GitLab pipeline is a plain shell job: run `judgegate verify`, map the exit code, and post the markdown report as a merge request note through the GitLab REST API.

## Minimal setup

```yaml
judge-gate:
  stage: test
  image: python:3.12-slim
  rules:
    - if: '$CI_PIPELINE_SOURCE == "merge_request_event"'
  script:
    - pip install judgegate
    - >
      judgegate verify evals/judge-labels.jsonl
      --config evals/judge.yaml
      --format markdown
      --output judge-report.md
  after_script:
    - |
      if [ -n "$CI_MERGE_REQUEST_IID" ] && [ -f judge-report.md ]; then
        curl --silent --show-error \
          --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
          --form "body=<judge-report.md" \
          "${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/merge_requests/${CI_MERGE_REQUEST_IID}/notes"
      fi
  artifacts:
    when: always
    paths:
      - judge-report.md
  allow_failure:
    exit_codes: [2]
```

The clean pattern is offline, same as the GitHub Action: commit or artifact a labels file whose rows already include `judge_label` (exported from your eval runs), and the job needs no LLM API key at all. For live judging, add the judge endpoint's key as a masked CI/CD variable and reference it through the environment variable named in your config's `api_key_env`.

## Exit codes

`judgegate verify` uses four exit codes. `0` (TRUSTED) is a normal pass. `1` (UNTRUSTED) fails the job, since GitLab fails a job on any non-zero exit code by default. `2` (INCONCLUSIVE) means there are not enough labels to decide either way; the `allow_failure: exit_codes: [2]` shown above lets the pipeline continue while still surfacing the job as an "allowed to fail" warning instead of a hard failure. `3` is an operational error (bad file, bad flag, endpoint error) and is left to fail the job like `1`, since something about the setup, not the judge, is broken.

## Posting the report as a merge request note

`--format markdown --output judge-report.md` writes the same report the GitHub Action posts, to a file instead of stdout. The `after_script` step above uploads that file's contents as a new note through the [merge request notes API](https://docs.gitlab.com/ee/api/notes.html#create-new-merge-request-note), guarded so it only runs in merge request pipelines and only when the report file exists (it will be absent if `judgegate verify` failed to run at all, for example on a missing config). `GITLAB_TOKEN` needs at least the `api` scope; a project access token stored as a masked, protected CI/CD variable is the simplest way to provide it, since the built-in `CI_JOB_TOKEN`'s permissions for the notes API vary by GitLab version and plan. Unlike the GitHub Action, this recipe always posts a new note rather than editing one in place, since an in-place edit needs an extra API call to find the previous note first.

## When should this gate run?

As with the GitHub Action, scope the job to run when the judge itself changes:

```yaml
judge-gate:
  rules:
    - if: '$CI_PIPELINE_SOURCE == "merge_request_event"'
      changes:
        - evals/judge.yaml
        - evals/judge-labels.jsonl
```
