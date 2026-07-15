# The labels file

judgegate reads human-labeled examples from JSON Lines: one JSON object
per line, UTF-8, blank lines ignored.

## Grader mode

The judge grades one output at a time:

```json
{"id": "q-001", "input": "What is a mutex?", "output": "A lock that ensures one thread at a time.", "human_label": "pass"}
{"id": "q-002", "input": "Capital of France?", "output": "Lyon.", "human_label": "fail"}
```

## Pairwise mode

The judge picks the better of two outputs. Labels must include `a` and
`b`; add `tie` to the label set if your judge may refuse to pick:

```json
{"id": "p-001", "input": "Summarize the report.", "output_a": "...", "output_b": "...", "human_label": "a"}
```

A file must be entirely one mode; mixing is an error.

## Fields

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | Unique per file. Used for caching, probe pairing, and error messages. |
| `input` | no | The task shown to the judge via the `{input}` placeholder. |
| `output` | grader mode | The response being judged. |
| `output_a`, `output_b` | pairwise mode | The two candidates. |
| `human_label` | yes | Your annotator's verdict. Must be one of `labels.values`. |
| `judge_label` | no | Precomputed judge verdict. See offline mode below. |
| `metadata` | no | Anything else. Unknown top-level keys are folded in here rather than rejected. |

## Offline mode

When every row carries a `judge_label`, judgegate never contacts any
endpoint: `verify`, `power`, and `sequential` run from the file alone, no
API key required. This is the recommended CI setup: export judge verdicts
from your existing eval runs, commit or artifact the labels file, and the
gate is deterministic and free.

When no row has a `judge_label`, judgegate calls the judge from
`judge.yaml` and caches every response. Files where only some rows have
judge labels are rejected, because a silently mixed origin would make the
statistics unreadable.

## Practical guidance

- Aim for label sets that reflect real traffic, including the ugly cases.
  A judge certified on easy items is certified for easy items.
- Keep ids stable across exports so caches and diffs line up.
- `judgegate validate labels.jsonl` reports counts, mode, label balance,
  and whether the file supports offline mode.
- If two annotators disagree with each other, measure that first by
  putting one annotator in `human_label` and the other in `judge_label`;
  the same kappa machinery applies.
