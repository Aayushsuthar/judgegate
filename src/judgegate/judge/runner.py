from collections.abc import Sequence

from judgegate.config import Config, render_prompt
from judgegate.data import LabeledExample
from judgegate.errors import DataError, JudgeError
from judgegate.judge.client import JudgeClient
from judgegate.judge.extraction import extract_label


def build_prompt(
    template: str, example: LabeledExample, swap_pairwise: bool = False
) -> str:
    """Render the judge prompt for one example."""
    mapping = {"input": example.input}
    if example.mode == "grader":
        mapping["output"] = example.output or ""
    else:
        first = example.output_b if swap_pairwise else example.output_a
        second = example.output_a if swap_pairwise else example.output_b
        mapping["output_a"] = first or ""
        mapping["output_b"] = second or ""
    return render_prompt(template, mapping)


def judge_examples(
    examples: Sequence[LabeledExample],
    config: Config,
    client: JudgeClient,
    swap_pairwise: bool = False,
    salt: str = "",
    prompt_transform: str | None = None,
) -> dict[str, str]:
    """Run the judge over examples and return extracted labels by example id.

    ``prompt_transform`` optionally replaces the template for probe
    variants. Extraction failures name the offending example so a single
    malformed response is debuggable instead of mysterious.
    """
    template = prompt_transform if prompt_transform is not None else config.judge.prompt
    prompts = {
        example.id: build_prompt(template, example, swap_pairwise) for example in examples
    }
    responses = client.complete_many(prompts, salt)
    labels: dict[str, str] = {}
    for example_id, text in responses.items():
        try:
            labels[example_id] = extract_label(
                text, config.judge.extraction, config.labels.values
            )
        except JudgeError as exc:
            raise JudgeError(f"example {example_id!r}: {exc}") from exc
    return labels


def resolve_judge_labels(
    examples: Sequence[LabeledExample],
    config: Config,
    client: JudgeClient | None,
) -> dict[str, str]:
    """Judge labels for every example, from the file or from the endpoint.

    Precomputed ``judge_label`` fields win when every example has one,
    which keeps verify fully offline for teams that already log judge
    output. Otherwise a client is required and the judge is called.
    """
    precomputed = {
        example.id: example.judge_label
        for example in examples
        if example.judge_label is not None
    }
    if len(precomputed) == len(examples):
        allowed = {label.lower() for label in config.labels.values}
        for example_id, label in precomputed.items():
            if label is None or label.lower() not in allowed:
                raise DataError(
                    f"example {example_id!r} has judge_label {label!r}, "
                    f"which is not in the configured labels {list(config.labels.values)}"
                )
        lookup = {label.lower(): label for label in config.labels.values}
        return {k: lookup[str(v).lower()] for k, v in precomputed.items()}
    if precomputed:
        missing = len(examples) - len(precomputed)
        raise DataError(
            f"{missing} example(s) lack a judge_label while others have one; "
            "either precompute all judge labels or none"
        )
    if client is None:
        raise DataError(
            "the labels file has no judge_label fields and no judge endpoint "
            "is available; add judge labels to the file or configure the judge"
        )
    return judge_examples(examples, config, client)
