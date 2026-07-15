import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from judgegate.errors import DataError

DatasetMode = Literal["grader", "pairwise"]

_KNOWN_KEYS = {
    "id",
    "input",
    "output",
    "output_a",
    "output_b",
    "human_label",
    "judge_label",
    "metadata",
}


class LabeledExample(BaseModel):
    """One human-labeled example, optionally with a precomputed judge label.

    Grader mode uses ``output``; pairwise mode uses ``output_a`` and
    ``output_b``. Every example carries the human verdict; the judge
    verdict is filled in either from the file (offline mode) or by
    calling the judge.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    input: str = ""
    output: str | None = None
    output_a: str | None = None
    output_b: str | None = None
    human_label: str = Field(min_length=1)
    judge_label: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_mode(self) -> Self:
        pairwise = self.output_a is not None or self.output_b is not None
        if pairwise:
            if self.output is not None:
                raise ValueError("provide either output or output_a/output_b, not both")
            if self.output_a is None or self.output_b is None:
                raise ValueError("pairwise examples need both output_a and output_b")
        elif self.output is None:
            raise ValueError("grader examples need an output field")
        return self

    @property
    def mode(self) -> DatasetMode:
        return "pairwise" if self.output_a is not None else "grader"

    @property
    def output_length(self) -> int:
        if self.output is not None:
            return len(self.output)
        return len(self.output_a or "") + len(self.output_b or "")


def load_examples(path: Path) -> list[LabeledExample]:
    """Parse a JSON Lines labels file, one example per line."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DataError(f"could not read {path}: {exc}") from exc

    examples: list[LabeledExample] = []
    seen: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            item = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise DataError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(item, dict):
            raise DataError(f"{path}:{line_number}: expected an object")
        metadata_field = item.get("metadata")
        metadata: dict[str, Any] = (
            dict(metadata_field) if isinstance(metadata_field, dict) else {}
        )
        metadata.update({k: v for k, v in item.items() if k not in _KNOWN_KEYS})
        payload = {k: v for k, v in item.items() if k in _KNOWN_KEYS and k != "metadata"}
        payload["metadata"] = metadata
        try:
            example = LabeledExample(**payload)
        except ValidationError as exc:
            first = exc.errors()[0]
            location = ".".join(str(part) for part in first["loc"]) or "example"
            raise DataError(f"{path}:{line_number}: {location}: {first['msg']}") from exc
        if example.id in seen:
            raise DataError(f"{path}:{line_number}: duplicate example id {example.id!r}")
        seen.add(example.id)
        examples.append(example)

    if not examples:
        raise DataError(f"{path}: no examples found")
    return examples


def dataset_mode(examples: Sequence[LabeledExample]) -> DatasetMode:
    """The single mode of a dataset; mixing modes is an error."""
    modes = {example.mode for example in examples}
    if len(modes) > 1:
        raise DataError("labels file mixes grader and pairwise examples")
    return next(iter(modes))


def observed_labels(examples: Sequence[LabeledExample]) -> list[str]:
    """Sorted distinct human labels, used when no label set is configured."""
    return sorted({example.human_label for example in examples})
