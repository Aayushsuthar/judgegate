import json
import re
from collections.abc import Iterator, Sequence
from typing import Any

from judgegate.config import ExtractionSettings
from judgegate.errors import JudgeError

_DECODER = json.JSONDecoder()


def _normalize(candidate: str, allowed: Sequence[str]) -> str:
    lookup = {label.strip().lower(): label for label in allowed}
    key = candidate.strip().strip('"').strip("'").lower()
    if key in lookup:
        return lookup[key]
    raise JudgeError(
        f"judge answered {candidate!r}, which is not one of the labels {list(allowed)}"
    )


def _iter_json_objects(text: str) -> Iterator[dict[str, Any]]:
    index = text.find("{")
    while index != -1:
        try:
            payload, _end = _DECODER.raw_decode(text, index)
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            yield payload
        index = text.find("{", index + 1)


def _from_json_field(text: str, field: str, allowed: Sequence[str]) -> str:
    found = None
    for payload in _iter_json_objects(text):
        if field in payload:
            found = payload[field]
    if found is not None:
        return _normalize(str(found), allowed)
    raise JudgeError(
        f"no JSON object with a {field!r} field found in the judge response; "
        f"response started with {text[:120]!r}"
    )


def _from_regex(text: str, pattern: str, allowed: Sequence[str]) -> str:
    try:
        compiled = re.compile(pattern, re.IGNORECASE | re.DOTALL)
    except re.error as exc:
        raise JudgeError(f"invalid extraction pattern {pattern!r}: {exc}") from exc
    matches = list(compiled.finditer(text))
    if not matches:
        raise JudgeError(
            f"extraction pattern found no match; response started with {text[:120]!r}"
        )
    last = matches[-1]
    candidate = last.group(1) if last.groups() else last.group(0)
    return _normalize(candidate, allowed)


def _is_word_char(char: str) -> bool:
    return char.isalnum() or char == "_"


def _from_label_search(text: str, allowed: Sequence[str]) -> str:
    best: tuple[int, int, str] | None = None
    for label in allowed:
        prefix = r"\b" if _is_word_char(label[0]) else ""
        suffix = r"\b" if _is_word_char(label[-1]) else ""
        pattern = re.compile(prefix + re.escape(label) + suffix, re.IGNORECASE)
        matches = list(pattern.finditer(text))
        if matches:
            candidate = (matches[-1].end(), len(label), label)
            if best is None or (candidate[0], candidate[1]) > (best[0], best[1]):
                best = candidate
    if best is None:
        lowered = text.lower()
        for label in allowed:
            position = lowered.rfind(label.lower())
            if position != -1:
                candidate = (position + len(label), len(label), label)
                if best is None or (candidate[0], candidate[1]) > (best[0], best[1]):
                    best = candidate
    if best is None:
        raise JudgeError(
            f"none of the labels {list(allowed)} appear in the judge response; "
            f"response started with {text[:120]!r}"
        )
    return best[2]


def extract_label(text: str, settings: ExtractionSettings, allowed: Sequence[str]) -> str:
    """Pull the verdict label out of a judge response."""
    if not text.strip():
        raise JudgeError("the judge returned an empty response")
    if settings.method == "json_field":
        return _from_json_field(text, settings.field, allowed)
    if settings.method == "regex":
        if settings.pattern is None:
            raise JudgeError("regex extraction is missing its pattern")
        return _from_regex(text, settings.pattern, allowed)
    return _from_label_search(text, allowed)
