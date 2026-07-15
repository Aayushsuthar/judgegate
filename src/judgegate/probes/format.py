import re
from collections.abc import Sequence

from judgegate.config import Config
from judgegate.data import LabeledExample
from judgegate.judge.client import JudgeClient
from judgegate.judge.runner import judge_examples
from judgegate.probes.base import ProbeResult, proven_beyond, skipped
from judgegate.stats.intervals import wilson_proportion_interval

NAME = "format"
EFFECT = "verdict flips when markdown is stripped"

_FENCED_CODE = re.compile(r"```[a-zA-Z0-9]*\n?(.*?)```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_BOLD = re.compile(r"\*\*(?=\S)([^*]+?)(?<=\S)\*\*")
_ITALIC = re.compile(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])")
_HEADING = re.compile(r"^#{1,6}\s+", re.MULTILINE)

_PLACEHOLDER = "\x00{}\x00"


def strip_markdown(text: str) -> str:
    """Remove markdown decoration while preserving the words.

    Code spans and fenced blocks are protected before emphasis is
    stripped, so multiplication signs, glob patterns, and ``**kwargs``
    survive untouched; only their backticks and fence markers go. The
    emphasis patterns require the delimiters to hug non-whitespace, so
    prose like ``2 ** 3`` is never treated as bold.
    """
    stash: list[str] = []

    def protect(match: re.Match[str]) -> str:
        stash.append(match.group(1))
        return _PLACEHOLDER.format(len(stash) - 1)

    result = _FENCED_CODE.sub(protect, text)
    result = _INLINE_CODE.sub(protect, result)
    result = _BOLD.sub(r"\1", result)
    result = _ITALIC.sub(r"\1", result)
    result = _HEADING.sub("", result)
    for index, content in enumerate(stash):
        result = result.replace(_PLACEHOLDER.format(index), content)
    return result


def _strip_example(example: LabeledExample) -> LabeledExample:
    if example.mode == "grader":
        return example.model_copy(update={"output": strip_markdown(example.output or "")})
    return example.model_copy(
        update={
            "output_a": strip_markdown(example.output_a or ""),
            "output_b": strip_markdown(example.output_b or ""),
        }
    )


def run(
    examples: Sequence[LabeledExample],
    config: Config,
    client: JudgeClient | None,
    base_labels: dict[str, str],
) -> ProbeResult:
    """Strip markdown decoration and check the verdicts hold.

    The words are identical; only bold, headings, backticks, and code
    fences are removed. A judge whose verdict changes is grading
    typography. Only items whose text actually contains markdown
    participate.
    """
    settings = config.probes.format
    if not settings.enabled:
        return skipped(NAME, EFFECT, "disabled in config")
    if client is None:
        return skipped(NAME, EFFECT, "requires a judge endpoint; offline labels only")

    affected = [e for e in examples if _strip_example(e) != e]
    if len(affected) < 5:
        return skipped(
            NAME, EFFECT, f"only {len(affected)} item(s) contain markdown to strip"
        )

    stripped_examples = [_strip_example(e) for e in affected]
    stripped_labels = judge_examples(stripped_examples, config, client)
    flips = sum(1 for e in affected if stripped_labels[e.id] != base_labels[e.id])

    rate = flips / len(affected)
    interval = wilson_proportion_interval(rate, len(affected), config.gate.confidence)
    passed = not proven_beyond(interval, settings.max_flip_rate)
    return ProbeResult(
        name=NAME,
        applicable=True,
        passed=passed,
        effect=rate,
        effect_label=EFFECT,
        tolerance=settings.max_flip_rate,
        interval=interval,
        detail=(
            f"{flips} of {len(affected)} markdown-bearing items changed verdict "
            "after decoration was removed"
        ),
    )
