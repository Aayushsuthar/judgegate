from judgegate.stats.intervals import Interval

_DEFAULT_WIDTH = 61


def _column(value: float, axis_low: float, axis_high: float, width: int) -> int:
    span = axis_high - axis_low
    if span <= 0.0:
        return width // 2
    position = (value - axis_low) / span
    return min(width - 1, max(0, round(position * (width - 1))))


def _place_label(row: list[str], column: int, label: str, width: int) -> None:
    start = min(max(0, column - len(label) // 2), width - len(label))
    span_start = max(0, start - 1)
    span_stop = min(width, start + len(label) + 1)
    if any(row[i] != " " for i in range(span_start, span_stop)):
        return
    for offset, char in enumerate(label):
        row[start + offset] = char


def render_error_bar(
    interval: Interval,
    point: float,
    threshold: float,
    width: int = _DEFAULT_WIDTH,
) -> str:
    """Draw the kappa confidence interval against the trust threshold.

    An interval entirely right of the threshold marker reads as trusted
    at a glance; entirely left reads as untrusted.
    """
    if width < 21:
        raise ValueError(f"width must be at least 21, got {width}")

    anchors = [interval.low, interval.high, point, threshold]
    axis_low = min(anchors)
    axis_high = max(anchors)
    span = max(axis_high - axis_low, 1e-9)
    padding = span * 0.12
    axis_low -= padding
    axis_high += padding

    bar = [" "] * width
    low_col = _column(interval.low, axis_low, axis_high, width)
    high_col = _column(interval.high, axis_low, axis_high, width)
    point_col = _column(point, axis_low, axis_high, width)
    for col in range(low_col, high_col + 1):
        bar[col] = "-"
    bar[low_col] = "["
    bar[high_col] = "]"
    bar[point_col] = "o"

    ruler = ["."] * width
    threshold_col = _column(threshold, axis_low, axis_high, width)
    ruler[threshold_col] = "|"

    labels = [" "] * width
    _place_label(labels, threshold_col, f"threshold ({threshold:.2f})", width)

    legend = (
        f"kappa {point:.3f}   "
        f"{interval.confidence:.0%} CI [{interval.low:.3f}, {interval.high:.3f}]"
    )
    return "\n".join(["".join(bar), "".join(ruler), "".join(labels).rstrip(), legend])
