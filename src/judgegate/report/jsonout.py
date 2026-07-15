import json
from typing import Any

from judgegate.__about__ import __version__
from judgegate.gate import GateReport
from judgegate.probes.base import ProbeResult
from judgegate.stats.intervals import Interval


def _interval_to_dict(interval: Interval | None) -> dict[str, float] | None:
    if interval is None:
        return None
    return {
        "low": interval.low,
        "high": interval.high,
        "confidence": interval.confidence,
    }


def _probe_to_dict(probe: ProbeResult) -> dict[str, Any]:
    return {
        "name": probe.name,
        "verdict": probe.verdict,
        "applicable": probe.applicable,
        "passed": probe.passed,
        "effect": probe.effect,
        "effect_label": probe.effect_label,
        "tolerance": probe.tolerance,
        "interval": _interval_to_dict(probe.interval),
        "detail": probe.detail,
    }


def report_to_dict(report: GateReport) -> dict[str, Any]:
    """Convert a trust report to a JSON serializable dictionary."""
    agreement = report.agreement
    return {
        "judgegate_version": __version__,
        "verdict": report.verdict.value,
        "exit_code": report.exit_code,
        "mode": report.mode,
        "kappa": agreement.kappa,
        "weighting": agreement.weighting,
        "interval": _interval_to_dict(report.interval),
        "threshold": report.threshold,
        "alpha": report.alpha,
        "observed_agreement": agreement.observed_agreement,
        "chance_agreement": agreement.chance_agreement,
        "n_labels": report.n_labels,
        "labels_needed": report.labels_needed,
        "confusion_matrix": agreement.matrix.tolist(),
        "labels": list(agreement.labels),
        "per_class": [
            {
                "label": row.label,
                "human_count": row.human_count,
                "judge_count": row.judge_count,
                "recall": row.recall,
                "precision": row.precision,
            }
            for row in agreement.per_class
        ],
        "probes": [_probe_to_dict(p) for p in report.probes],
        "notes": list(report.notes),
    }


def render_json(report: GateReport) -> str:
    """Render the trust report as pretty printed JSON."""
    return json.dumps(report_to_dict(report), indent=2)
