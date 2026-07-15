from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from judgegate.errors import AnalysisError

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

Weighting = Literal["none", "linear", "quadratic"]

_EPS = 1e-12


@dataclass(frozen=True)
class PerClassAgreement:
    """How the judge treats one label class relative to humans."""

    label: str
    human_count: int
    judge_count: int
    recall: float
    precision: float


@dataclass(frozen=True)
class AgreementResult:
    """Agreement statistics between judge labels and human labels."""

    labels: tuple[str, ...]
    matrix: IntArray
    n_items: int
    observed_agreement: float
    chance_agreement: float
    kappa: float
    weighting: Weighting
    per_class: tuple[PerClassAgreement, ...]


def confusion_matrix(
    human: Sequence[str], judge: Sequence[str], labels: Sequence[str]
) -> IntArray:
    """Count matrix with human labels on rows and judge labels on columns."""
    if len(human) != len(judge):
        raise AnalysisError(
            f"human and judge label counts differ: {len(human)} vs {len(judge)}"
        )
    if not human:
        raise AnalysisError("no labeled items to analyze")
    index = {label: i for i, label in enumerate(labels)}
    if len(index) != len(labels):
        raise AnalysisError("label set contains duplicates")
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for h, j in zip(human, judge, strict=True):
        if h not in index:
            raise AnalysisError(f"human label {h!r} is not in the label set {list(labels)}")
        if j not in index:
            raise AnalysisError(f"judge label {j!r} is not in the label set {list(labels)}")
        matrix[index[h], index[j]] += 1
    return matrix


def weight_matrix(n_labels: int, weighting: Weighting) -> FloatArray:
    """Disagreement weights: 0 on the diagonal, growing off it."""
    if n_labels < 2:
        raise AnalysisError("at least two label classes are required")
    if weighting == "none":
        return 1.0 - np.eye(n_labels, dtype=np.float64)
    idx = np.arange(n_labels, dtype=np.float64)
    distance = np.abs(idx[:, None] - idx[None, :]) / (n_labels - 1)
    return distance if weighting == "linear" else distance**2


def kappa_from_counts(matrix: npt.ArrayLike, weighting: Weighting = "none") -> float:
    """Cohen's kappa, optionally weighted, from a confusion count matrix."""
    counts = np.asarray(matrix, dtype=np.float64)
    return float(kappa_from_counts_batch(counts[None, :, :], weighting)[0])


def kappa_from_counts_batch(counts: FloatArray, weighting: Weighting = "none") -> FloatArray:
    """Vectorized kappa over a batch of confusion count matrices.

    The disagreement-weight formulation is used for every weighting, since
    unweighted kappa is the special case where every off-diagonal cell
    weighs 1. Degenerate matrices where chance disagreement is zero
    (all mass in one identical class) return kappa 1 when observed
    disagreement is also zero, else 0.
    """
    if counts.ndim != 3 or counts.shape[1] != counts.shape[2]:
        raise AnalysisError("expected a batch of square confusion matrices")
    n = counts.sum(axis=(1, 2))
    if np.any(n <= 0):
        raise AnalysisError("confusion matrix has no observations")
    proportions = counts / n[:, None, None]
    row = proportions.sum(axis=2)
    col = proportions.sum(axis=1)
    expected = row[:, :, None] * col[:, None, :]
    weights = weight_matrix(counts.shape[1], weighting)
    observed_dis = (proportions * weights).sum(axis=(1, 2))
    expected_dis = (expected * weights).sum(axis=(1, 2))
    result = np.where(
        expected_dis > _EPS,
        1.0 - observed_dis / np.maximum(expected_dis, _EPS),
        np.where(observed_dis <= _EPS, 1.0, 0.0),
    )
    return np.asarray(result, dtype=np.float64)


def kappa_standard_error_batch(counts: FloatArray) -> FloatArray:
    """Vectorized large-sample standard error of unweighted Cohen's kappa.

    Fleiss, Cohen, and Everitt (1969). Used inside power simulations
    where a bootstrap per replicate would be prohibitive; reported
    intervals use the bootstrap instead.
    """
    if counts.ndim != 3 or counts.shape[1] != counts.shape[2]:
        raise AnalysisError("expected a batch of square confusion matrices")
    n = counts.sum(axis=(1, 2))
    if np.any(n < 2):
        raise AnalysisError("standard error needs at least two observations")
    p = counts / n[:, None, None]
    row = p.sum(axis=2)
    col = p.sum(axis=1)
    po = np.trace(p, axis1=1, axis2=2)
    pe = np.sum(row * col, axis=1)
    safe = 1.0 - pe >= _EPS
    kappa = np.where(safe, (po - pe) / np.maximum(1.0 - pe, _EPS), 0.0)

    k = counts.shape[1]
    diag = p[:, np.arange(k), np.arange(k)]
    term_a = np.sum(diag * (1.0 - (row + col) * (1.0 - kappa[:, None])) ** 2, axis=1)
    cross = (col[:, :, None] + row[:, None, :]) ** 2
    off = p * cross
    off[:, np.arange(k), np.arange(k)] = 0.0
    term_b = (1.0 - kappa) ** 2 * np.sum(off, axis=(1, 2))
    term_c = (kappa - pe * (1.0 - kappa)) ** 2
    variance = (term_a + term_b - term_c) / (n * np.maximum(1.0 - pe, _EPS) ** 2)
    result = np.where(safe, np.sqrt(np.maximum(variance, 0.0)), 0.0)
    return np.asarray(result, dtype=np.float64)


def kappa_standard_error(matrix: npt.ArrayLike) -> float:
    """Large-sample standard error of unweighted Cohen's kappa."""
    counts = np.asarray(matrix, dtype=np.float64)
    return float(kappa_standard_error_batch(counts[None, :, :])[0])


def measure_agreement(
    human: Sequence[str],
    judge: Sequence[str],
    labels: Sequence[str],
    weighting: Weighting = "none",
) -> AgreementResult:
    """Full agreement summary between judge and human labels."""
    matrix = confusion_matrix(human, judge, labels)
    n = int(matrix.sum())
    proportions = matrix.astype(np.float64) / n
    row = proportions.sum(axis=1)
    col = proportions.sum(axis=0)
    observed = float(np.trace(proportions))
    chance = float(np.sum(row * col))
    kappa = kappa_from_counts(matrix, weighting)

    per_class = []
    for i, label in enumerate(labels):
        human_count = int(matrix[i, :].sum())
        judge_count = int(matrix[:, i].sum())
        recall = float(matrix[i, i] / human_count) if human_count else 0.0
        precision = float(matrix[i, i] / judge_count) if judge_count else 0.0
        per_class.append(
            PerClassAgreement(
                label=label,
                human_count=human_count,
                judge_count=judge_count,
                recall=recall,
                precision=precision,
            )
        )

    return AgreementResult(
        labels=tuple(labels),
        matrix=matrix,
        n_items=n,
        observed_agreement=observed,
        chance_agreement=chance,
        kappa=kappa,
        weighting=weighting,
        per_class=tuple(per_class),
    )
