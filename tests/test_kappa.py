import numpy as np
import pytest

from judgegate.errors import AnalysisError
from judgegate.stats.kappa import (
    confusion_matrix,
    kappa_from_counts,
    kappa_standard_error,
    measure_agreement,
    weight_matrix,
)


class TestConfusionMatrix:
    def test_counts_land_in_the_right_cells(self):
        matrix = confusion_matrix(
            ["fail", "pass", "pass", "fail"],
            ["fail", "pass", "fail", "pass"],
            ["fail", "pass"],
        )
        assert matrix.tolist() == [[1, 1], [1, 1]]

    def test_unknown_labels_are_rejected(self):
        with pytest.raises(AnalysisError, match="human label"):
            confusion_matrix(["maybe"], ["pass"], ["fail", "pass"])
        with pytest.raises(AnalysisError, match="judge label"):
            confusion_matrix(["pass"], ["maybe"], ["fail", "pass"])

    def test_length_mismatch_and_empty_are_rejected(self):
        with pytest.raises(AnalysisError, match="differ"):
            confusion_matrix(["pass"], [], ["fail", "pass"])
        with pytest.raises(AnalysisError, match="no labeled items"):
            confusion_matrix([], [], ["fail", "pass"])


class TestKappa:
    def test_perfect_agreement_is_one(self):
        assert kappa_from_counts(np.diag([30, 70])) == pytest.approx(1.0)

    def test_known_textbook_value(self):
        matrix = np.array([[20, 5], [10, 15]])
        po = 35 / 50
        pe = (25 / 50) * (30 / 50) + (25 / 50) * (20 / 50)
        expected = (po - pe) / (1 - pe)
        assert kappa_from_counts(matrix) == pytest.approx(expected)

    def test_independence_is_near_zero(self):
        rng = np.random.default_rng(3)
        joint = np.outer([0.6, 0.4], [0.6, 0.4])
        matrix = rng.multinomial(5000, joint.flatten()).reshape(2, 2)
        assert abs(kappa_from_counts(matrix)) < 0.05

    def test_systematic_disagreement_is_negative(self):
        matrix = np.array([[0, 50], [50, 0]])
        assert kappa_from_counts(matrix) < -0.9

    def test_single_identical_class_degenerates_to_one(self):
        matrix = np.array([[40, 0], [0, 0]])
        assert kappa_from_counts(matrix) == pytest.approx(1.0)

    def test_weighted_kappa_forgives_near_misses(self):
        near = np.array([[10, 5, 0], [5, 10, 5], [0, 5, 10]])
        far = np.array([[10, 0, 5], [5, 10, 5], [5, 0, 10]])
        assert kappa_from_counts(near, "quadratic") > kappa_from_counts(far, "quadratic")

    def test_weight_matrix_shapes(self):
        w = weight_matrix(3, "linear")
        assert w[0, 0] == 0.0
        assert w[0, 2] == pytest.approx(1.0)
        assert w[0, 1] == pytest.approx(0.5)
        q = weight_matrix(3, "quadratic")
        assert q[0, 1] == pytest.approx(0.25)


class TestStandardError:
    def test_matches_bootstrap_spread_on_large_samples(self):
        rng = np.random.default_rng(11)
        joint = 0.7 * np.diag([0.6, 0.4]) + 0.3 * np.outer([0.6, 0.4], [0.6, 0.4])
        matrix = rng.multinomial(2000, joint.flatten()).reshape(2, 2)
        se = kappa_standard_error(matrix)
        boots = []
        p = matrix.flatten() / matrix.sum()
        for i in range(300):
            resample = np.random.default_rng(i).multinomial(2000, p).reshape(2, 2)
            boots.append(kappa_from_counts(resample))
        assert se == pytest.approx(float(np.std(boots)), rel=0.35)

    def test_needs_two_observations(self):
        with pytest.raises(AnalysisError):
            kappa_standard_error(np.array([[1, 0], [0, 0]]))


class TestMeasureAgreement:
    def test_full_summary(self):
        result = measure_agreement(
            ["pass", "pass", "fail", "fail"],
            ["pass", "fail", "fail", "fail"],
            ["fail", "pass"],
        )
        assert result.n_items == 4
        assert result.observed_agreement == pytest.approx(0.75)
        assert result.per_class[0].label == "fail"
        assert result.per_class[0].recall == pytest.approx(1.0)
        assert result.per_class[1].recall == pytest.approx(0.5)
        assert result.per_class[1].precision == pytest.approx(1.0)
