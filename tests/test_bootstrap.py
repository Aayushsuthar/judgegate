import numpy as np
import pytest

from judgegate.errors import AnalysisError
from judgegate.stats.bootstrap import kappa_interval
from judgegate.stats.power import joint_distribution


class TestKappaInterval:
    def test_coverage_matches_nominal_level(self):
        joint = joint_distribution([0.65, 0.35], 0.7)
        covered = 0
        simulations = 200
        for i in range(simulations):
            matrix = np.random.default_rng(1000 + i).multinomial(
                150, joint.flatten()
            ).reshape(2, 2)
            interval = kappa_interval(matrix, resamples=1200, seed=5)
            if interval.contains(0.7):
                covered += 1
        assert 0.90 <= covered / simulations <= 0.99

    def test_multiclass_coverage(self):
        joint = joint_distribution([0.5, 0.3, 0.2], 0.6)
        covered = 0
        simulations = 120
        for i in range(simulations):
            matrix = np.random.default_rng(2000 + i).multinomial(
                200, joint.flatten()
            ).reshape(3, 3)
            interval = kappa_interval(matrix, resamples=1200, seed=5)
            if interval.contains(0.6):
                covered += 1
        assert 0.88 <= covered / simulations <= 1.0

    def test_interval_narrows_with_more_labels(self):
        joint = joint_distribution([0.6, 0.4], 0.75)
        small = np.random.default_rng(1).multinomial(60, joint.flatten()).reshape(2, 2)
        large = np.random.default_rng(1).multinomial(2000, joint.flatten()).reshape(2, 2)
        wide = kappa_interval(small, resamples=1500, seed=3)
        narrow = kappa_interval(large, resamples=1500, seed=3)
        assert narrow.width < wide.width

    def test_deterministic_with_seed(self):
        matrix = np.array([[40, 8], [6, 46]])
        a = kappa_interval(matrix, resamples=800, seed=9)
        b = kappa_interval(matrix, resamples=800, seed=9)
        assert a == b

    def test_perfect_agreement_degenerates_gracefully(self):
        interval = kappa_interval(np.diag([25, 25]), resamples=500, seed=1)
        assert interval.high == pytest.approx(1.0)
        assert interval.contains(1.0)

    def test_rejects_bad_input(self):
        with pytest.raises(AnalysisError):
            kappa_interval(np.array([[1, 0], [0, 0]]))
        with pytest.raises(AnalysisError):
            kappa_interval(np.array([[5, 5], [5, 5]]), resamples=10)
        with pytest.raises(AnalysisError):
            kappa_interval(np.array([1, 2, 3]))
