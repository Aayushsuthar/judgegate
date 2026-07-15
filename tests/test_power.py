import numpy as np
import pytest

from judgegate.errors import AnalysisError
from judgegate.stats.bootstrap import kappa_interval
from judgegate.stats.power import (
    detectable_kappa,
    joint_distribution,
    required_labels,
    simulate_decision_rate,
)


class TestJointDistribution:
    def test_marginals_and_kappa_are_recovered(self):
        marginals = [0.5, 0.3, 0.2]
        joint = joint_distribution(marginals, 0.65)
        assert joint.sum() == pytest.approx(1.0)
        assert joint.sum(axis=1) == pytest.approx(marginals)
        assert joint.sum(axis=0) == pytest.approx(marginals)
        po = float(np.trace(joint))
        pe = float(sum(m * m for m in marginals))
        assert (po - pe) / (1 - pe) == pytest.approx(0.65)

    def test_rejects_invalid_input(self):
        with pytest.raises(AnalysisError):
            joint_distribution([0.7, 0.4], 0.5)
        with pytest.raises(AnalysisError):
            joint_distribution([0.7, 0.3], 1.5)
        with pytest.raises(AnalysisError):
            joint_distribution([1.0], 0.5)


class TestDecisionRate:
    def test_rate_grows_with_sample_size(self):
        small = simulate_decision_rate([0.7, 0.3], 0.8, 0.6, 30, sims=800, seed=1)
        large = simulate_decision_rate([0.7, 0.3], 0.8, 0.6, 300, sims=800, seed=1)
        assert large > small

    def test_rate_near_zero_when_judge_is_at_threshold(self):
        rate = simulate_decision_rate([0.7, 0.3], 0.6, 0.6, 200, sims=800, seed=2)
        assert rate < 0.10


class TestRequiredLabels:
    def test_budget_certifies_at_target_power_against_real_gate(self):
        marginals = [0.65, 0.35]
        budget = required_labels(marginals, 0.8, 0.6, sims=1200, seed=3)
        assert budget.required is not None
        joint = joint_distribution(marginals, 0.8)
        certified = 0
        trials = 60
        for i in range(trials):
            matrix = np.random.default_rng(3000 + i).multinomial(
                budget.required, joint.flatten()
            ).reshape(2, 2)
            interval = kappa_interval(matrix, resamples=800, seed=4)
            if interval.low > 0.6:
                certified += 1
        assert certified / trials >= budget.power - 0.15

    def test_more_labels_needed_for_smaller_margins(self):
        wide = required_labels([0.7, 0.3], 0.85, 0.6, sims=800, seed=5)
        narrow = required_labels([0.7, 0.3], 0.7, 0.6, sims=800, seed=5)
        assert wide.required is not None
        assert narrow.required is not None
        assert narrow.required > wide.required

    def test_judge_below_threshold_cannot_be_certified(self):
        budget = required_labels([0.7, 0.3], 0.5, 0.6, sims=800, seed=6)
        assert budget.required is None

    def test_curve_is_recorded(self):
        budget = required_labels([0.7, 0.3], 0.85, 0.6, sims=600, seed=7)
        assert budget.curve
        assert all(n >= 10 for n, _ in budget.curve)


class TestDetectableKappa:
    def test_smaller_suites_demand_better_judges(self):
        with_50 = detectable_kappa(50, [0.7, 0.3], 0.6, sims=800, seed=8)
        with_500 = detectable_kappa(500, [0.7, 0.3], 0.6, sims=800, seed=8)
        assert with_50 is not None
        assert with_500 is not None
        assert with_50 > with_500

    def test_tiny_suite_certifies_nothing(self):
        result = detectable_kappa(10, [0.7, 0.3], 0.85, sims=600, seed=9)
        assert result is None or result > 0.9
