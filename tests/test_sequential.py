import numpy as np
import pytest

from judgegate.stats.sequential import MixtureSPRT, agreement_threshold


class TestAgreementThreshold:
    def test_interpolates_between_chance_and_one(self):
        assert agreement_threshold(0.5, 0.0) == pytest.approx(0.5)
        assert agreement_threshold(0.5, 1.0) == pytest.approx(1.0)
        assert agreement_threshold(0.5, 0.6) == pytest.approx(0.8)

    def test_rejects_bad_chance(self):
        with pytest.raises(ValueError):
            agreement_threshold(1.0, 0.5)


class TestMixtureSPRT:
    def test_type_one_error_is_controlled_under_optional_stopping(self):
        rng = np.random.default_rng(31)
        rejections = 0
        simulations = 150
        for _ in range(simulations):
            sprt = MixtureSPRT(theta0=0.8, alpha=0.05, min_samples=10)
            for _ in range(8):
                batch = (rng.uniform(size=25) < 0.8).astype(float)
                decision = sprt.update(batch)
                if decision.decided:
                    rejections += 1
                    break
        assert rejections / simulations <= 0.10

    def test_detects_agreement_above_threshold(self):
        rng = np.random.default_rng(32)
        sprt = MixtureSPRT(theta0=0.7, alpha=0.05, min_samples=10)
        decided = False
        for _ in range(20):
            batch = (rng.uniform(size=25) < 0.95).astype(float)
            decision = sprt.update(batch)
            if decision.decided:
                decided = True
                assert decision.direction == 1
                break
        assert decided

    def test_detects_agreement_below_threshold(self):
        rng = np.random.default_rng(33)
        sprt = MixtureSPRT(theta0=0.8, alpha=0.05, min_samples=10)
        decided = False
        for _ in range(20):
            batch = (rng.uniform(size=25) < 0.5).astype(float)
            decision = sprt.update(batch)
            if decision.decided:
                decided = True
                assert decision.direction == -1
                break
        assert decided

    def test_identical_early_values_do_not_decide_instantly(self):
        sprt = MixtureSPRT(theta0=0.8, alpha=0.05, min_samples=10)
        decision = sprt.update(np.ones(10))
        assert not decision.decided
        assert decision.p_value == 1.0

    def test_recovers_after_degenerate_first_window(self):
        rng = np.random.default_rng(35)
        sprt = MixtureSPRT(theta0=0.5, alpha=0.05, min_samples=10)
        first = sprt.update(np.ones(10))
        assert not first.decided
        decision = sprt.update((rng.uniform(size=200) < 0.95).astype(float))
        assert decision.decided
        assert decision.direction == 1

    def test_pvalue_is_monotone_nonincreasing(self):
        rng = np.random.default_rng(34)
        sprt = MixtureSPRT(theta0=0.5, alpha=0.01, min_samples=5)
        previous = 1.0
        for _ in range(10):
            decision = sprt.update((rng.uniform(size=10) < 0.6).astype(float))
            assert decision.p_value <= previous + 1e-12
            previous = decision.p_value

    def test_rejects_invalid_parameters(self):
        with pytest.raises(ValueError):
            MixtureSPRT(alpha=0.0)
        with pytest.raises(ValueError):
            MixtureSPRT(tau=-1.0)
        with pytest.raises(ValueError):
            MixtureSPRT(min_samples=1)
        with pytest.raises(ValueError):
            MixtureSPRT().update([1.0, float("nan")])
