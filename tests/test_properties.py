import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from judgegate.stats.intervals import wilson_proportion_interval
from judgegate.stats.kappa import kappa_from_counts
from judgegate.stats.power import joint_distribution

cell_counts = st.lists(
    st.integers(min_value=0, max_value=200), min_size=4, max_size=4
).filter(lambda cells: (sum(cells) >= 2 and (cells[0] + cells[3]) != sum(cells)) or True)


class TestKappaProperties:
    @given(cells=st.lists(st.integers(min_value=0, max_value=200), min_size=4, max_size=4))
    @settings(max_examples=300)
    def test_kappa_is_bounded(self, cells):
        if sum(cells) < 1:
            return
        matrix = np.asarray(cells).reshape(2, 2)
        kappa = kappa_from_counts(matrix)
        assert -1.0 - 1e-9 <= kappa <= 1.0 + 1e-9

    @given(
        diag=st.lists(st.integers(min_value=1, max_value=200), min_size=2, max_size=4)
    )
    @settings(max_examples=100)
    def test_perfect_agreement_is_always_one(self, diag):
        assert kappa_from_counts(np.diag(diag)) == 1.0


class TestJointProperties:
    @given(
        first=st.floats(min_value=0.05, max_value=0.95),
        kappa=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=200)
    def test_construction_is_a_valid_distribution(self, first, kappa):
        joint = joint_distribution([first, 1.0 - first], kappa)
        assert np.all(joint >= -1e-12)
        assert joint.sum() == np.float64(1.0) or abs(joint.sum() - 1.0) < 1e-9
        assert abs(joint.sum(axis=1)[0] - first) < 1e-9


class TestWilsonProperties:
    @given(
        total=st.integers(min_value=1, max_value=10_000),
        data=st.data(),
        confidence=st.floats(min_value=0.5, max_value=0.999),
    )
    @settings(max_examples=200)
    def test_bounds_bracket_the_estimate(self, total, data, confidence):
        successes = data.draw(st.integers(min_value=0, max_value=total))
        p_hat = successes / total
        interval = wilson_proportion_interval(p_hat, total, confidence)
        assert 0.0 <= interval.low <= p_hat <= interval.high <= 1.0
