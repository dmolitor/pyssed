import numpy as np
import pyssed.utils as utils
import pytest


# Test that ATE calculation is correct
def test_ate():
    ites = [0.1, 0.9, 0.6, 0.4, 0.0012, 0.9988]
    assert utils.ate(ites) == 0.5


# Test that the shrinkage rate validation is correct
class TestShrinkage:

    def test_invalid(self):
        with pytest.raises(AssertionError) as excinfo:
            utils.check_shrinkage_rate(t=100, delta_t=1 / (100 ** (0.25)))
        assert excinfo.type is AssertionError

    def test_valid(self):
        assert utils.check_shrinkage_rate(t=100, delta_t=1 / (100 ** (0.2499))) is None


# Test that the confidence sequence calculation is correct
class TestConfidenceSequence:

    t = 10
    truth = 0.5541954577819679
    var1 = [0.0, 0.0, 0.5, 0.7, 1.2, 0.1, 0.1, 0.2, 0.1, 0.1]
    var2 = [0.5, 0.7, 1.2, 0.1, 0.1, 0.2, 0.1, 0.1]

    def test_cs_radius(self):
        assert np.isclose(
            utils.cs_radius(var=self.var1, t=self.t, t_star=self.t, alpha=0.05),
            self.truth,
            rtol=1e-15,
        )

    def test_cs_comparison(self):
        assert np.isclose(
            utils.cs_radius(var=self.var1, t=self.t, t_star=self.t, alpha=0.05),
            utils.cs_radius(var=self.var2, t=self.t, t_star=self.t, alpha=0.05),
            rtol=1e-15,
        )


# Test that the ITE calculation is correct
class TestITE:

    def test_ite_control(self):
        assert utils.ite(2.0, 0, 0.25) == -8

    def test_ite_treatment(self):
        assert utils.ite(2.0, 1, 0.5) == 4


# Test that the 'last' function works correctly
class TestLast:

    def test_last_empty(self):
        assert utils.last([]) is None

    def test_last_nonempty(self):
        assert utils.last([1, 2, 3]) == 3


# Test that the estimated variance calculation is correct
def test_var():
    assert np.isclose(utils.var(outcome=0.75, propensity=0.5), 2.25, rtol=1e-15)
