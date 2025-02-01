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
            utils.check_shrinkage_rate(t=100, delta_t=1/(100**(0.25)))
        assert excinfo.type is AssertionError

    def test_valid(self):
        assert utils.check_shrinkage_rate(t=100, delta_t=1/(100**(0.2499))) is None

# Test that the confidence sequence calculation is correct
def test_cs_radius():
    truth = 0.5541954577819679
    t = 10
    eta = 0.8908509752892212
    var1 = [0., 0., 0.5, 0.7, 1.2, 0.1, 0.1, 0.2, 0.1, 0.1]
    assert np.isclose(utils.cs_radius(var=var1, t=t, t_star=t, alpha=0.05), truth, rtol=0.00001)
