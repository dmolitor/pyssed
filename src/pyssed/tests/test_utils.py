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