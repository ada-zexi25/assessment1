# tests/test_core.py
import pytest
import numpy as np
from seriestools import core

# Parameterized test for accuracy against NumPy over a grid of values
@pytest.mark.parametrize("x", np.linspace(-4 * np.pi, 4 * np.pi, 200))
def test_trig_accuracy_numpy(x):
    """Test sin and cos functions against NumPy for a range of values."""
    assert core.sin(x) == pytest.approx(np.sin(x))
    assert core.cos(x) == pytest.approx(np.cos(x))
    assert core.exp(x) == pytest.approx(np.exp(x))
    # Avoid testing tan near its poles
    if abs(np.cos(x)) > 1e-10:
        assert core.tan(x) == pytest.approx(np.tan(x))

# tests/test_core.py

def test_sinc_near_zero():
    """Test the sinc function at and near x=0 for stability."""
    assert core.sinc(0.0) == 1.0
    # For a very small x, the value should be slightly less than 1
    small_x = 1e-7
    expected_sinc = 1.0 - (small_x**2) / 6.0 # From series 1 - x^2/3!
    assert core.sinc(small_x) == pytest.approx(expected_sinc)

def test_tan_poles():
    """Test that tan raises ValueError near its poles."""
    with pytest.raises(ValueError, match="near a pole"):
        core.tan(np.pi / 2.0)
    with pytest.raises(ValueError, match="near a pole"):
        core.tan(-3 * np.pi / 2.0)

# tests/test_core.py

@pytest.mark.parametrize("x", np.random.uniform(-10, 10, 20))
def test_pythagorean_identity_property(x):
    """Property test: sin(x)^2 + cos(x)^2 should be approximately 1."""
    sin_x = core.sin(x)
    cos_x = core.cos(x)
    assert sin_x**2 + cos_x**2 == pytest.approx(1.0)

@pytest.mark.parametrize("x", np.random.uniform(-10, 10, 20))
def test_sin_is_odd_property(x):
    """Property test: sin(-x) should be equal to -sin(x)."""
    assert core.sin(-x) == pytest.approx(-core.sin(x))