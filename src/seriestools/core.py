# src/seriestools/core.py
import math


def reduce_angle(x: float) -> tuple[int, float]:
    """
    Reduces an angle to the primary interval [-pi/2, pi/2].

    Args:
        x: The input angle in radians.

    Returns:
        A tuple (q, r) where q is the integer quadrant index and r is the
        remainder angle in [-pi/2, pi/2], such that x = q * (pi/2) + r.
    """
    if not isinstance(x, (int, float)):
        raise TypeError("Input angle must be a real number.")

    pi_half = math.pi / 2.0

    # Calculate the quadrant index q
    q = int(round(x / pi_half))

    # Calculate the remainder r
    r = x - q * pi_half

    return q, r


# (在 core.py 文件顶部)
_TOLERANCE = 1e-12
_MAX_ITER = 100


def _sin_series(r: float) -> float:
    """Evaluates the Maclaurin series for sin(r)."""
    r_squared = r * r
    term = r
    partial_sum = term
    n = 1
    while abs(term) > _TOLERANCE * max(1.0, abs(partial_sum)) and n < _MAX_ITER:
        n += 2
        term = -term * r_squared / (n * (n - 1))
        partial_sum += term
    return partial_sum


def _cos_series(r: float) -> float:
    """Evaluates the Maclaurin series for cos(r)."""
    r_squared = r * r
    term = 1.0
    partial_sum = term
    n = 0
    while abs(term) > _TOLERANCE * max(1.0, abs(partial_sum)) and n < _MAX_ITER:
        n += 2
        term = -term * r_squared / (n * (n - 1))
        partial_sum += term
    return partial_sum


def sin(x: float) -> float:
    """Computes sin(x) using argument reduction and Taylor series."""
    q, r = reduce_angle(x)

    # Use quadrant logic to reconstruct the final value
    q_mod_4 = q % 4

    sin_r = _sin_series(r)
    cos_r = _cos_series(r)

    if q_mod_4 == 0:  # sin(r)
        return sin_r
    elif q_mod_4 == 1:  # cos(r)
        return cos_r
    elif q_mod_4 == 2:  # -sin(r)
        return -sin_r
    else:  # -cos(r)
        return -cos_r


def cos(x: float) -> float:
    """Computes cos(x) using argument reduction and Taylor series."""
    q, r = reduce_angle(x)

    # Use quadrant logic to reconstruct the final value
    q_mod_4 = q % 4

    sin_r = _sin_series(r)
    cos_r = _cos_series(r)

    if q_mod_4 == 0:  # cos(r)
        return cos_r
    elif q_mod_4 == 1:  # -sin(r)
        return -sin_r
    elif q_mod_4 == 2:  # -cos(r)
        return -cos_r
    else:  # sin(r)
        return sin_r


# src/seriestools/core.py
def exp(x: float) -> float:
    """Computes exp(x) using its Maclaurin series."""
    term = 1.0
    partial_sum = term
    n = 0
    while abs(term) > _TOLERANCE * max(1.0, abs(partial_sum)) and n < _MAX_ITER:
        n += 1
        term = term * x / n
        partial_sum += term
    return partial_sum


# src/seriestools/core.py
def tan(x: float) -> float:
    """
    Computes tan(x) via sin(x)/cos(x).

    Args:
        x: The input angle in radians.

    Returns:
        The value of tan(x).

    Raises:
        ValueError: If x is close to a pole of tan(x) (i.e., where cos(x) is near zero).
    """
    cos_val = cos(x)
    # Define a threshold for detecting poles
    pole_threshold = 1e-15
    if abs(cos_val) < pole_threshold:
        raise ValueError("Input x is near a pole of tan(x).")

    return sin(x) / cos_val


# src/seriestools/core.py
def sinc(x: float) -> float:
    """
    Computes the sinc function, sinc(x) = sin(x)/x.

    Uses a series expansion for small |x| to maintain numerical stability.
    """
    # Threshold to switch to series expansion to avoid cancellation
    series_threshold = 1e-6

    if abs(x) < series_threshold:
        if x == 0.0:
            return 1.0
        # Evaluate the series for sinc(x)
        x_squared = x * x
        term = 1.0
        partial_sum = term
        n = 1
        while abs(term) > _TOLERANCE * max(1.0, abs(partial_sum)) and n < _MAX_ITER:
            n += 2
            term = -term * x_squared / (n * (n - 1))
            partial_sum += term
        return partial_sum
    else:
        return sin(x) / x
