"""Cryptographically secure random sampling.

All randomness comes from the `secrets` module (the OS CSPRNG), never from
`random` or `numpy.random`.
"""
import math
import secrets
from bisect import bisect_left
from typing import Callable, Sequence


def uniform(a: float = 0.0, b: float = 1.0) -> float:
    """Cryptographically secure uniform sample."""
    u = secrets.randbits(53) / (1 << 53)  # in [0, 1)
    return a + (b - a) * u


def uniform_samples(n: int, a: float = 0.0, b: float = 1.0) -> list:
    """Return n independent secure uniform samples on [a, b)."""
    return [uniform(a, b) for _ in range(n)]


def inverse_transform_sample(inv_cdf: Callable[[float], float]) -> float:
    """One sample from any distribution whose inverse CDF is known."""
    return inv_cdf(uniform())


def inverse_transform_samples(inv_cdf: Callable[[float], float], n: int) -> list:
    """n independent samples via inverse transform sampling."""
    return [inverse_transform_sample(inv_cdf) for _ in range(n)]


def exponential(beta: float = 1.0) -> float:
    """Secure Exponential sample with mean beta."""
    return inverse_transform_sample(lambda y: -beta * math.log(1.0 - y))


def discrete(values: Sequence, cdf_values: Sequence[float]) -> float:
    """Secure sample from a discrete distribution via F^{-1}(y) = min{x: F(x) >= y}."""
    y = uniform()
    return values[bisect_left(cdf_values, y)]
