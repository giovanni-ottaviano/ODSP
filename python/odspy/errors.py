"""Exceptions raised by odspy, and checks of non-data arguments that raise them."""

import numbers

import numpy as np


class InputError(Exception):
    """Malformed or missing input; the message names the file (and column)."""


def rng_from(seed, name: str = "seed") -> np.random.Generator:
    """A NumPy generator from a non-negative integer seed; anything else is an InputError."""
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise InputError(f"{name} must be a non-negative integer, got {seed!r}")
    return np.random.default_rng(int(seed))


def check_threshold(p, name: str = "p_threshold") -> float:
    """A significance threshold: a number strictly between 0 and 1."""
    if isinstance(p, bool) or not isinstance(p, numbers.Real) or not 0 < p < 1:
        raise InputError(f"{name} must be a number between 0 and 1, got {p!r}")
    return float(p)
