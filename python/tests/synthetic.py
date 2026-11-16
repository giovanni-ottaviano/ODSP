"""Synthetic size curves shared by the crossing, collapse and evaluation-figure tests."""
import numpy as np

from odspy import SizeCurve

XC, NU, A = 2.27, 1.0, 0.25
SIGMA = 0.01  # noise on the scaled ordinate Y = y L^(-a)
SIZES = (8, 16, 32)


def curve(size, x, y, yerr=0.002):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if yerr is not None:
        yerr = np.full(len(x), yerr) if np.isscalar(yerr) else np.asarray(yerr, float)
    return SizeCurve(size, "L", x, y, yerr, "temperature", "m", "test")


def scaling_curves(seed=3, sizes=SIZES):
    """y = L^a tanh((x - xc) L^(1/nu)) with known Gaussian noise of sd SIGMA L^a."""
    rng = np.random.default_rng(seed)
    x = XC + np.linspace(-0.08, 0.08, 33)
    out = []
    for L in sizes:
        f = np.tanh((x - XC) * L ** (1 / NU)) + rng.normal(0, SIGMA, len(x))
        out.append(curve(L, x, L ** A * f, L ** A * SIGMA))
    return out


def crossing_model(L, x):
    """Straight lines through (2.27, 0.6) with slope growing with L: every pair crosses at 2.27."""
    return 0.6 + 0.05 * (x - 2.27) * L
