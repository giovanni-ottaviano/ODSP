"""The noisy voter model's check (validation/models/noisy.py), at smoke scale."""

from __future__ import annotations

from validation.models import MODELS

from .smoke import run_smoke


def test_noisy_smoke(tmp_path, odsp_bin):
    assert "noisy" in MODELS
    (row,) = run_smoke(tmp_path, odsp_bin, "noisy", ["noisy.stationary"])
    assert float(row["predicted"]) == 1.0          # <m^2>_sim / <m^2>_exact


def test_bin_index_one_lattice_value_per_bin():
    """With --bins N + 1 over [-1, 1] each bin holds exactly one m = (2n - N)/N, and the
    driver's floor rule (upper edge in the last bin) is reproduced."""
    import numpy as np

    from validation.models.noisy import bin_index

    for N in (32, 100, 256):
        assert list(bin_index(N, -1.0, 1.0, N + 1)) == list(range(N + 1))
    # a range that does not reach +-1: values outside are clamped to the end bins
    idx = bin_index(10, -0.6, 0.6, 3)
    assert idx[0] == 0 and idx[-1] == 2 and np.all(np.diff(idx) >= 0)
