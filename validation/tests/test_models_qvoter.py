"""The q-voter model's checks (validation/models/qvoter.py), at smoke scale."""

from __future__ import annotations

import pytest

from validation.models import MODELS
from validation.theory import qvoter_pc

from .smoke import run_smoke


def test_qvoter_smoke(tmp_path, odsp_bin):
    assert "qvoter" in MODELS
    rows = run_smoke(tmp_path, odsp_bin, "qvoter",
                     ["qvoter.exact_finite_n", "qvoter.pc_q3", "qvoter.pc_q4"])
    predicted = {r["check"]: float(r["predicted"]) for r in rows}
    assert predicted["qvoter.exact_finite_n"] == 1.0
    assert predicted["qvoter.pc_q3"] == pytest.approx(qvoter_pc(3), abs=1e-12)   # 1/3
    assert predicted["qvoter.pc_q4"] == pytest.approx(qvoter_pc(4), abs=1e-12)   # 3/11


def test_exact_crossing_drifts_to_mean_field():
    """The exact chain's Binder crossing of N and 2N lies below p_c by ~0.6/N and halves when
    N doubles (the finite-N offset the pc checks report)."""
    from validation.models.qvoter import exact_crossing

    grid = [round(0.30 + 0.0025 * i, 6) for i in range(21)]
    off = [exact_crossing(N, 2 * N, 3, grid) - qvoter_pc(3) for N in (200, 400, 800)]
    assert all(-1.5 / N < o < -0.3 / N for o, N in zip(off, (200, 400, 800)))
    assert off[0] < off[1] < off[2] < 0
