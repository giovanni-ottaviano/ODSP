"""The nonlinear voter model's checks (validation/models/nonlinear.py), at smoke scale."""

from __future__ import annotations

from validation.models import MODELS

from .smoke import run_smoke


def test_nonlinear_smoke(tmp_path, odsp_bin):
    assert "nonlinear" in MODELS
    rows = run_smoke(tmp_path, odsp_bin, "nonlinear", ["nonlinear.exit", "nonlinear.time"])
    assert {float(r["predicted"]) for r in rows} == {1.0}     # ratios to the exact chain
