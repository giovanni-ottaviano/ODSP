"""The voter model's checks (validation/models/voter.py), at smoke scale."""

from __future__ import annotations

from validation.models import MODELS

from .smoke import run_smoke

CHECKS = ["voter.complete_time", "voter.exit_probability", "voter.lattice_1d",
          "voter.lattice_2d", "voter.lattice_3d", "voter.ba_heterogeneous", "voter.rho_2d",
          "voter.rho_2d_exponent"]


def test_voter_smoke(tmp_path, odsp_bin):
    assert "voter" in MODELS
    rows = run_smoke(tmp_path, odsp_bin, "voter", CHECKS)
    predicted = {r["check"]: float(r["predicted"]) for r in rows}
    assert predicted.pop("voter.lattice_1d") == 2.0           # T ~ L^2
    # the exact rho(t)'s exponent over a finite window: above 1, approaching it as ln t grows
    assert 1.0 < predicted.pop("voter.rho_2d_exponent") < 1.2
    assert set(predicted.values()) == {1.0}                   # ratios, and exponents 1
    rho = next(r for r in rows if r["check"] == "voter.rho_2d")
    assert "T^2/dof (Hotelling)" in rho["warnings"]           # correlated-point consistency
