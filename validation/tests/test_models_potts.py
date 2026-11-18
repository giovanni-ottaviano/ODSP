"""The 2D 3-state Potts model's checks (validation/models/potts.py), at smoke scale."""

from __future__ import annotations

import pytest

from validation.models import MODELS
from validation.theory import POTTS3_EXPONENTS, POTTS3_TC

from .smoke import run_smoke


def test_potts_smoke(tmp_path, odsp_bin):
    assert "potts" in MODELS
    rows = run_smoke(tmp_path, odsp_bin, "potts",
                     ["potts.tc", "potts.inv_nu", "potts.gamma_nu", "potts.beta_nu"])
    predicted = {r["check"]: float(r["predicted"]) for r in rows}
    assert predicted["potts.tc"] == pytest.approx(POTTS3_TC, abs=1e-12)
    assert predicted["potts.inv_nu"] == pytest.approx(1.0 / POTTS3_EXPONENTS["nu"])   # 6/5
    assert predicted["potts.gamma_nu"] == pytest.approx(26.0 / 15.0)
    assert predicted["potts.beta_nu"] == pytest.approx(-2.0 / 15.0)
    from validation.models._critical import FIT_RULE

    by = {r["check"]: r for r in rows}
    for name in ("potts.inv_nu", "potts.gamma_nu", "potts.beta_nu"):
        assert "(fit L >= 12)" in by[name]["sizes"], by[name]["sizes"]
        assert FIT_RULE in by[name]["warnings"]
        assert "finite-size diagnostic: fit over all sizes (L >= 8)" in by[name]["warnings"]
