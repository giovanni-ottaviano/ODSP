"""The rejection-free voter model's check (validation/models/rfvoter.py), at smoke scale."""

from __future__ import annotations

from validation.models import MODELS

from .smoke import run_smoke


def test_rfvoter_smoke(tmp_path, odsp_bin):
    assert "rfvoter" in MODELS
    (row,) = run_smoke(tmp_path, odsp_bin, "rfvoter", ["rfvoter.same_statistics"])
    assert float(row["predicted"]) == 1.0
