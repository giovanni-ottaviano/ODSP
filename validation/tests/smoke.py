"""The smoke test every model task runs: the model's smoke-scale campaign, end to end."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from validation import campaign
from validation.core import SCALES


def run_smoke(root: Path, odsp: Path, model: str, checks: list[str]) -> list[dict[str, str]]:
    """Run `campaign run --scale smoke --models <model>` into `root` and assert that it
    succeeded, that summary.csv holds exactly the listed checks of the model with finite
    measured and error, and that each check's figure exists as PDF and PNG. Returns the
    model's summary rows."""
    assert "smoke" in SCALES
    code = campaign.main(["run", "--scale", "smoke", "--models", model, "--root", str(root),
                          "--odsp", str(odsp)])
    scale_root = Path(root) / "smoke"
    log = (scale_root / "log.txt").read_text() if (scale_root / "log.txt").exists() else ""
    assert code == 0, f"campaign run failed:\n{log}"
    with open(scale_root / "summary.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["model"] == model]
    found = sorted(r["check"] for r in rows)
    assert found == sorted(checks), f"checks in summary.csv: {found}"
    for r in rows:
        assert math.isfinite(float(r["measured"])), r
        assert math.isfinite(float(r["error"])) and float(r["error"]) > 0, r
        assert r["figure"], f"{r['check']}: no figure"
        for ext in ("pdf", "png"):
            assert (scale_root / "figures" / f"{r['figure']}.{ext}").exists(), (r["check"], ext)
    return rows
