import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks" / "results"
LEGACY = ROOT / "python" / "tests" / "fixtures" / "analyse_legacy.py"
NEW = ROOT / "benchmarks" / "analyse.py"

pytestmark = pytest.mark.skipif(not RESULTS.is_dir(), reason="benchmarks/results/ not present")


def run(script):
    p = subprocess.run([sys.executable, str(script), str(RESULTS)], capture_output=True, text=True,
                       timeout=280)
    return p.returncode, p.stdout.splitlines()


def verdicts(lines):
    return [l for l in lines if l.startswith(("  [PASS]", "  [FAIL]"))]


def summary(lines):
    return [l for l in lines if l.endswith(" checks passed")]


def test_same_verdicts_and_values():
    rc0, old = run(LEGACY)
    rc1, new = run(NEW)
    assert rc0 == rc1
    assert verdicts(old) == verdicts(new) and verdicts(old)
    assert summary(old) == summary(new) and len(summary(old)) == 1


def test_info_lines_present():
    _, new = run(NEW)
    info = [l for l in new if l.startswith("    info:")]
    assert info
    sections = {"B": "ising_L16_sweep.csv", "C": "potts3_L16_sweep.csv", "D": "mv_L16_sweep.csv",
                "E": "qvoter3_N500_sweep.csv"}
    text = "\n".join(new)
    for sec, f in sections.items():
        if (RESULTS / f).exists():
            start = next(i for i, l in enumerate(new) if l.startswith(f"{sec}. "))
            end = next((i for i in range(start + 1, len(new)) if new[i].startswith("=" * 78)
                        and i > start + 2), len(new))
            block = new[start:end]
            assert any("info:" in l and "crossing" in l for l in block), sec
    if (RESULTS / "coarsen_ising_timeseries.csv").exists():
        assert "errors not trustworthy: correlated points" in text
