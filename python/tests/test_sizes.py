import numpy as np
import pytest

from odspy.errors import InputError
from odspy.sizes import SizeCurve, families, size_of


def write_sweep(tmp_path, name, preamble, key="temperature", rows=None, model="ising"):
    rows = rows if rows is not None else [(2.1, 0.60, 0.01), (2.2, 0.55, 0.02)]
    lines = [f"#   model = {model}"] + [f"#   {k} = {v}" for k, v in preamble.items()]
    lines.append(f"{key},binder,binder_err")
    lines += [",".join(str(c) for c in r) for r in rows]
    prefix = tmp_path / name
    (tmp_path / f"{name}_sweep.csv").write_text("\n".join(lines) + "\n")
    return str(prefix)


def test_sweep2_families(outputs):
    curves = families([outputs["stat2"]], "binder")
    assert [c.size for c in curves] == [8, 12, 16]
    assert all(c.label == "L" and c.xkey == "temperature" for c in curves)
    assert all(isinstance(c, SizeCurve) and c.column == "binder" for c in curves)
    assert all(c.x.shape == c.y.shape == c.yerr.shape and c.x.size == 7 for c in curves)
    assert curves[0].x[0] == pytest.approx(2.1)


def test_separate_prefixes(tmp_path):
    ps = [write_sweep(tmp_path, f"r{L}", {"dims": f"{L}x{L}"}) for L in (16, 8, 12)]
    curves = families(ps, "binder")
    assert [c.size for c in curves] == [8, 12, 16]
    assert curves[0].label == "L" and curves[0].xkey == "temperature"
    assert curves[0].yerr is not None
    assert str(tmp_path / "r8") in curves[0].source


def test_no_err_column(tmp_path):
    p = write_sweep(tmp_path, "a", {"dims": "8x8"})
    (tmp_path / "a_sweep.csv").write_text("#   dims = 8x8\ntemperature,binder\n2.1,0.5\n")
    assert families([p], "binder")[0].yerr is None


def test_mismatched_models(tmp_path):
    a = write_sweep(tmp_path, "alpha_run", {"dims": "8x8"}, model="ising")
    b = write_sweep(tmp_path, "beta_run", {"dims": "12x12"}, model="potts")
    with pytest.raises(InputError) as e:
        families([a, b], "binder")
    msg = str(e.value)
    assert "alpha_run" in msg and "beta_run" in msg and "ising" in msg and "potts" in msg


def test_different_swept_keys(tmp_path):
    a = write_sweep(tmp_path, "alpha_run", {"dims": "8x8"}, key="temperature")
    b = write_sweep(tmp_path, "beta_run", {"dims": "12x12"}, key="noise")
    with pytest.raises(InputError) as e:
        families([a, b], "binder")
    msg = str(e.value)
    assert "alpha_run" in msg and "beta_run" in msg
    assert "temperature" in msg and "noise" in msg


def test_duplicate_sizes(tmp_path):
    a = write_sweep(tmp_path, "alpha_run", {"dims": "8x8"})
    b = write_sweep(tmp_path, "beta_run", {"dims": "8x8"})
    with pytest.raises(InputError) as e:
        families([a, b], "binder")
    msg = str(e.value)
    assert "alpha_run" in msg and "beta_run" in msg and "8" in msg


def test_mixed_dims_and_n(tmp_path):
    a = write_sweep(tmp_path, "alpha_run", {"dims": "8x8"})
    b = write_sweep(tmp_path, "beta_run", {"graph": "complete", "n": "100"})
    with pytest.raises(InputError) as e:
        families([a, b], "binder")
    msg = str(e.value)
    assert "alpha_run" in msg and "beta_run" in msg and "L" in msg and "N" in msg
    curves = families([a, b], "binder", size="N")
    assert [c.size for c in curves] == [64, 100] and curves[0].label == "N"


def test_noncubic_needs_size(tmp_path):
    a = write_sweep(tmp_path, "alpha_run", {"dims": "8x16"})
    b = write_sweep(tmp_path, "beta_run", {"dims": "12x12"})
    with pytest.raises(InputError) as e:
        families([a, b], "binder")
    msg = str(e.value)
    assert "alpha_run" in msg and "8x16" in msg and "beta_run" not in msg


@pytest.mark.parametrize("bad", ["nan", "inf", "0", "-5", "12.5", "abc"])
def test_bad_n(tmp_path, bad):
    a = write_sweep(tmp_path, "alpha_run", {"graph": "complete", "n": bad})
    with pytest.raises(InputError) as e:
        families([a], "binder")
    assert "alpha_run" in str(e.value) and bad in str(e.value)
    with pytest.raises(InputError, match="n"):
        size_of({"n": bad}, where="x")


def test_noncubic_with_size_N(tmp_path):
    a = write_sweep(tmp_path, "nca", {"dims": "8x16"})
    b = write_sweep(tmp_path, "ncb", {"dims": "12x12"})
    curves = families([a, b], "binder", size="N")
    assert [c.size for c in curves] == [128, 144]
    assert size_of({"dims": "8x16"}, "N") == (128, "N")


def test_size_of():
    assert size_of({"dims": "16x16"}) == (16, "L")
    assert size_of({"n": "200"}) == (200, "N")
    assert size_of({"graph": "complete", "n": "200", "dims": "4x4"}) == (200, "N")
    size, label = size_of({"dims": "4x4x4"}, "N")
    assert (size, label) == (64, "N")
    size, label = size_of({"dims": "8x8x8"}, "L")
    assert label == "L" and size == pytest.approx(8)
    assert size_of({"dims": "8x32"}, "L")[0] == pytest.approx(16)
    with pytest.raises(InputError):
        size_of({"dims": "8x16"})
    with pytest.raises(InputError):
        size_of({"model": "ising"})
    with pytest.raises(InputError):
        size_of({"dims": "8x8"}, "bogus")


def test_bad_column_and_missing_sweep(tmp_path, outputs):
    with pytest.raises(InputError, match="nope"):
        families([outputs["stat2"]], "nope")
    with pytest.raises(InputError):
        families([outputs["run"]], "binder")
    with pytest.raises(InputError):
        families([], "binder")
