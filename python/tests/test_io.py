import numpy as np
import pytest

from odspy.errors import InputError
from odspy.io import find_outputs, read_table


def kinds(o):
    return set(o.tables)


def test_read_every_kind(outputs):
    o = find_outputs(outputs["run"])
    assert "timeseries_run" in kinds(o)
    o = find_outputs(outputs["ens"])
    assert {"timeseries_ensemble", "trajectories", "replicas", "domains"} <= kinds(o)
    assert o.get("trajectories")["replica"].size > 0
    assert find_outputs(outputs["ens_sweep"]).get("sweep") is not None
    assert find_outputs(outputs["stat2"]).get("sweep") is not None
    o = find_outputs(outputs["cont"])
    assert o.get("sweep") is not None
    assert o.points
    pk = set().union(*(kinds(p) for p in o.points.values()))
    assert {"clusters", "fixation"} <= pk
    o = find_outputs(outputs["stat"])
    assert {"histogram", "samples", "states", "correlation_stationary"} <= kinds(o)
    st = o.get("states")
    assert st.names[0] == "node" and len(st) == 64
    o = find_outputs(outputs["maj"])
    assert {"replicas", "domains"} <= kinds(o)
    assert o.get("replicas").has("freeze_time")


def test_provenance_and_config(outputs):
    t = find_outputs(outputs["stat2"]).get("sweep")
    assert t.provenance.config["model"] == "ising"
    assert t.provenance.get("sweep2").startswith("temperature=")
    assert t.provenance.get("nope", "d") == "d"
    assert "ising" in t.provenance.caption()
    assert "ODSP" in t.provenance.caption()


def test_text_columns_kept(outputs):
    t = find_outputs(outputs["stat2"]).get("sweep")
    assert t["dims"].dtype == object
    assert "8x8" in set(t["dims"])
    assert t["temperature"].dtype == float


def test_no_provenance(outputs, tmp_path):
    # The driver still writes P_config.txt with --provenance false, so the
    # table itself has no provenance; find_outputs fills it in from the config.
    t = read_table(outputs["noprov"] + "_timeseries.csv")
    assert t.provenance.config == {} and t.provenance.caption() == ""
    assert find_outputs(outputs["noprov"]).get("timeseries_run").provenance.config["model"] == "voter"
    import shutil
    shutil.copy(outputs["noprov"] + "_timeseries.csv", tmp_path / "n_timeseries.csv")
    o = find_outputs(tmp_path / "n")
    assert o.get("timeseries_run").provenance.config == {}
    assert o.provenance.config == {} and o.provenance.caption() == ""


def test_config_used_when_no_preamble(outputs, tmp_path):
    import shutil
    src = outputs["run"]
    dst = tmp_path / "x"
    shutil.copy(src + "_config.txt", str(dst) + "_config.txt")
    shutil.copy(src + "_timeseries.csv", str(dst) + "_timeseries.csv")
    lines = open(str(dst) + "_timeseries.csv").read().splitlines()
    open(str(dst) + "_timeseries.csv", "w").write(
        "\n".join(l for l in lines if not l.startswith("#")) + "\n")
    o = find_outputs(dst)
    assert o.get("timeseries_run").provenance.config["model"] == "voter"


def test_missing_column_names_file(tmp_path):
    p = tmp_path / "a_sweep.csv"
    p.write_text("x,y\n1,2\n3,4\n")
    t = read_table(p)
    with pytest.raises(InputError) as e:
        t["nope"]
    assert str(p) in str(e.value) and "nope" in str(e.value)
    assert t.has("x") and not t.has("nope") and len(t) == 2


def test_nan_cells(tmp_path):
    p = tmp_path / "a_sweep.csv"
    p.write_text("x,y\n1,nan\n2,inf\n3,-inf\n")
    t = read_table(p)
    assert t["y"].dtype == float
    assert np.isnan(t["y"][0]) and t["y"][1] == np.inf and t["y"][2] == -np.inf


def test_malformed_rows(tmp_path):
    p = tmp_path / "a_sweep.csv"
    p.write_text("x,y\n1,2\n3,4,5\n")
    with pytest.raises(InputError) as e:
        read_table(p)
    assert str(p) in str(e.value) and "line 3" in str(e.value)


def test_unknown_suffix_and_empty(tmp_path):
    p = tmp_path / "a_weird.csv"
    p.write_text("x\n1\n")
    with pytest.raises(InputError):
        read_table(p)
    q = tmp_path / "b_sweep.csv"
    q.write_text("# only comment\n")
    with pytest.raises(InputError):
        read_table(q)
    with pytest.raises(InputError):
        read_table(tmp_path / "missing_sweep.csv")


def test_prefix_with_suffix(outputs):
    a = find_outputs(outputs["ens"])
    b = find_outputs(outputs["ens"] + "_replicas.csv")
    assert kinds(a) == kinds(b)
    c = find_outputs(outputs["cont"] + "_point1_clusters.csv")
    assert 1 in c.points


def test_unknown_prefix(tmp_path):
    with pytest.raises(InputError, match="no ODSP outputs found for prefix"):
        find_outputs(tmp_path / "nothing")


def test_header_only_table(tmp_path):
    p = tmp_path / "h_sweep.csv"
    p.write_text("# ODSP 0.1.0, git abc, written now\n#   model = ising\nx,y\n")
    t = read_table(p)
    assert len(t) == 0 and t.names == ["x", "y"] and t["x"].size == 0
    o = find_outputs(tmp_path / "h")
    assert len(o.get("sweep")) == 0
    assert o.provenance.config["model"] == "ising"
