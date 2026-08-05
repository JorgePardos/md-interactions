"""Solvent molecules bridging two groups."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from md_interactions.analyses import water_bridges
from md_interactions.config import Config
from md_interactions.inputfile import input_to_dict, parse_input
from md_interactions.system import load_system


def _config(make_config, cutoff=3.5, group_a="SER_OG", group_b="LIG_C1"):
    return make_config({"bridges": {"pairs": [
        {"name": "br", "group_a": group_a, "group_b": group_b, "cutoff": cutoff,
         "solvent": "resname WAT and name O"},
    ]}})


def test_bridge_columns_and_ranges(make_config):
    system = load_system(_config(make_config, cutoff=6.0), verbose=False)
    frame = water_bridges.compute_bridges(system, system.config.bridges.pairs[0])

    assert list(frame.columns)[:2] == ["frame", "time"]
    assert {"n_bridges", "resid", "d_a", "d_b", "d_direct"} <= set(frame.columns)
    assert len(frame) == system.n_frames
    assert (frame["n_bridges"] >= 0).all()

    bridged = frame[frame["n_bridges"] > 0]
    if not bridged.empty:
        # both distances must respect the cutoff that defines a bridge
        assert (bridged["d_a"] <= 6.0 + 1e-6).all()
        assert (bridged["d_b"] <= 6.0 + 1e-6).all()
        assert (bridged["resid"] > 0).all()


def test_a_tight_cutoff_finds_nothing(make_config):
    system = load_system(_config(make_config, cutoff=1.5), verbose=False)
    frame = water_bridges.compute_bridges(system, system.config.bridges.pairs[0])
    assert (frame["n_bridges"] == 0).all()
    assert frame["d_a"].isna().all()


def test_bridge_distances_are_consistent_with_the_geometry(make_config):
    """The reported bridge really is within reach of both groups."""
    system = load_system(_config(make_config, cutoff=6.0), verbose=False)
    spec = system.config.bridges.pairs[0]
    frame = water_bridges.compute_bridges(system, spec)
    bridged = frame[frame["n_bridges"] > 0]
    if bridged.empty:
        pytest.skip("no bridge in the toy system at this cutoff")

    position = int(bridged.index[0])
    system.goto(position)
    water = system.select(f"resname WAT and resid {int(bridged.iloc[0]['resid'])} "
                          "and name O")
    group_a = system.select(spec.group_a)
    group_b = system.select(spec.group_b)
    d_a = float(np.linalg.norm(group_a.positions[0] - water.positions[0]))
    d_b = float(np.linalg.norm(group_b.positions[0] - water.positions[0]))
    assert bridged.iloc[0]["d_a"] == pytest.approx(d_a, abs=1e-3)
    assert bridged.iloc[0]["d_b"] == pytest.approx(d_b, abs=1e-3)


def test_residence_statistics():
    frame = pd.DataFrame({"resid": [5, 5, 5, -1, 7, 7, 5]})
    stats = water_bridges.residence_statistics(frame)
    assert stats["longest_frames"] == 3        # the run of three 5s
    assert stats["n_distinct"] == 2
    assert stats["most_common"] == 5


def test_residence_with_no_bridge():
    stats = water_bridges.residence_statistics(pd.DataFrame({"resid": [-1, -1]}))
    assert stats["longest_frames"] == 0 and stats["n_distinct"] == 0


def test_run_writes_tables_and_figures(make_config, out_paths):
    system = load_system(_config(make_config, cutoff=6.0), verbose=False)
    result = water_bridges.run(system, out_paths, verbose=False)

    assert (out_paths.data / "bridge_br.csv").is_file()
    assert (out_paths.plots / "bridge_br_count.png").is_file()

    summary = result.summary.set_index("observable")
    assert "br" in summary.index
    assert 0.0 <= summary.loc["br", "occupancy_pct"] <= 100.0
    assert result.series is not None and "nbridge_br" in result.series.columns


def test_input_file_section(tmp_path):
    text = ("topology s.prmtop\ntrajectory a.nc\n\n"
            "[bridges]\n"
            "br1  ASP20:OD2  TRH:O1  cutoff=3.0\n"
            "     ASP20:OD1  TRH:O1\n")
    path = tmp_path / "a.in"
    path.write_text(text, encoding="utf-8")
    config = Config.from_dict(input_to_dict(parse_input(path)))

    assert config.bridges.enabled
    first, second = config.bridges.pairs
    assert first.name == "br1" and first.cutoff == 3.0
    assert config.selections[first.group_a] == "resname ASP and resid 20 and name OD2"
    assert second.cutoff == 3.5                 # the default
