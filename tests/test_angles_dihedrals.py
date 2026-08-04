"""Angles and dihedrals: values, ranges and circular statistics."""

from __future__ import annotations

import numpy as np
import pytest

from md_interactions.analyses import angles_dihedrals
from md_interactions.system import load_system

_ANGLE = {"name": "a_attack", "atoms": ["SER_OG", "LIG_C1", "LIG_O1"]}
_DIHEDRAL = {
    "name": "chi1",
    "atoms": ["resid 1 and name N", "resid 1 and name CA",
              "resid 1 and name CB", "SER_OG"],
}


def test_angle_matches_manual_computation(make_config):
    system = load_system(make_config({"angles": {"definitions": [_ANGLE]}}), verbose=False)
    angles, dihedrals = angles_dihedrals.compute_angles(system)
    assert dihedrals.shape[1] == 2  # only frame/time

    system.goto(0)
    a = system.select("SER_OG").positions[0]
    b = system.select("LIG_C1").positions[0]
    c = system.select("LIG_O1").positions[0]
    v1, v2 = a - b, c - b
    expected = np.degrees(np.arccos(
        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    ))
    assert angles["a_attack"].iloc[0] == pytest.approx(float(expected), abs=1e-3)
    assert ((angles["a_attack"] >= 0) & (angles["a_attack"] <= 180)).all()


def test_dihedral_range_and_circular_summary(make_config, out_paths):
    config = make_config({
        "angles": {"definitions": [_ANGLE]},
        "dihedrals": {"definitions": [_DIHEDRAL]},
    })
    system = load_system(config, verbose=False)
    result = angles_dihedrals.run(system, out_paths, verbose=False)

    dihedrals = result.tables["dihedrals"]
    assert ((dihedrals["chi1"] > -180.0) & (dihedrals["chi1"] <= 180.0)).all()

    summary = result.summary.set_index("observable")
    assert summary.loc["chi1", "circular"] is True
    assert summary.loc["a_attack", "unit"] == "°"

    for name in ("angle_a_attack_timeseries", "angle_a_attack_histogram",
                 "dihedral_chi1_timeseries", "dihedral_chi1_histogram"):
        assert (out_paths.plots / f"{name}.png").is_file()
    assert (out_paths.data / "angles.csv").is_file()
    assert (out_paths.data / "dihedrals.csv").is_file()


def test_series_contains_both_kinds(make_config, out_paths):
    config = make_config({
        "angles": {"definitions": [_ANGLE]},
        "dihedrals": {"definitions": [_DIHEDRAL]},
    })
    system = load_system(config, verbose=False)
    result = angles_dihedrals.run(system, out_paths, verbose=False)
    assert {"a_attack", "chi1"} <= set(result.series.columns)
