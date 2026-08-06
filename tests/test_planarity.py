"""Planarity of a centre and its three substituents."""

from __future__ import annotations

import numpy as np
import pytest

from md_interactions.analyses import planarity
from md_interactions.config import Config
from md_interactions.exceptions import ConfigError
from md_interactions.system import load_system

# An ideal trigonal-planar centre at the origin: three arms 120° apart in z = 0.
PLANAR = {
    "centre": np.array([0.0, 0.0, 0.0]),
    "a": np.array([1.4, 0.0, 0.0]),
    "b": np.array([-0.7, 1.212435565, 0.0]),
    "c": np.array([-0.7, -1.212435565, 0.0]),
}


def test_a_planar_centre_measures_zero():
    distance, angle_sum = planarity.out_of_plane(**PLANAR)
    assert distance == pytest.approx(0.0, abs=1e-9)
    assert angle_sum == pytest.approx(360.0, abs=1e-6)


def test_lifting_the_centre_gives_exactly_the_height():
    # the substituents lie in z = 0, so the distance is the centre's own z
    for height in (0.1, 0.35, -0.35):
        distance, _ = planarity.out_of_plane(
            np.array([0.0, 0.0, height]), PLANAR["a"], PLANAR["b"], PLANAR["c"])
        assert distance == pytest.approx(height, abs=1e-9)


def test_the_sign_follows_the_order_of_the_substituents():
    up = np.array([0.0, 0.0, 0.4])
    forward, _ = planarity.out_of_plane(up, PLANAR["a"], PLANAR["b"], PLANAR["c"])
    # swapping two substituents flips the normal, so it must flip the sign:
    # this is what lets an inversion of configuration be read off the series
    reversed_, _ = planarity.out_of_plane(up, PLANAR["a"], PLANAR["c"], PLANAR["b"])
    assert forward == pytest.approx(-reversed_, abs=1e-9)


def test_an_ideal_tetrahedral_centre_matches_the_reference():
    # methane geometry: three of the four arms, centre at the origin
    arms = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0]],
                    dtype=float)
    _distance, angle_sum = planarity.out_of_plane(
        np.zeros(3), arms[0], arms[1], arms[2])
    assert angle_sum == pytest.approx(planarity.TETRAHEDRAL_ANGLE_SUM, abs=0.1)


def test_collinear_substituents_report_nan_instead_of_a_wrong_plane():
    # every plane through three collinear points is equally valid, so there is
    # no out-of-plane distance; silently returning one would be a fabrication
    distance, angle_sum = planarity.out_of_plane(
        np.array([0.0, 1.0, 0.0]), np.array([-1.0, 0.0, 0.0]),
        np.zeros(3) + np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    assert np.isnan(distance) and np.isnan(angle_sum)


def test_run_writes_the_series_and_both_summary_rows(make_config, out_paths):
    config = make_config({"planarity": {"definitions": [
        {"name": "p_lig", "atoms": ["LIG_C1", "LIG_O1", "SER_OG", "ASP_OD2"]},
    ]}})
    system = load_system(config, verbose=False)
    result = planarity.run(system, out_paths, verbose=False)

    table = result.tables["planarity"]
    assert "p_lig" in table.columns
    assert "p_lig_angle_sum" in table.columns
    assert ((table["p_lig_angle_sum"] > 0) & (table["p_lig_angle_sum"] <= 360)).all()

    observables = set(result.summary["observable"])
    # the magnitude is reported separately: averaging the signed value of a
    # centre that oscillates through the plane would report ~0 wrongly
    assert {"p_lig", "p_lig_abs", "p_lig_angle_sum"} <= observables
    abs_row = result.summary[result.summary["observable"] == "p_lig_abs"].iloc[0]
    assert abs_row["mean"] >= 0

    assert (out_paths.plots / "planarity_p_lig_timeseries.png").is_file()
    assert (out_paths.plots / "planarity_p_lig_histogram.png").is_file()


def test_planarity_needs_four_atoms(make_config):
    with pytest.raises(ConfigError):
        make_config({"planarity": {"definitions": [
            {"name": "bad", "atoms": ["LIG_C1", "LIG_O1", "SER_OG"]},
        ]}})


def test_a_name_cannot_clash_with_another_observable(make_config):
    with pytest.raises(ConfigError, match="Duplicated"):
        make_config({
            "distances": {"pairs": [
                {"name": "shared", "atoms": ["LIG_C1", "LIG_O1"]}]},
            "planarity": {"definitions": [
                {"name": "shared",
                 "atoms": ["LIG_C1", "LIG_O1", "SER_OG", "ASP_OD2"]}]},
        })


def test_input_file_section_and_its_alias(tmp_path):
    from md_interactions.inputfile import input_to_dict, parse_input

    text = """
    topology    top.prmtop
    trajectory  traj.nc

    [pplane]
    p_anomeric  TRH453:C1  TRH453:O5  TRH453:C2  TRH453:H1
    """
    path = tmp_path / "planar.in"
    path.write_text(text, encoding="utf-8")
    data = input_to_dict(parse_input(path))

    definitions = data["analyses"]["planarity"]["definitions"]
    assert len(definitions) == 1
    assert definitions[0]["name"] == "p_anomeric"
    assert len(definitions[0]["atoms"]) == 4
    # the centre is written first and stays first: the whole measurement
    # depends on which of the four is the centre
    centre = definitions[0]["atoms"][0]
    assert data["selections"][centre] == "resname TRH and resid 453 and name C1"


def test_config_round_trip_keeps_planarity(make_config):
    config: Config = make_config({"planarity": {"definitions": [
        {"name": "p_lig", "atoms": ["LIG_C1", "LIG_O1", "SER_OG", "ASP_OD2"]},
    ]}})
    assert config.planarity.enabled
    assert config.planarity.definitions[0].atoms[0] == "LIG_C1"
