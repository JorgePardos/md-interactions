"""Hydrogen bonds: tracked pairs, occupancy and automatic detection."""

from __future__ import annotations

import numpy as np
import pytest

from md_interactions.analyses import hbonds
from md_interactions.exceptions import MissingTopologyInfoError
from md_interactions.system import load_system

_PAIRS = [
    {"name": "hb_ser_his", "donor": "SER_OG", "acceptor": "HIS_NE2"},
    {"name": "hb_his_asp", "donor": "HIS_NE2", "acceptor": "ASP_OD2"},
]


def _config(make_config, **overrides):
    return make_config({"hbonds": {"pairs": _PAIRS, **overrides}})


def test_tracked_hbonds_columns_and_ranges(make_config):
    system = load_system(_config(make_config), verbose=False)
    df = hbonds.compute_tracked_hbonds(system)

    for name in ("hb_ser_his", "hb_ser_his_angle", "hb_ser_his_present"):
        assert name in df.columns
    assert (df["hb_ser_his"] > 0).all()
    assert ((df["hb_ser_his_angle"] >= 0) & (df["hb_ser_his_angle"] <= 180)).all()
    assert df["hb_ser_his_present"].dtype == bool


def test_occupancy_matches_the_criteria(make_config):
    system = load_system(_config(make_config, d_a_cutoff=3.5,
                                 d_h_a_angle_cutoff=150.0), verbose=False)
    df = hbonds.compute_tracked_hbonds(system)
    expected = ((df["hb_his_asp"] <= 3.5) & (df["hb_his_asp_angle"] >= 150.0)).mean()
    assert df["hb_his_asp_present"].mean() == pytest.approx(expected)

    # the toy triad is built to be hydrogen bonded most of the time
    assert df["hb_his_asp_present"].mean() > 0.5


def test_tight_cutoff_gives_zero_occupancy(make_config):
    system = load_system(_config(make_config, d_a_cutoff=1.0), verbose=False)
    df = hbonds.compute_tracked_hbonds(system)
    assert not df["hb_ser_his_present"].any()


def test_missing_hydrogen_raises_clear_error(make_config):
    """A donor without hydrogens must be reported, not silently ignored."""
    config = make_config({"hbonds": {"pairs": [
        {"name": "bad", "donor": "resid 3 and name OD1", "acceptor": "HIS_NE2"},
    ]}})
    system = load_system(config, verbose=False)
    with pytest.raises(MissingTopologyInfoError, match="No hydrogen found"):
        hbonds.compute_tracked_hbonds(system)


def test_automatic_detection(make_config):
    config = make_config({"hbonds": {
        "pairs": _PAIRS,
        "auto": True,
        "auto_between": [["resname WAT", "protein"]],
        "min_occupancy": 0.0,
        "donors_sel": "name N* O*",
        "hydrogens_sel": "mass 0.5 to 1.5",
        "acceptors_sel": "name N* O*",
    }})
    system = load_system(config, verbose=False)
    detected = hbonds.detect_hbonds(system)
    assert set(detected.columns) >= {"donor", "hydrogen", "acceptor", "occupancy_%"}
    assert (detected["occupancy_%"] <= 100.0).all()


def test_run_writes_plots_and_summary(make_config, out_paths):
    system = load_system(_config(make_config), verbose=False)
    result = hbonds.run(system, out_paths, verbose=False)

    assert (out_paths.data / "hbonds_tracked.csv").is_file()
    assert (out_paths.plots / "hbond_hb_ser_his_timeseries.png").is_file()
    assert (out_paths.plots / "hbond_occupancy.png").is_file()

    summary = result.summary.set_index("observable")
    assert summary.loc["hb_ser_his", "unit"] == "Å"
    assert summary.loc["hb_ser_his_angle", "unit"] == "°"
    assert np.isfinite(summary.loc["hb_ser_his", "occupancy_pct"])
