"""Distance module: values, modes, outputs and error handling."""

from __future__ import annotations

import numpy as np
import pytest

from md_interactions.analyses import distances
from md_interactions.exceptions import AtomCountError
from md_interactions.system import load_system


def _config(make_config, **overrides):
    pairs = overrides.pop("pairs", [
        {"name": "d_nuc", "atoms": ["SER_OG", "LIG_C1"], "threshold": 3.2},
        {"name": "d_acid", "atoms": ["SER_OG", "HIS_NE2"]},
    ])
    return make_config({"distances": {"pairs": pairs, **overrides}})


def test_distance_values_match_numpy(make_config):
    system = load_system(_config(make_config), verbose=False)
    df = distances.compute_distances(system)

    assert list(df.columns) == ["frame", "time", "d_nuc", "d_acid"]
    assert len(df) == system.n_frames

    # recompute the first frame by hand
    system.goto(0)
    og = system.select("SER_OG").positions[0]
    c1 = system.select("LIG_C1").positions[0]
    expected = float(np.linalg.norm(og - c1))
    assert df["d_nuc"].iloc[0] == pytest.approx(expected, abs=1e-4)


def test_min_mode_uses_the_closest_atom(make_config):
    system = load_system(
        _config(make_config, pairs=[
            {"name": "d_wat", "atoms": ["LIG_C1", "WATERS"], "mode": "min"},
        ]),
        verbose=False,
    )
    df = distances.compute_distances(system)
    system.goto(0)
    c1 = system.select("LIG_C1").positions[0]
    waters = system.select("WATERS").positions
    expected = float(np.min(np.linalg.norm(waters - c1, axis=1)))
    assert df["d_wat"].iloc[0] == pytest.approx(expected, abs=1e-4)


def test_com_mode_runs_on_multi_atom_groups(make_config):
    system = load_system(
        _config(make_config, pairs=[
            {"name": "d_com", "atoms": ["resname LIG", "resname WAT"], "mode": "com"},
        ]),
        verbose=False,
    )
    df = distances.compute_distances(system)
    assert np.isfinite(df["d_com"]).all()


def test_atom_mode_rejects_multi_atom_selection(make_config):
    system = load_system(
        _config(make_config, pairs=[
            {"name": "bad", "atoms": ["SER_OG", "WATERS"]},
        ]),
        verbose=False,
    )
    with pytest.raises(AtomCountError, match="exactly 1 atom"):
        distances.compute_distances(system)


def test_run_writes_csv_plots_and_summary(make_config, out_paths):
    system = load_system(_config(make_config), verbose=False)
    result = distances.run(system, out_paths, verbose=False)

    assert (out_paths.data / "distances.csv").is_file()
    for name in ("dist_d_nuc_timeseries", "dist_d_nuc_histogram",
                 "dist_all_timeseries", "dist_distributions"):
        assert (out_paths.plots / f"{name}.png").is_file()

    summary = result.summary.set_index("observable")
    assert summary.loc["d_nuc", "unit"] == "Å"
    assert summary.loc["d_nuc", "n_frames"] == system.n_frames
    assert 0.0 <= summary.loc["d_nuc", "frames_below_threshold_%"] <= 100.0
    assert result.series is not None and "d_nuc" in result.series.columns
