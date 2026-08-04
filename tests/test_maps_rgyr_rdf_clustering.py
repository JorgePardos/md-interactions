"""Free-energy maps, radius of gyration, RDF and conformational clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from md_interactions.analyses import (
    clustering,
    free_energy_map,
    radius_of_gyration,
    rdf,
)
from md_interactions.config import FreeEnergyMap
from md_interactions.exceptions import AnalysisError
from md_interactions.system import load_system


# --------------------------------------------------------------------------- #
# free-energy maps
# --------------------------------------------------------------------------- #
def _gaussian_sample(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(3.0, 0.3, n), rng.normal(120.0, 8.0, n)


def test_free_energy_is_zero_at_the_most_populated_bin():
    x, y = _gaussian_sample()
    spec = FreeEnergyMap(name="m", x="x", y="y", bins=40, temperature=300.0)
    _xe, _ye, density, energy = free_energy_map.compute_map(x, y, spec)

    assert np.nanmin(energy) == pytest.approx(0.0)
    assert np.isnan(energy[density == 0]).all()
    # the minimum of the free energy sits on the maximum of the density
    assert np.unravel_index(np.nanargmin(energy), energy.shape) == \
        np.unravel_index(np.argmax(density), density.shape)


def test_free_energy_scales_with_temperature():
    x, y = _gaussian_sample()
    low = free_energy_map.compute_map(
        x, y, FreeEnergyMap(name="m", x="x", y="y", bins=30, temperature=150.0))[3]
    high = free_energy_map.compute_map(
        x, y, FreeEnergyMap(name="m", x="x", y="y", bins=30, temperature=300.0))[3]
    ratio = np.nanmax(high) / np.nanmax(low)
    assert ratio == pytest.approx(2.0, rel=1e-6)


def test_bins_are_capped_for_short_trajectories():
    assert free_energy_map.effective_bins(60, 200) == 10     # sqrt(200/2)
    assert free_energy_map.effective_bins(60, 20_000) == 60  # request honoured
    assert free_energy_map.effective_bins(60, 10) == 10      # floor


def test_smoothing_fills_empty_bins():
    x, y = _gaussian_sample(n=300)
    rough = FreeEnergyMap(name="m", x="x", y="y", bins=30)
    smooth = FreeEnergyMap(name="m", x="x", y="y", bins=30, smooth=1.0)
    _xe, _ye, rough_density, _e = free_energy_map.compute_map(x, y, rough)
    _xe, _ye, smooth_density, _e = free_energy_map.compute_map(x, y, smooth)
    assert (smooth_density > 0).sum() > (rough_density > 0).sum()


def test_unknown_observable_is_reported(make_config, out_paths):
    config = make_config({"free_energy_maps": {
        "maps": [{"name": "m", "x": "d_nuc", "y": "missing"}]
    }})
    system = load_system(config, verbose=False)
    observables = pd.DataFrame({"frame": [0, 1], "time": [0.0, 0.1],
                                "d_nuc": [3.0, 3.1]})
    with pytest.raises(AnalysisError, match="unknown observable"):
        free_energy_map.run(system, out_paths, observables=observables, verbose=False)


def test_map_run_writes_plot_and_grid(make_config, out_paths):
    config = make_config({"free_energy_maps": {
        "maps": [{"name": "fes_test", "x": "d1", "y": "d2", "bins": 20}]
    }})
    system = load_system(config, verbose=False)
    x, y = _gaussian_sample(n=500)
    observables = pd.DataFrame({"frame": np.arange(500), "time": np.arange(500) * 0.01,
                                "d1": x, "d2": y})
    result = free_energy_map.run(system, out_paths, observables=observables,
                                 units={"d1": "Å", "d2": "°"}, verbose=False)
    assert (out_paths.plots / "fes_test.png").is_file()
    assert (out_paths.data / "fes_test.csv").is_file()
    assert set(result.tables["fes_test"].columns) == {"d1", "d2", "density", "free_energy"}


def test_kde_method(make_config, out_paths):
    config = make_config({"free_energy_maps": {
        "maps": [{"name": "kde_test", "x": "d1", "y": "d2", "method": "kde",
                  "bins": 15, "free_energy": False}]
    }})
    system = load_system(config, verbose=False)
    x, y = _gaussian_sample(n=300)
    observables = pd.DataFrame({"frame": np.arange(300), "time": np.arange(300) * 0.01,
                                "d1": x, "d2": y})
    free_energy_map.run(system, out_paths, observables=observables, verbose=False)
    # names that do not already start with the prefix get it added
    assert (out_paths.plots / "fes_kde_test.png").is_file()


# --------------------------------------------------------------------------- #
# radius of gyration
# --------------------------------------------------------------------------- #
def test_rgyr_values_and_outputs(make_config, out_paths):
    config = make_config({"radius_of_gyration": {
        "groups": [{"name": "rg_protein", "selection": "protein"}]
    }})
    system = load_system(config, verbose=False)
    result = radius_of_gyration.run(system, out_paths, verbose=False)

    values = result.tables["radius_of_gyration"]["rg_protein"].to_numpy()
    assert len(values) == system.n_frames
    assert (values > 0).all()

    system.goto(0)
    assert values[0] == pytest.approx(
        system.select("protein").radius_of_gyration(), abs=1e-4
    )
    assert (out_paths.plots / "rgyr_rg_protein_timeseries.png").is_file()
    assert (out_paths.data / "radius_of_gyration.csv").is_file()


# --------------------------------------------------------------------------- #
# RDF
# --------------------------------------------------------------------------- #
def test_rdf_shape_and_coordination(make_config, out_paths):
    config = make_config({"rdf": {"pairs": [
        {"name": "rdf_lig_wat", "g1": "LIG_O1", "g2": "WATERS",
         "nbins": 30, "range": [0.0, 10.0]},
    ]}})
    system = load_system(config, verbose=False)
    result = rdf.run(system, out_paths, verbose=False)

    df = result.tables["rdf_lig_wat"]
    assert len(df) == 30
    assert (df["g_r"] >= 0).all()
    assert (np.diff(df["n_r"]) >= -1e-12).all()   # n(r) is cumulative
    assert (out_paths.plots / "rdf_lig_wat.png").is_file()


# --------------------------------------------------------------------------- #
# clustering
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["hierarchical", "kmeans"])
def test_clustering_populations_and_medoids(make_config, out_paths, method):
    config = make_config({"clustering": {
        "enabled": True,
        "selection": "protein and not name H*",
        "method": method,
        "n_clusters": 2,
    }})
    system = load_system(config, verbose=False)
    result = clustering.run(system, out_paths, verbose=False)

    assignment = result.tables["cluster_assignment"]
    summary = result.tables["clusters"]
    assert len(assignment) == system.n_frames
    assert summary["population"].sum() == system.n_frames
    assert summary["population_%"].sum() == pytest.approx(100.0)
    assert set(summary["representative_frame"]) <= set(system.frame_indices.tolist())

    assert (out_paths.plots / "cluster_populations.png").is_file()
    assert (out_paths.plots / "cluster_pca.png").is_file()
    structures = list(out_paths.structures.glob("*.pdb"))
    assert len(structures) == len(summary)


def test_write_selection_trims_the_representative_pdb(make_config, out_paths):
    """By default the whole system is written; write_selection trims it."""
    full = make_config({"clustering": {
        "enabled": True, "selection": "protein", "n_clusters": 1,
    }})
    system = load_system(full, verbose=False)
    clustering.run(system, out_paths, verbose=False)
    whole = next(out_paths.structures.glob("*.pdb")).read_text().count("ATOM")

    trimmed_cfg = make_config({"clustering": {
        "enabled": True, "selection": "protein", "n_clusters": 1,
        "write_selection": "resname LIG",
    }})
    system = load_system(trimmed_cfg, verbose=False)
    clustering.run(system, out_paths, verbose=False)
    trimmed = next(out_paths.structures.glob("*.pdb")).read_text().count("ATOM")

    assert trimmed < whole
    assert trimmed == system.select("resname LIG").n_atoms


def test_clustering_subsamples_long_trajectories(make_config, out_paths):
    config = make_config({"clustering": {
        "enabled": True, "selection": "protein", "n_clusters": 2, "max_frames": 10,
    }})
    system = load_system(config, verbose=False)
    assignment, summary, coords = clustering.compute_clusters(system)
    assert len(assignment) <= 10
    assert coords.shape[0] == len(assignment)
