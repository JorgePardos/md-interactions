"""Replica handling (trajectories and tables) and the ``data:`` mode."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from md_interactions import cli
from md_interactions.analyses import replicas
from md_interactions.config import Config
from md_interactions.exceptions import ConfigError, MDInteractionsError
from md_interactions.plotting import facet_layout, plot_facet_distributions
from md_interactions.runner import run_analyses
from md_interactions.system import load_system
from md_interactions.tabular import load_dataset, read_table
from md_interactions.testing import write_toy_system

DISTANCES = {"distances": {"pairs": [
    {"name": "d_nuc", "atoms": ["SER_OG", "LIG_C1"]},
    {"name": "d_acid", "atoms": ["SER_OG", "HIS_NE2"]},
]}}


# --------------------------------------------------------------------------- #
# layout / facet figure
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_panels,expected_cols", [(2, 2), (8, 2), (11, 3), (30, 5)])
def test_facet_layout_fits_a_page(n_panels, expected_cols):
    nrows, ncols = facet_layout(n_panels)
    assert ncols == expected_cols
    assert nrows * 1.85 <= 9.0 or ncols == 5


def test_facet_figure_has_one_panel_per_observable():
    rng = np.random.default_rng(0)
    series = {f"d{i}": rng.normal(3 + i, 0.3, 200) for i in range(5)}
    fig = plot_facet_distributions(series, groups={"rep1": series, "rep2": series})
    assert len(fig.axes) == 5           # empty panels are removed
    fig.clf()


# --------------------------------------------------------------------------- #
# replicas from trajectories
# --------------------------------------------------------------------------- #
@pytest.fixture
def replica_files(tmp_path_factory):
    directory = tmp_path_factory.mktemp("replicas")
    first = write_toy_system(directory / "rep1", n_frames=20, seed=1)
    second = write_toy_system(directory / "rep2", n_frames=15, seed=2)
    return first, second


def _replica_config(replica_files, tmp_path, analyses=None):
    (topology, traj1), (_top2, traj2) = replica_files
    return Config.from_dict({
        "system": {
            "topology": str(topology),
            "replicas": {"rep1": [str(traj1)], "rep2": [str(traj2)]},
            "time": {"dt": 0.01, "unit": "ns"},
        },
        "output": {"directory": str(tmp_path / "results"), "formats": ["png"],
                   "dpi": 80},
        "selections": {"SER_OG": "resid 1 and name OG",
                       "HIS_NE2": "resid 2 and name NE2",
                       "LIG_C1": "resname LIG and name C1"},
        "analyses": analyses or DISTANCES,
        "report": {"enabled": False},
    })


def test_replica_labels_follow_the_files(replica_files, tmp_path):
    system = load_system(_replica_config(replica_files, tmp_path), verbose=False)
    assert system.n_frames == 35
    assert system.replica_names == ["rep1", "rep2"]
    masks = system.replica_masks()
    assert masks["rep1"].sum() == 20
    assert masks["rep2"].sum() == 15
    # the label is carried into every time-series CSV
    assert "replica" in system.time_frame_columns()


def test_trajectory_and_replicas_are_exclusive(replica_files):
    (topology, traj1), _ = replica_files
    with pytest.raises(ConfigError, match="not both"):
        Config.from_dict({"system": {
            "topology": str(topology),
            "trajectory": [str(traj1)],
            "replicas": {"rep1": [str(traj1)]},
        }})


def test_run_with_replicas_adds_the_comparison(replica_files, tmp_path):
    output = run_analyses(_replica_config(replica_files, tmp_path),
                          verbose=False, strict=True)
    assert output.ok
    comparison = output.result("replicas")
    assert comparison is not None
    overlap = comparison.tables["replica_overlap"]
    assert set(overlap["observable"]) == {"d_nuc", "d_acid"}
    assert ((overlap["overlap"] >= 0) & (overlap["overlap"] <= 1)).all()
    assert (output.paths.summary / "replica_overlap.csv").is_file()
    assert (output.paths.plots / "replica_overlap.png").is_file()
    assert "replica" in output.observables.columns


# --------------------------------------------------------------------------- #
# overlap coefficient
# --------------------------------------------------------------------------- #
def test_overlap_is_one_for_identical_and_zero_for_disjoint():
    rng = np.random.default_rng(3)
    a = rng.normal(3.0, 0.2, 5000)
    b = rng.normal(3.0, 0.2, 5000)
    far = rng.normal(9.0, 0.2, 5000)
    assert replicas.overlap_coefficient(a, b) > 0.9
    assert replicas.overlap_coefficient(a, far) == pytest.approx(0.0, abs=1e-6)


def test_identical_replicas_are_flagged_as_duplicated_input():
    """Bit-identical replicas mean the same file was read twice, not convergence."""
    rng = np.random.default_rng(7)
    values = rng.normal(3.0, 0.3, 200)
    frame = pd.DataFrame({
        "replica": ["rep1"] * 200 + ["rep2"] * 200,
        "d": np.concatenate([values, values]),
    })
    _stats, pairwise = replicas.compare_replicas(frame, ["d"])
    assert bool(pairwise["identical"].iloc[0])
    assert pairwise["overlap"].iloc[0] == pytest.approx(1.0)
    assert pairwise["delta_mean"].iloc[0] == pytest.approx(0.0)


def test_different_replicas_are_not_flagged_as_identical():
    rng = np.random.default_rng(8)
    frame = pd.DataFrame({
        "replica": ["rep1"] * 200 + ["rep2"] * 200,
        "d": np.concatenate([rng.normal(3.0, 0.3, 200), rng.normal(3.0, 0.3, 200)]),
    })
    _stats, pairwise = replicas.compare_replicas(frame, ["d"])
    assert not bool(pairwise["identical"].iloc[0])


def test_divergent_replica_is_flagged():
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({
        "replica": ["rep1"] * 300 + ["rep2"] * 300,
        "d": np.concatenate([rng.normal(3.0, 0.2, 300), rng.normal(6.0, 0.2, 300)]),
        "same": np.concatenate([rng.normal(2.0, 0.3, 300), rng.normal(2.0, 0.3, 300)]),
    })
    _stats, pairwise = replicas.compare_replicas(frame, ["d", "same"])
    assert bool(pairwise.loc[pairwise["observable"] == "d", "diverges"].iloc[0])
    assert not bool(pairwise.loc[pairwise["observable"] == "same", "diverges"].iloc[0])


# --------------------------------------------------------------------------- #
# tables (data: mode)
# --------------------------------------------------------------------------- #
CPPTRAJ = """\
#Frame     d_nuc   d_acid
       1   2.5962   2.3542
       2   1.8416   1.9695
       3   2.1000   2.0500
       4   2.3000   2.2500
       5   1.9500   2.1000
"""


def _write_dat(path, text=CPPTRAJ):
    path.write_text(text, encoding="utf-8")
    return path


def test_read_cpptraj_table(tmp_path):
    table = read_table(_write_dat(tmp_path / "d.dat"))
    assert list(table.columns) == ["index", "d_nuc", "d_acid"]
    assert len(table) == 5
    assert table["d_nuc"].iloc[0] == pytest.approx(2.5962)


def test_read_table_without_header(tmp_path):
    path = tmp_path / "plain.dat"
    path.write_text("1 2.0 3.0\n2 2.1 3.1\n", encoding="utf-8")
    table = read_table(path)
    assert len(table) == 2
    assert len(table.columns) == 3


def test_column_mismatch_between_replicas_is_reported(tmp_path):
    _write_dat(tmp_path / "a.dat")
    _write_dat(tmp_path / "b.dat", CPPTRAJ.replace("d_acid", "other"))
    config = Config.from_dict({
        "data": {"replicas": {"rep1": [str(tmp_path / "a.dat")],
                              "rep2": [str(tmp_path / "b.dat")]}},
        "output": {"directory": str(tmp_path / "results")},
    })
    with pytest.raises(MDInteractionsError, match="Column mismatch"):
        load_dataset(config, verbose=False)


def test_data_mode_run(tmp_path):
    _write_dat(tmp_path / "rep1.dat")
    _write_dat(tmp_path / "rep2.dat")
    config = Config.from_dict({
        "data": {
            "replicas": {"rep1": [str(tmp_path / "rep1.dat")],
                         "rep2": [str(tmp_path / "rep2.dat")]},
            "time": {"dt": 0.5, "unit": "ps"},
            "bins": 5,
            "thresholds": {"d_nuc": 2.2},
        },
        "output": {"directory": str(tmp_path / "results"), "formats": ["png"],
                   "dpi": 80},
        "analyses": {"free_energy_maps": {"maps": [
            {"name": "fes", "x": "d_nuc", "y": "d_acid", "bins": 5},
        ]}},
        "report": {"enabled": True, "formats": ["markdown"]},
    })
    assert config.data_only
    assert config.enabled_analyses() == ["distributions", "replicas", "free_energy_maps"]

    output = run_analyses(config, verbose=False, strict=True)
    assert output.ok and output.system is None
    assert output.dataset.n_frames == 10
    assert output.dataset.times[1] == pytest.approx(1.0)     # dt = 0.5 ps

    summary = output.summary.set_index("observable")
    assert summary.loc["d_nuc", "frames_below_threshold_%"] > 0
    assert (output.paths.plots / "distributions.png").is_file()
    assert (output.paths.plots / "fes.png").is_file()
    assert (output.paths.root / "report.md").is_file()
    assert output.result("replicas") is not None


def test_data_mode_labels_and_column_subset(tmp_path):
    _write_dat(tmp_path / "a.dat")
    config = Config.from_dict({
        "data": {"files": [str(tmp_path / "a.dat")],
                 "columns": ["d_nuc"],
                 "labels": {"d_nuc": "d_attack"}},
        "output": {"directory": str(tmp_path / "results")},
    })
    dataset = load_dataset(config, verbose=False)
    assert dataset.columns == ["d_attack"]
    assert "d_acid" not in dataset.data.columns


def test_data_mode_unknown_column(tmp_path):
    _write_dat(tmp_path / "a.dat")
    config = Config.from_dict({
        "data": {"files": [str(tmp_path / "a.dat")], "columns": ["nope"]},
        "output": {"directory": str(tmp_path / "results")},
    })
    with pytest.raises(ConfigError, match="unknown column"):
        load_dataset(config, verbose=False)


def test_config_requires_system_or_data():
    with pytest.raises(ConfigError, match="'system:'"):
        Config.from_dict({"output": {"directory": "out"}})


def test_cli_check_on_data_mode(tmp_path, capsys):
    _write_dat(tmp_path / "a.dat")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump({
        "data": {"files": [str(tmp_path / "a.dat")]},
        "output": {"directory": str(tmp_path / "results")},
    }), encoding="utf-8")
    assert cli.main(["check", "-c", str(config_file)]) == 0
    assert "All columns available" in capsys.readouterr().out
