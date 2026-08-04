"""RMSD (global/local) and per-residue RMSF."""

from __future__ import annotations

import numpy as np
import pytest

from md_interactions.analyses import rmsd_rmsf
from md_interactions.exceptions import AnalysisError
from md_interactions.system import load_system

_GROUPS = [
    {"name": "backbone", "selection": "backbone"},
    {"name": "ligand", "selection": "resname LIG", "superposition": "backbone"},
]


def test_rmsd_starts_at_zero_and_is_positive(make_config):
    config = make_config({"rmsd": {"groups": _GROUPS, "ref_frame": 0}})
    system = load_system(config, verbose=False)
    df = rmsd_rmsf.compute_rmsd(system)

    assert list(df.columns) == ["frame", "time", "backbone", "ligand"]
    assert df["backbone"].iloc[0] == pytest.approx(0.0, abs=1e-5)
    assert (df["backbone"] >= 0).all()
    assert (df["ligand"] > 0).iloc[1:].all()


def test_rmsd_reference_frame_out_of_range(make_config):
    config = make_config({"rmsd": {"groups": _GROUPS, "ref_frame": 10_000}})
    system = load_system(config, verbose=False)
    with pytest.raises(AnalysisError, match="out of range"):
        rmsd_rmsf.compute_rmsd(system)


def test_rmsf_per_residue(make_config):
    config = make_config({"rmsf": {"enabled": True, "selection": "protein and name CA"}})
    system = load_system(config, verbose=False)
    per_atom, per_residue = rmsd_rmsf.compute_rmsf(system)

    assert len(per_atom) == 3          # three protein residues in the toy system
    assert len(per_residue) == 3
    assert (per_residue["rmsf"] > 0).all()
    assert list(per_residue["resid"]) == [1, 2, 3]


def test_rmsf_aggregates_several_atoms_per_residue(make_config):
    config = make_config({"rmsf": {"enabled": True, "selection": "protein"}})
    system = load_system(config, verbose=False)
    per_atom, per_residue = rmsd_rmsf.compute_rmsf(system)
    assert len(per_atom) > len(per_residue)
    assert (per_residue["n_atoms"] > 1).all()


def test_alignment_reduces_apparent_fluctuation(make_config):
    """Fitting must remove the global drift built into the toy trajectory."""
    config = make_config({"rmsf": {"enabled": True, "selection": "protein"}})
    system = load_system(config, verbose=False)
    _atoms, aligned = rmsd_rmsf.compute_rmsf(system)

    config_no_fit = make_config(
        {"rmsf": {"enabled": True, "selection": "protein", "align": False}}
    )
    system_no_fit = load_system(config_no_fit, verbose=False)
    _atoms2, raw = rmsd_rmsf.compute_rmsf(system_no_fit)

    assert aligned["rmsf"].mean() < raw["rmsf"].mean()


def test_run_writes_everything(make_config, out_paths):
    config = make_config({
        "rmsd": {"groups": _GROUPS},
        "rmsf": {"enabled": True, "selection": "protein and name CA", "highlight": [2]},
    })
    system = load_system(config, verbose=False)
    result = rmsd_rmsf.run(system, out_paths, verbose=False)

    for name in ("rmsd.csv", "rmsf_per_atom.csv", "rmsf_per_residue.csv"):
        assert (out_paths.data / name).is_file()
    for name in ("rmsd_backbone_timeseries", "rmsd_all_timeseries", "rmsf_per_residue"):
        assert (out_paths.plots / f"{name}.png").is_file()
    assert set(result.summary["observable"]) == {"backbone", "ligand", "rmsf_per_residue"}
    assert np.all(result.summary["unit"] == "Å")
