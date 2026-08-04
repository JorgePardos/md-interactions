"""The compact input-file format and its atom specifications."""

from __future__ import annotations

import numpy as np
import pytest

from md_interactions.config import Config
from md_interactions.exceptions import ConfigError
from md_interactions.inputfile import (
    input_to_dict,
    load_input,
    parse_input,
    resolve_atom_spec,
)


# --------------------------------------------------------------------------- #
# atom specifications
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spec,selection", [
    ("ASP20:OD1", "resname ASP and resid 20 and name OD1"),
    ("20:OD1", "resid 20 and name OD1"),
    (":20@OD1", "resid 20 and name OD1"),
    ("TRH:O2P", "resname TRH and name O2P"),
    ("ASP20:OD1,OD2", "resname ASP and resid 20 and name OD1 OD2"),
    ("@CA", "name CA"),
    (":20-30", "resid 20:30"),
    ("TRH:*", "resname TRH"),
    ("protein", "protein"),
    ("{resid 20 and name OD1}", "resid 20 and name OD1"),
    ('"resid 20 and name OD1"', "resid 20 and name OD1"),
])
def test_atom_spec_formats(spec, selection):
    assert resolve_atom_spec(spec)[1] == selection


def test_atom_numbers_are_one_based():
    """@1123 is cpptraj/VMD numbering; MDAnalysis 'index' is 0-based."""
    assert resolve_atom_spec("@1123")[1] == "index 1122"
    assert resolve_atom_spec("@1,2,3")[1] == "index 0 1 2"
    with pytest.raises(ConfigError, match="1-based"):
        resolve_atom_spec("@0")


def test_water_keyword_selects_oxygens():
    selection = resolve_atom_spec("water")[1]
    assert "resname WAT" in selection and "name O" in selection


def test_alias_is_readable():
    assert resolve_atom_spec("ASP20:OD1")[0] == "ASP20_OD1"
    assert resolve_atom_spec("@1123")[0] == "at1123"


def test_user_defined_alias_wins(tmp_path):
    aliases = {"NUC": "resid 20 and name OD1"}
    assert resolve_atom_spec("NUC", aliases) == ("NUC", "resid 20 and name OD1")


# --------------------------------------------------------------------------- #
# file parsing
# --------------------------------------------------------------------------- #
BASIC = """\
# a comment
topology    system.prmtop
trajectory  a.nc b.nc
time        0.1 ps
stride      5
output      out
formats     png pdf

[selections]
NUC  ASP20:OD1

[distances]
d_nuc   NUC          TRH453:C1   threshold=3.5
d_wat   TRH453:O2P   water       mode=min
        ASP20:OD1    ASP20:OD2

[angles]
a1      ASP20:OD1    TRH453:C1   TRH453:O1

[dihedrals]
chi1    20:N  20:CA  20:CB  20:OG
"""


def _write(tmp_path, text=BASIC, name="analysis.in"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_basic_structure(tmp_path):
    parsed = parse_input(_write(tmp_path))
    assert parsed["global"]["topology"] == ["system.prmtop"]
    assert parsed["global"]["trajectory"] == ["a.nc", "b.nc"]
    assert set(parsed["sections"]) == {"selections", "distances", "angles", "dihedrals"}


def test_input_to_config(tmp_path):
    data = input_to_dict(parse_input(_write(tmp_path)))
    config = Config.from_dict(data)

    assert [p.name for p in config.system.trajectory] == ["a.nc", "b.nc"]
    assert config.system.time.dt == 0.1
    assert config.system.frames.stride == 5
    assert config.output.formats == ["png", "pdf"]

    names = [d.name for d in config.distances.pairs]
    assert names[0] == "d_nuc" and names[1] == "d_wat"
    assert config.distances.pairs[0].threshold == 3.5
    assert config.distances.pairs[1].mode == "min"
    assert len(names) == 3                      # the unnamed line is accepted too
    assert [a.name for a in config.angles.angles] == ["a1"]
    assert [d.name for d in config.angles.dihedrals] == ["chi1"]
    # the alias keeps the residue-name check that ASP20:OD1 implies
    assert config.selections["NUC"] == "resname ASP and resid 20 and name OD1"


def test_rmsd_and_rmsf_run_by_default(tmp_path):
    text = "topology s.prmtop\ntrajectory a.nc\n"
    config = Config.from_dict(input_to_dict(parse_input(_write(tmp_path, text))))
    assert config.rmsd.enabled and config.rmsf.enabled
    assert [g.name for g in config.rmsd.groups] == ["backbone"]
    assert config.rmsf.selection == "protein and name CA"


def test_rmsd_and_rmsf_can_be_switched_off(tmp_path):
    text = "topology s.prmtop\ntrajectory a.nc\nrmsd off\nrmsf off\n"
    config = Config.from_dict(input_to_dict(parse_input(_write(tmp_path, text))))
    assert not config.rmsd.enabled
    assert not config.rmsf.enabled


def test_other_analyses_are_opt_in(tmp_path):
    text = "topology s.prmtop\ntrajectory a.nc\n"
    config = Config.from_dict(input_to_dict(parse_input(_write(tmp_path, text))))
    for section in (config.rdf, config.clustering, config.hbonds,
                    config.free_energy, config.rgyr):
        assert not section.enabled


def test_replicas(tmp_path):
    text = ("topology s.prmtop\n"
            "replica rep1 r1/prod.nc\n"
            "replica rep2 r2/prod.nc\n")
    config = Config.from_dict(input_to_dict(parse_input(_write(tmp_path, text))))
    assert list(config.system.replicas) == ["rep1", "rep2"]


def test_rdf_options(tmp_path):
    text = ("topology s.prmtop\ntrajectory a.nc\n\n"
            "[rdf]\n"
            "w1  ASP20:OD1  water  rmax=8 bins=80\n"
            "w2  {resid 20} water  by=residue\n")
    config = Config.from_dict(input_to_dict(parse_input(_write(tmp_path, text))))
    first, second = config.rdf.pairs
    assert first.range == (0.0, 8.0) and first.nbins == 80
    assert first.center == "atom"
    assert second.center == "residue"


def test_hbond_and_fes_sections(tmp_path):
    text = ("topology s.prmtop\ntrajectory a.nc\n\n"
            "[distances]\nd1 ASP20:OD1 ARG414:NH1\nd2 ASP20:OD2 ARG414:NH2\n\n"
            "[hbonds]\nhb1 donor=ARG414:NH1 acceptor=ASP20:OD1\n"
            "auto {resname TRH} protein\n\n"
            "[fes]\nmap1 x=d1 y=d2 bins=40\n")
    config = Config.from_dict(input_to_dict(parse_input(_write(tmp_path, text))))
    assert [h.name for h in config.hbonds.pairs] == ["hb1"]
    assert config.hbonds.auto and len(config.hbonds.auto_between) == 1
    assert config.free_energy.maps[0].bins == 40


def test_unknown_section_is_reported(tmp_path):
    text = "topology s.prmtop\ntrajectory a.nc\n\n[nonsense]\nfoo bar\n"
    with pytest.raises(ConfigError, match="Unknown section"):
        parse_input(_write(tmp_path, text))


def test_missing_topology_is_reported(tmp_path):
    text = "trajectory a.nc\n"
    with pytest.raises(ConfigError, match="topology"):
        input_to_dict(parse_input(_write(tmp_path, text)))


def test_too_few_atoms_for_an_angle(tmp_path):
    text = ("topology s.prmtop\ntrajectory a.nc\n\n"
            "[angles]\na1 ASP20:OD1 TRH453:C1\n")
    with pytest.raises(ConfigError, match="needs 3 atom"):
        input_to_dict(parse_input(_write(tmp_path, text)))


def test_unbalanced_braces(tmp_path):
    text = "topology s.prmtop\ntrajectory a.nc\n\n[distances]\nd1 {resid 20 ASP20:OD2\n"
    with pytest.raises(ConfigError, match="Unbalanced"):
        parse_input(_write(tmp_path, text))


def test_paths_are_relative_to_the_input_file(tmp_path, toy_files):
    topology, trajectory = toy_files
    text = f"topology {topology}\ntrajectory {trajectory}\n"
    config = load_input(_write(tmp_path, text))
    assert config.system.topology.is_file()


# --------------------------------------------------------------------------- #
# end to end on the toy system
# --------------------------------------------------------------------------- #
def test_run_from_input_file(tmp_path, toy_files):
    from md_interactions.runner import run_analyses

    topology, trajectory = toy_files
    text = f"""\
topology   {topology}
trajectory {trajectory}
time       0.01 ns
output     {tmp_path / 'results'}
formats    png
report     off

[distances]
d_nuc  1:OG   LIG:C1   threshold=3.2
d_idx  @4     @26

[angles]
a1     1:OG   LIG:C1   LIG:O1

[rmsf]
selection  protein and name CA
"""
    config = load_input(_write(tmp_path, text))
    output = run_analyses(config, verbose=False, strict=True)

    assert output.ok
    summary = output.summary.set_index("observable")
    assert {"d_nuc", "d_idx", "a1"} <= set(summary.index)
    # @4 and @26 are the 1-based numbers of SER1:OG and LIG4:C1
    assert summary.loc["d_idx", "mean"] == pytest.approx(
        summary.loc["d_nuc", "mean"], abs=1e-6)
    assert np.isfinite(summary.loc["a1", "mean"])
