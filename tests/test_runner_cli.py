"""End-to-end: runner, report generation and command line interface."""

from __future__ import annotations

import yaml

import pytest

from md_interactions import cli
from md_interactions.runner import run_analyses

FULL_ANALYSES = {
    "distances": {"pairs": [
        {"name": "d_nuc", "atoms": ["SER_OG", "LIG_C1"], "threshold": 3.2},
        {"name": "d_acid", "atoms": ["SER_OG", "HIS_NE2"]},
    ]},
    "angles": {"definitions": [
        {"name": "a_attack", "atoms": ["SER_OG", "LIG_C1", "LIG_O1"]},
    ]},
    "rmsd": {"groups": [{"name": "backbone", "selection": "backbone"}]},
    "rmsf": {"enabled": True, "selection": "protein and name CA"},
    "hbonds": {"pairs": [
        {"name": "hb_ser_his", "donor": "SER_OG", "acceptor": "HIS_NE2"},
    ]},
    "free_energy_maps": {"maps": [
        {"name": "fes_dnuc_dacid", "x": "d_nuc", "y": "d_acid", "bins": 15},
    ]},
    "radius_of_gyration": {"groups": [{"name": "rg", "selection": "protein"}]},
    "rdf": {"pairs": [{"name": "rdf_lig_wat", "g1": "LIG_O1", "g2": "WATERS",
                       "nbins": 20, "range": [0.0, 10.0]}]},
    "clustering": {"enabled": True, "selection": "protein", "n_clusters": 2},
}


def test_full_run_produces_everything(make_config):
    config = make_config(FULL_ANALYSES)
    config.report.enabled = True
    config.report.formats = ["markdown", "html"]

    output = run_analyses(config, verbose=False, strict=True)

    assert output.ok
    assert {r.name for r in output.results} == {
        "distances", "angles_dihedrals", "rmsd_rmsf", "hbonds",
        "radius_of_gyration", "rdf", "clustering", "free_energy_maps",
    }
    # observables from different modules end up in a single table
    assert {"d_nuc", "d_acid", "a_attack", "backbone", "rg"} <= set(output.observables.columns)
    assert (output.paths.summary / "summary.csv").is_file()
    assert (output.paths.summary / "summary.md").is_file()
    assert (output.paths.root / "report.md").is_file()

    html = (output.paths.root / "report.html").read_text(encoding="utf-8")
    assert "<img src='plots/" in html
    assert "d_nuc" in html


def test_failure_in_one_analysis_does_not_stop_the_rest(make_config):
    analyses = {
        "distances": {"pairs": [{"name": "d_nuc", "atoms": ["SER_OG", "LIG_C1"]}]},
        "radius_of_gyration": {"groups": [
            {"name": "ghost", "selection": "resname NOPE"},
        ]},
    }
    output = run_analyses(make_config(analyses), verbose=False, strict=False)

    assert not output.ok
    assert [name for name, _msg in output.failures] == ["radius_of_gyration"]
    assert output.result("distances") is not None


def test_strict_mode_raises(make_config):
    analyses = {"radius_of_gyration": {"groups": [
        {"name": "ghost", "selection": "resname NOPE"},
    ]}}
    with pytest.raises(Exception):
        run_analyses(make_config(analyses), verbose=False, strict=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _write_config(path, toy_files, tmp_path, analyses) -> None:
    topology, trajectory = toy_files
    path.write_text(yaml.safe_dump({
        "system": {"topology": str(topology), "trajectory": [str(trajectory)],
                   "time": {"dt": 0.01, "unit": "ns"}},
        "output": {"directory": str(tmp_path / "results"), "formats": ["png"], "dpi": 80},
        "selections": {"SER_OG": "resid 1 and name OG",
                       "LIG_C1": "resname LIG and name C1"},
        "analyses": analyses,
        "report": {"enabled": False},
    }), encoding="utf-8")


def test_cli_run(tmp_path, toy_files, capsys):
    config_file = tmp_path / "config.yaml"
    _write_config(config_file, toy_files, tmp_path, {
        "distances": {"pairs": [{"name": "d_nuc", "atoms": ["SER_OG", "LIG_C1"]}]},
    })
    code = cli.main(["run", "-c", str(config_file), "--stride", "2"])
    assert code == 0
    assert (tmp_path / "results" / "data" / "distances.csv").is_file()


def test_cli_check_reports_bad_selection(tmp_path, toy_files, capsys):
    config_file = tmp_path / "config.yaml"
    _write_config(config_file, toy_files, tmp_path, {
        "distances": {"pairs": [
            {"name": "d_bad", "atoms": ["SER_OG", "resid 999 and name X"]},
        ]},
    })
    code = cli.main(["check", "-c", str(config_file)])
    captured = capsys.readouterr().out
    assert code == 1
    assert "[FAIL]" in captured
    assert "[ok]" in captured


def test_cli_check_ok(tmp_path, toy_files, capsys):
    config_file = tmp_path / "config.yaml"
    _write_config(config_file, toy_files, tmp_path, {
        "distances": {"pairs": [{"name": "d_nuc", "atoms": ["SER_OG", "LIG_C1"]}]},
    })
    assert cli.main(["check", "-c", str(config_file)]) == 0
    assert "All selections resolve" in capsys.readouterr().out


def test_cli_init_writes_valid_template(tmp_path):
    target = tmp_path / "template.yaml"
    assert cli.main(["init", "-o", str(target)]) == 0
    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert "system" in data and "analyses" in data
    # writing twice without --force must fail
    assert cli.main(["init", "-o", str(target)]) == 1
    assert cli.main(["init", "-o", str(target), "--force"]) == 0


def test_cli_bad_config_returns_error_code(tmp_path):
    config_file = tmp_path / "bad.yaml"
    config_file.write_text("system: {topology: x.prmtop}", encoding="utf-8")
    assert cli.main(["run", "-c", str(config_file)]) == 2
