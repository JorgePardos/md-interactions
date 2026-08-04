"""The pre-flight check that ``run`` performs before touching the trajectory."""

from __future__ import annotations

import yaml

import pytest

from md_interactions import cli
from md_interactions.system import load_system
from md_interactions.validation import (
    expected_observables,
    selection_tokens,
    validate,
)


def _analyses(bad_selection: bool = False, bad_map: bool = False) -> dict:
    analyses: dict = {
        "distances": {"pairs": [
            {"name": "d_nuc", "atoms": ["SER_OG", "LIG_C1"]},
            {"name": "d_acid", "atoms": ["SER_OG", "HIS_NE2"]},
        ]},
    }
    if bad_selection:
        analyses["distances"]["pairs"].append(
            {"name": "d_bad", "atoms": ["SER_OG", "resid 999 and name ZZ"]}
        )
    if bad_map:
        analyses["free_energy_maps"] = {"maps": [
            {"name": "fes", "x": "d_nuc", "y": "d_nuk"},      # typo
        ]}
    return analyses


# --------------------------------------------------------------------------- #
# what the configuration refers to
# --------------------------------------------------------------------------- #
def test_selection_tokens_cover_every_section(make_config):
    config = make_config({
        "distances": {"pairs": [{"name": "d", "atoms": ["SER_OG", "LIG_C1"]}]},
        "rmsd": {"groups": [{"name": "bb", "selection": "backbone",
                             "superposition": "protein"}]},
        "rdf": {"pairs": [{"name": "r", "g1": "SER_OG", "g2": "WATERS"}]},
        "clustering": {"enabled": True, "selection": "protein"},
    })
    contexts = {context for context, _token in selection_tokens(config)}
    assert {"distance:d", "rmsd:bb", "rmsd:bb.fit", "rdf:r.g1", "rdf:r.g2",
            "clustering"} <= contexts


def test_expected_observables(make_config):
    config = make_config({
        "distances": {"pairs": [{"name": "d_nuc", "atoms": ["SER_OG", "LIG_C1"]}]},
        "angles": {"definitions": [
            {"name": "a1", "atoms": ["SER_OG", "LIG_C1", "LIG_O1"]}]},
        "hbonds": {"pairs": [
            {"name": "hb", "donor": "SER_OG", "acceptor": "HIS_NE2"}]},
    })
    names = expected_observables(config)
    assert {"d_nuc", "a1", "hb", "hb_angle"} <= names


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def test_valid_configuration_passes(make_config):
    config = make_config(_analyses())
    system = load_system(config, verbose=False)
    report = validate(config, system=system)
    assert report.ok
    assert len(report.resolved) >= 4


def test_bad_selection_is_caught(make_config):
    config = make_config(_analyses(bad_selection=True))
    system = load_system(config, verbose=False)
    report = validate(config, system=system)
    assert not report.ok
    issue = report.issues[0]
    assert issue.context.startswith("distance:d_bad")
    assert "0 atoms" in issue.detail


def test_map_pointing_at_a_missing_observable_is_caught(make_config):
    """The failure mode that used to appear only after every trajectory pass."""
    config = make_config(_analyses(bad_map=True))
    system = load_system(config, verbose=False)
    report = validate(config, system=system)
    assert not report.ok
    detail = report.issues[0].detail
    assert "d_nuk" in detail
    assert "did you mean 'd_nuc'" in detail        # the typo gets a suggestion


# --------------------------------------------------------------------------- #
# the CLI wiring
# --------------------------------------------------------------------------- #
def _write_config(path, toy_files, tmp_path, analyses) -> None:
    topology, trajectory = toy_files
    path.write_text(yaml.safe_dump({
        "system": {"topology": str(topology), "trajectory": [str(trajectory)],
                   "time": {"dt": 0.01, "unit": "ns"}},
        "output": {"directory": str(tmp_path / "results"), "formats": ["png"],
                   "dpi": 80},
        "selections": {"SER_OG": "resid 1 and name OG",
                       "HIS_NE2": "resid 2 and name NE2",
                       "LIG_C1": "resname LIG and name C1"},
        "analyses": analyses,
        "report": {"enabled": False},
    }), encoding="utf-8")


def test_run_aborts_before_doing_any_work(tmp_path, toy_files, capsys):
    config_file = tmp_path / "config.yaml"
    _write_config(config_file, toy_files, tmp_path, _analyses(bad_selection=True))

    code = cli.main(["run", "-c", str(config_file)])
    captured = capsys.readouterr()

    assert code == 1
    assert "[FAIL]" in captured.err
    assert "Nothing was run" in captured.err
    # the analysis never started: no results were written
    assert not (tmp_path / "results" / "data").exists()


def test_run_aborts_on_a_bad_free_energy_map(tmp_path, toy_files, capsys):
    config_file = tmp_path / "config.yaml"
    _write_config(config_file, toy_files, tmp_path, _analyses(bad_map=True))

    assert cli.main(["run", "-c", str(config_file)]) == 1
    assert "d_nuk" in capsys.readouterr().err
    assert not (tmp_path / "results" / "data").exists()


def test_no_check_skips_the_preflight(tmp_path, toy_files, capsys):
    """--no-check lets the run start; the bad analysis then fails on its own."""
    config_file = tmp_path / "config.yaml"
    _write_config(config_file, toy_files, tmp_path, _analyses(bad_selection=True))

    code = cli.main(["run", "-c", str(config_file), "--no-check"])
    captured = capsys.readouterr()

    assert code == 1
    assert "Nothing was run" not in captured.err
    assert "pre-flight" not in captured.out


def test_valid_run_reports_the_preflight_and_completes(tmp_path, toy_files, capsys):
    config_file = tmp_path / "config.yaml"
    _write_config(config_file, toy_files, tmp_path, _analyses())

    code = cli.main(["run", "-c", str(config_file)])
    captured = capsys.readouterr()

    assert code == 0
    assert "pre-flight check" in captured.out
    assert (tmp_path / "results" / "data" / "distances.csv").is_file()


def test_topology_is_read_only_once(tmp_path, toy_files, monkeypatch):
    """The pre-flight reuses its system instead of loading the topology twice."""
    import md_interactions.runner as runner

    config_file = tmp_path / "config.yaml"
    _write_config(config_file, toy_files, tmp_path, _analyses())

    calls = []
    original = runner.load_system

    def counting(config, verbose=True):
        calls.append(1)
        return original(config, verbose=verbose)

    monkeypatch.setattr(runner, "load_system", counting)
    assert cli.main(["run", "-c", str(config_file)]) == 0
    assert calls == []          # run_analyses got the system from the CLI
