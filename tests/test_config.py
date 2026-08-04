"""Configuration parsing, validation and error messages."""

from __future__ import annotations

import pytest
import yaml

from md_interactions.config import Config, load_config
from md_interactions.exceptions import ConfigError


def _minimal(tmp_path) -> dict:
    return {
        "system": {"topology": "sys.prmtop", "trajectory": ["a.nc", "b.nc"]},
    }


def test_minimal_config_defaults(tmp_path):
    config = Config.from_dict(_minimal(tmp_path))
    assert config.system.trajectory[1].name == "b.nc"
    assert config.system.frames.stride == 1
    assert config.output.formats == ["png", "pdf"]
    assert config.enabled_analyses() == []


def test_unknown_key_is_rejected():
    data = _minimal(None)
    data["analyses"] = {"distances": {"enabled": True, "pares": []}}
    with pytest.raises(ConfigError, match="Unknown key"):
        Config.from_dict(data)


def test_missing_trajectory_is_rejected():
    with pytest.raises(ConfigError, match="trajectory"):
        Config.from_dict({"system": {"topology": "sys.prmtop"}})


def test_wrong_number_of_atoms_in_angle():
    data = _minimal(None)
    data["analyses"] = {
        "angles": {"definitions": [{"name": "a", "atoms": ["x", "y"]}]}
    }
    with pytest.raises(ConfigError, match="exactly 3"):
        Config.from_dict(data)


def test_duplicated_observable_names():
    data = _minimal(None)
    data["analyses"] = {
        "distances": {"pairs": [{"name": "d", "atoms": ["a", "b"]}]},
        "angles": {"definitions": [{"name": "d", "atoms": ["a", "b", "c"]}]},
    }
    with pytest.raises(ConfigError, match="Duplicated observable name"):
        Config.from_dict(data)


def test_selection_alias_resolution():
    data = _minimal(None)
    data["selections"] = {"OG": "resid 1 and name OG"}
    config = Config.from_dict(data)
    assert config.resolve_selection("OG") == "resid 1 and name OG"
    assert config.resolve_selection("protein") == "protein"


def test_paths_are_relative_to_the_yaml_file(tmp_path):
    (tmp_path / "data").mkdir()
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump({
        "system": {"topology": "data/sys.prmtop", "trajectory": ["data/traj.nc"]},
        "output": {"directory": "out"},
    }), encoding="utf-8")

    config = load_config(config_file)
    assert config.system.topology == (tmp_path / "data" / "sys.prmtop").resolve()
    assert config.output.directory == (tmp_path / "out").resolve()


def test_cli_overrides(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump({
        "system": {"topology": "a.prmtop", "trajectory": ["a.nc"],
                   "frames": {"stride": 1}},
    }), encoding="utf-8")

    config = load_config(config_file, overrides={"stride": 10, "trajectory": ["b.dcd"]})
    assert config.system.frames.stride == 10
    assert config.system.trajectory[0].name == "b.dcd"


def test_disabled_when_no_entries():
    data = _minimal(None)
    data["analyses"] = {"distances": {"enabled": True, "pairs": []}}
    assert Config.from_dict(data).distances.enabled is False


def test_free_energy_defaults_propagate_to_maps():
    data = _minimal(None)
    data["analyses"] = {
        "free_energy_maps": {
            "temperature": 310.0,
            "bins": 25,
            "maps": [{"name": "m", "x": "d1", "y": "d2"}],
        }
    }
    spec = Config.from_dict(data).free_energy.maps[0]
    assert spec.temperature == 310.0
    assert spec.bins == 25
