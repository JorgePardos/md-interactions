"""Shared fixtures: a tiny synthetic system written to disk once per session."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from md_interactions.config import Config
from md_interactions.io_utils import prepare_output
from md_interactions.system import load_system
from md_interactions.testing import write_toy_system

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

N_FRAMES = 40

BASE_SELECTIONS = {
    "SER_OG": "resid 1 and name OG",
    "HIS_NE2": "resid 2 and name NE2",
    "ASP_OD2": "resid 3 and name OD2",
    "LIG_C1": "resname LIG and name C1",
    "LIG_O1": "resname LIG and name O1",
    "WATERS": "resname WAT and name O",
}


@pytest.fixture(scope="session")
def toy_files(tmp_path_factory) -> tuple[Path, Path]:
    """(topology, trajectory) of the toy system, written once per test session."""
    directory = tmp_path_factory.mktemp("toy_system")
    return write_toy_system(directory, n_frames=N_FRAMES)


@pytest.fixture
def make_config(toy_files, tmp_path):
    """Factory building a :class:`Config` for the toy system.

    ``make_config(analyses={...}, **system_overrides)`` returns a fully
    validated configuration whose output directory is inside ``tmp_path``.
    """
    topology, trajectory = toy_files

    def _factory(analyses: dict | None = None, **overrides) -> Config:
        data = {
            "system": {
                "topology": str(topology),
                "trajectory": [str(trajectory)],
                "time": {"dt": 0.01, "unit": "ns"},
                "frames": overrides.pop("frames", {"stride": 1}),
                **overrides,
            },
            "output": {
                "directory": str(tmp_path / "results"),
                "formats": ["png"],
                "dpi": 80,
            },
            "selections": dict(BASE_SELECTIONS),
            "analyses": analyses or {},
            "report": {"enabled": False},
        }
        return Config.from_dict(data)

    return _factory


@pytest.fixture
def toy_system(make_config):
    """A loaded :class:`MDSystem` with no analysis enabled."""
    return load_system(make_config(), verbose=False)


@pytest.fixture
def out_paths(tmp_path):
    """A ``results/`` tree inside the test's temporary directory."""
    return prepare_output(tmp_path / "results")
