"""Loading, frame bookkeeping, time axis and selection error handling."""

from __future__ import annotations

import numpy as np
import pytest

from md_interactions.exceptions import (
    AtomCountError,
    EmptySelectionError,
    MDInteractionsError,
    SelectionError,
)
from md_interactions.system import load_system

from conftest import N_FRAMES


def test_frames_and_time_axis(toy_system):
    assert toy_system.n_frames == N_FRAMES
    assert toy_system.universe.atoms.n_atoms == 42
    assert toy_system.time_unit == "ns"
    np.testing.assert_allclose(toy_system.times[:3], [0.0, 0.01, 0.02])
    assert toy_system.time_label == "Time (ns)"


def test_stride_and_range(make_config):
    system = load_system(make_config(frames={"start": 5, "stop": 25, "stride": 5}),
                         verbose=False)
    np.testing.assert_array_equal(system.frame_indices, [5, 10, 15, 20])
    assert system.n_frames == 4
    # iter_frames must visit exactly those frames
    visited = [ts.frame for ts in system.iter_frames()]
    assert visited == [5, 10, 15, 20]


def test_goto_matches_frame_indices(toy_system):
    toy_system.goto(7)
    assert toy_system.universe.trajectory.frame == toy_system.frame_indices[7]


def test_empty_selection_error_mentions_the_selection(toy_system):
    with pytest.raises(EmptySelectionError) as excinfo:
        toy_system.select("resid 999 and name OG", name="d_test")
    assert "0 atoms" in str(excinfo.value)
    assert "d_test" in str(excinfo.value)


def test_invalid_selection_syntax(toy_system):
    with pytest.raises(SelectionError):
        toy_system.select("resid and and name", name="bad")


def test_atom_count_error(toy_system):
    with pytest.raises(AtomCountError):
        toy_system.select("resname WAT and name O", name="waters", expected=1)


def test_missing_file_gives_clear_error(make_config):
    config = make_config()
    config.system.topology = config.system.topology.with_name("does_not_exist.prmtop")
    with pytest.raises(MDInteractionsError, match="not found"):
        load_system(config, verbose=False)


def test_empty_frame_range(make_config):
    config = make_config(frames={"start": 500, "stride": 1})
    with pytest.raises(MDInteractionsError, match="0 frames"):
        load_system(config, verbose=False)


def test_alignment_in_memory(make_config):
    config = make_config(align={"enabled": True, "selection": "protein and name CA"},
                         frames={"stride": 4})
    system = load_system(config, verbose=False)
    assert system.in_memory is True
    assert system.n_frames == len(range(0, N_FRAMES, 4))
    # frame indices still refer to the original trajectory
    assert system.frame_indices[1] == 4
    assert [ts.frame for ts in system.iter_frames()] == list(range(system.n_frames))


def test_describe_selection(toy_system):
    assert toy_system.describe_selection("SER_OG") == "SER1:OG"
