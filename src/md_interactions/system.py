"""Trajectory loading, frame bookkeeping and selection handling.

Everything that touches MDAnalysis' :class:`~MDAnalysis.core.universe.Universe`
goes through :class:`MDSystem`, so the analysis modules only deal with
:class:`~MDAnalysis.core.groups.AtomGroup` objects and a consistent time axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import MDAnalysis as mda
import numpy as np
from MDAnalysis.exceptions import SelectionError as MDASelectionError

from .config import Config
from .exceptions import (
    AtomCountError,
    EmptySelectionError,
    MDInteractionsError,
    SelectionError,
)

__all__ = ["MDSystem", "load_system"]

_PS_PER_UNIT = {"ps": 1.0, "ns": 1000.0, "frame": 1.0}


@dataclass
class MDSystem:
    """A loaded topology + trajectory with a fixed frame selection.

    Parameters
    ----------
    universe
        The MDAnalysis universe (may already be an in-memory, aligned copy).
    config
        The validated configuration; used for selection aliases and time axis.
    frame_indices
        Indices, in the *original* trajectory, of the frames that will be
        analysed.  Kept separately from ``run_kwargs`` because aligning in
        memory renumbers frames.
    run_kwargs
        ``start``/``stop``/``step`` to pass to MDAnalysis ``run()`` calls so
        that every module sees exactly the same frames.
    """

    universe: mda.Universe
    config: Config
    frame_indices: np.ndarray
    run_kwargs: dict[str, int | None] = field(default_factory=dict)
    in_memory: bool = False
    #: Replica name of every analysed frame (``None`` when no replicas declared).
    replica_labels: np.ndarray | None = None

    # -- replicas ---------------------------------------------------------- #
    @property
    def replica_names(self) -> list[str]:
        """Replica names in declaration order (empty when there are none)."""
        if self.replica_labels is None:
            return []
        seen: list[str] = []
        for label in self.replica_labels:
            if label not in seen:
                seen.append(str(label))
        return seen

    def replica_masks(self) -> dict[str, np.ndarray]:
        """``{replica: boolean mask over the analysed frames}``."""
        if self.replica_labels is None:
            return {}
        return {name: self.replica_labels == name for name in self.replica_names}

    # -- basic properties -------------------------------------------------- #
    @property
    def n_frames(self) -> int:
        return int(len(self.frame_indices))

    @property
    def time_unit(self) -> str:
        return self.config.system.time.unit

    @property
    def time_label(self) -> str:
        """Axis label for the time coordinate."""
        return "Frame" if self.time_unit == "frame" else f"Time ({self.time_unit})"

    @property
    def dt_per_frame(self) -> float:
        """Time between *consecutive saved frames* of the input trajectory."""
        cfg_dt = self.config.system.time.dt
        if cfg_dt is not None:
            return float(cfg_dt)
        if self.time_unit == "frame":
            return 1.0
        dt_ps = float(getattr(self.universe.trajectory, "dt", 1.0) or 1.0)
        return dt_ps / _PS_PER_UNIT[self.time_unit]

    @property
    def times(self) -> np.ndarray:
        """Time value of every analysed frame, in ``config.system.time.unit``."""
        if self.time_unit == "frame":
            return self.frame_indices.astype(float)
        return self.frame_indices.astype(float) * self.dt_per_frame

    def time_frame_columns(self) -> dict[str, np.ndarray]:
        """The leading columns shared by every time-series CSV.

        ``replica`` is added whenever replicas were declared, so every CSV
        keeps track of which run each frame came from.
        """
        columns: dict[str, np.ndarray] = {
            "frame": self.frame_indices,
            "time": self.times,
        }
        if self.replica_labels is not None:
            columns["replica"] = self.replica_labels
        return columns

    def goto(self, position: int) -> None:
        """Seek to the ``position``-th *analysed* frame (0-based).

        Handles the renumbering introduced when the trajectory was transferred
        to memory for alignment, so ``position`` always refers to the same
        frame as ``frame_indices[position]``.
        """
        if not 0 <= position < self.n_frames:
            raise IndexError(
                f"Frame position {position} out of range (0..{self.n_frames - 1})."
            )
        start = self.run_kwargs.get("start", 0) or 0
        step = self.run_kwargs.get("step", 1) or 1
        self.universe.trajectory[start + position * step]

    def iter_frames(self) -> Iterator[mda.coordinates.base.Timestep]:
        """Iterate over the analysed frames of the trajectory."""
        start = self.run_kwargs.get("start", 0) or 0
        stop = self.run_kwargs.get("stop")
        step = self.run_kwargs.get("step", 1) or 1
        for ts in self.universe.trajectory[start:stop:step]:
            yield ts

    # -- selections -------------------------------------------------------- #
    def select(
        self,
        token: str,
        *,
        name: str | None = None,
        expected: int | None = None,
    ) -> mda.core.groups.AtomGroup:
        """Resolve a selection alias (or raw selection string) to an AtomGroup.

        Parameters
        ----------
        token
            Either a key of the YAML ``selections:`` block or a raw MDAnalysis
            selection string.
        name
            Label used in error messages (defaults to ``token``).
        expected
            If given, the number of atoms the selection must match.

        Raises
        ------
        EmptySelectionError
            The selection matched no atoms.
        AtomCountError
            The selection matched a number of atoms different from ``expected``.
        SelectionError
            The selection string could not be parsed by MDAnalysis.
        """
        label = name or token
        selection = self.config.resolve_selection(token)
        try:
            group = self.universe.select_atoms(selection)
        except MDASelectionError as exc:
            raise SelectionError(
                f"Invalid selection '{label}' -> \"{selection}\": {exc}"
            ) from exc
        if group.n_atoms == 0:
            raise EmptySelectionError(label, selection)
        if expected is not None and group.n_atoms != expected:
            raise AtomCountError(label, selection, group.n_atoms, f"{expected} atom(s)")
        return group

    def describe_selection(self, token: str) -> str:
        """Human readable description, e.g. ``SER145:OG``."""
        group = self.select(token)
        if group.n_atoms == 1:
            atom = group[0]
            return f"{atom.resname}{atom.resid}:{atom.name}"
        return f"{token} ({group.n_atoms} atoms)"


def _check_files(config: Config) -> None:
    missing = [p for p in [config.system.topology, *config.system.trajectory]
               if not Path(p).is_file()]
    if missing:
        listed = "\n  ".join(str(p) for p in missing)
        raise MDInteractionsError(f"Input file(s) not found:\n  {listed}")


def load_system(config: Config, verbose: bool = True) -> MDSystem:
    """Build an :class:`MDSystem` from a validated :class:`~.config.Config`.

    Handles multi-file trajectories (they are concatenated by MDAnalysis'
    ``ChainReader``), the requested frame range/stride and, if requested, a
    global RMSD fit performed in memory.
    """
    _check_files(config)

    try:
        universe = mda.Universe(
            str(config.system.topology),
            *[str(p) for p in config.system.trajectory],
        )
    except Exception as exc:  # pragma: no cover - depends on user files
        raise MDInteractionsError(
            f"Could not load topology/trajectory: {exc}\n"
            "  Check that the topology matches the trajectory (same atom count) "
            "and that the file extensions are recognised by MDAnalysis "
            "(.prmtop/.parm7 + .nc/.dcd)."
        ) from exc

    n_total = len(universe.trajectory)
    frames_cfg = config.system.frames
    indices = np.arange(n_total)[frames_cfg.start:frames_cfg.stop:frames_cfg.stride]
    if indices.size == 0:
        raise MDInteractionsError(
            f"The requested frame range (start={frames_cfg.start}, "
            f"stop={frames_cfg.stop}, stride={frames_cfg.stride}) selects 0 frames "
            f"out of {n_total} available."
        )

    system = MDSystem(
        universe=universe,
        config=config,
        frame_indices=indices,
        run_kwargs=frames_cfg.as_slice_kwargs(),
        replica_labels=_replica_labels(universe, config, indices),
    )

    if config.system.align.enabled:
        system = _align_in_memory(system, verbose=verbose)

    if verbose:
        print(
            f"[md_interactions] {universe.atoms.n_atoms} atoms, "
            f"{n_total} frames in trajectory, {system.n_frames} analysed "
            f"(start={frames_cfg.start}, stop={frames_cfg.stop}, "
            f"stride={frames_cfg.stride})"
        )
        if system.replica_labels is not None:
            counts = {name: int(mask.sum())
                      for name, mask in system.replica_masks().items()}
            listed = ", ".join(f"{name}: {n}" for name, n in counts.items())
            print(f"[md_interactions] replicas ({listed} frames)")
    return system


def _replica_labels(
    universe: mda.Universe, config: Config, indices: np.ndarray
) -> np.ndarray | None:
    """Label every analysed frame with the replica its file belongs to."""
    replicas = config.system.replicas
    if not replicas:
        return None

    reader = universe.trajectory
    readers = getattr(reader, "readers", None)
    per_file = ([int(r.n_frames) for r in readers] if readers is not None
                else [int(len(reader))])
    expected = sum(len(paths) for paths in replicas.values())
    if len(per_file) != expected:
        raise MDInteractionsError(
            f"Cannot assign frames to replicas: the configuration lists "
            f"{expected} trajectory file(s) but MDAnalysis opened "
            f"{len(per_file)}."
        )

    labels: list[str] = []
    cursor = 0
    for name, paths in replicas.items():
        for _path in paths:
            labels.extend([name] * per_file[cursor])
            cursor += 1
    return np.asarray(labels, dtype=object)[indices]


def _align_in_memory(system: MDSystem, verbose: bool = True) -> MDSystem:
    """Apply a global RMSD fit, keeping the result in memory.

    Only the analysed frames are transferred, so the memory cost is
    ``n_frames * n_atoms * 3 * 4`` bytes.
    """
    from MDAnalysis.analysis import align

    cfg = system.config.system.align
    universe = system.universe
    try:
        universe.transfer_to_memory(**system.run_kwargs)
    except MemoryError as exc:  # pragma: no cover - hardware dependent
        raise MDInteractionsError(
            "Not enough memory to align the trajectory in memory. "
            "Increase 'system.frames.stride' or set 'system.align.enabled: false' "
            "(distances, angles and dihedrals do not need alignment)."
        ) from exc

    if cfg.reference is not None:
        reference = mda.Universe(str(system.config.system.topology), str(cfg.reference))
    else:
        reference = universe.copy()
        reference.trajectory[0]

    system.run_kwargs = {"start": 0, "stop": None, "step": 1}
    system.in_memory = True

    # Validate the selection on both universes before the (costly) fit.
    system.select(cfg.selection, name="system.align.selection")
    align.AlignTraj(universe, reference, select=cfg.selection, in_memory=True).run()
    if verbose:
        print(f"[md_interactions] trajectory aligned on \"{cfg.selection}\" (in memory)")
    return system
