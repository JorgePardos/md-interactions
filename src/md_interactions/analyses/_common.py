"""Helpers shared by the geometric analysis modules."""

from __future__ import annotations

from dataclasses import dataclass

import MDAnalysis as mda
import numpy as np

from ..exceptions import AtomCountError
from ..system import MDSystem

__all__ = ["GeometryGroup", "resolve_group", "current_box", "safe_name", "basename",
           "replica_groups"]


@dataclass
class GeometryGroup:
    """A selection together with the way it is reduced to a single point.

    ``mode``:

    ``atom``
        The selection must match exactly one atom (default).
    ``com``
        The centre of mass of the group is used (centre of geometry if the
        topology carries no masses).
    ``min``
        Only meaningful for distances: the minimum distance between the two
        groups is measured instead of a centre-to-centre distance.
    """

    name: str
    token: str
    atoms: mda.core.groups.AtomGroup
    mode: str = "atom"

    @property
    def n_atoms(self) -> int:
        return self.atoms.n_atoms

    def position(self) -> np.ndarray:
        """Representative position of the group at the current frame."""
        if self.mode == "atom":
            return self.atoms.positions[0]
        if self.mode == "com":
            try:
                return self.atoms.center_of_mass()
            except (mda.exceptions.NoDataError, AttributeError):
                return self.atoms.center_of_geometry()
        return self.atoms.center_of_geometry()

    def describe(self) -> str:
        if self.n_atoms == 1:
            atom = self.atoms[0]
            return f"{atom.resname}{atom.resid}:{atom.name}"
        suffix = {"com": "COM", "min": "min-dist"}.get(self.mode, "group")
        return f"{self.token} [{self.n_atoms} atoms, {suffix}]"


def resolve_group(
    system: MDSystem, token: str, mode: str, observable: str
) -> GeometryGroup:
    """Resolve one selection of an observable, enforcing the mode's arity."""
    atoms = system.select(token, name=f"{observable}:{token}")
    if mode == "atom" and atoms.n_atoms != 1:
        raise AtomCountError(
            f"{observable}:{token}",
            system.config.resolve_selection(token),
            atoms.n_atoms,
            "exactly 1 atom (mode: atom)",
        )
    return GeometryGroup(name=f"{observable}:{token}", token=token, atoms=atoms, mode=mode)


def current_box(system: MDSystem, pbc: bool) -> np.ndarray | None:
    """Box vector to pass to MDAnalysis distance functions (``None`` if unused).

    Returns ``None`` when PBC handling is disabled or when the trajectory
    carries no valid box (e.g. gas-phase or in-vacuo QM/MM snapshots).
    """
    if not pbc:
        return None
    box = system.universe.dimensions
    if box is None:
        return None
    box = np.asarray(box, dtype=np.float32)
    if box.shape != (6,) or not np.all(np.isfinite(box)) or np.any(box[:3] <= 0):
        return None
    return box


def safe_name(name: str) -> str:
    """Sanitise an observable name so it can be used in a file name."""
    keep = [c if (c.isalnum() or c in "-_.") else "_" for c in str(name)]
    return "".join(keep).strip("_") or "unnamed"


def replica_groups(system: MDSystem, df, columns: list[str]):
    """Split a time-series table by replica, for the per-replica plot curves.

    Returns ``None`` when no replicas were declared, so callers can simply
    forward the result to the plotting helpers.
    """
    masks = system.replica_masks()
    if not masks:
        return None
    return {
        name: {column: df[column].to_numpy()[mask] for column in columns}
        for name, mask in masks.items()
    }


def basename(prefix: str, name: str, suffix: str = "") -> str:
    """Build a file name, avoiding ``fes_fes_x`` style duplication."""
    clean = safe_name(name)
    stem = clean if clean.startswith(f"{prefix}_") or clean == prefix else f"{prefix}_{clean}"
    return f"{stem}{suffix}"
