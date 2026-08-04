"""A tiny synthetic enzyme-like system, used by the test suite and for demos.

The system is *not* chemically meaningful: it is a 5-residue cartoon (Ser/His/
Asp triad + a ligand + a few waters) whose coordinates oscillate so that every
analysis has something to measure.  Its purpose is to exercise the code paths
without shipping a multi-megabyte trajectory.

::

    import md_interactions as mdi
    u = mdi.testing.make_toy_universe(n_frames=100)
    top, traj = mdi.testing.write_toy_system("/tmp/toy")
"""

from __future__ import annotations

from pathlib import Path

import MDAnalysis as mda
import numpy as np
from MDAnalysis.core.topologyattrs import Bonds

__all__ = ["make_toy_universe", "write_toy_system", "TOY_SELECTIONS"]

# Selections that are guaranteed to work on the toy system.
TOY_SELECTIONS: dict[str, str] = {
    "SER_OG": "resid 1 and name OG",
    "SER_HG": "resid 1 and name HG",
    "HIS_NE2": "resid 2 and name NE2",
    "HIS_HE2": "resid 2 and name HE2",
    "ASP_OD2": "resid 3 and name OD2",
    "LIG_C1": "resname LIG and name C1",
    "LIG_O1": "resname LIG and name O1",
    "WATERS": "resname WAT and name O",
}

# (resname, [(atom name, element), ...])
_RESIDUES: list[tuple[str, list[tuple[str, str]]]] = [
    ("SER", [("N", "N"), ("CA", "C"), ("CB", "C"), ("OG", "O"), ("HG", "H"),
             ("C", "C"), ("O", "O")]),
    ("HIE", [("N", "N"), ("CA", "C"), ("CB", "C"), ("CG", "C"), ("ND1", "N"),
             ("CE1", "C"), ("NE2", "N"), ("HE2", "H"), ("C", "C"), ("O", "O")]),
    ("ASP", [("N", "N"), ("CA", "C"), ("CB", "C"), ("CG", "C"), ("OD1", "O"),
             ("OD2", "O"), ("C", "C"), ("O", "O")]),
    ("LIG", [("C1", "C"), ("O1", "O"), ("O2", "O"), ("C2", "C"), ("H1", "H")]),
    ("WAT", [("O", "O"), ("H1", "H"), ("H2", "H")]),
    ("WAT", [("O", "O"), ("H1", "H"), ("H2", "H")]),
    ("WAT", [("O", "O"), ("H1", "H"), ("H2", "H")]),
    ("WAT", [("O", "O"), ("H1", "H"), ("H2", "H")]),
]

_MASSES = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999}
_CHARGES = {"H": 0.42, "C": 0.1, "N": -0.55, "O": -0.60}


def _base_positions() -> np.ndarray:
    """Reference coordinates (Å) with a sensible Ser-His-Asp/ligand geometry."""
    pos: dict[tuple[int, str], np.ndarray] = {}

    def put(resindex: int, name: str, xyz) -> None:
        pos[(resindex, name)] = np.asarray(xyz, dtype=float)

    # Catalytic triad, roughly collinear along x.
    put(0, "OG", [0.0, 0.0, 0.0])
    put(0, "HG", [0.90, 0.30, 0.0])          # points towards His NE2
    put(0, "CB", [-1.2, -0.9, 0.2])
    put(0, "CA", [-2.4, -0.2, 0.6])
    put(0, "N", [-3.4, -1.0, 1.2])
    put(0, "C", [-3.0, 0.6, -0.6])
    put(0, "O", [-3.8, 1.5, -0.4])

    put(1, "NE2", [2.90, 0.10, 0.0])
    put(1, "HE2", [3.85, 0.35, 0.15])        # points towards Asp OD2
    put(1, "CE1", [3.60, -1.05, -0.20])
    put(1, "ND1", [4.90, -0.95, -0.15])
    put(1, "CG", [3.20, 1.20, 0.25])
    put(1, "CB", [2.60, 2.50, 0.55])
    put(1, "CA", [3.40, 3.60, 1.10])
    put(1, "N", [4.60, 3.90, 0.40])
    put(1, "C", [2.60, 4.85, 1.30])
    put(1, "O", [1.55, 4.95, 0.70])

    put(2, "OD2", [5.85, 0.60, 0.10])
    put(2, "OD1", [6.40, -1.40, 0.30])
    put(2, "CG", [6.55, -0.20, 0.15])
    put(2, "CB", [7.95, 0.25, 0.05])
    put(2, "CA", [8.95, -0.85, 0.35])
    put(2, "N", [9.20, -1.05, 1.75])
    put(2, "C", [10.25, -0.55, -0.35])
    put(2, "O", [10.35, 0.55, -0.90])

    # Ligand: C1 sits above Ser OG (nucleophilic attack coordinate).
    put(3, "C1", [0.0, 3.20, 0.0])
    put(3, "O1", [1.10, 3.75, 0.10])
    put(3, "O2", [-1.05, 3.90, -0.35])
    put(3, "C2", [0.05, 1.75, 2.60])
    put(3, "H1", [0.85, 1.40, 3.10])

    # Waters around the site.
    water_centres = [
        [-2.20, 2.60, 2.20],
        [2.10, 2.40, -2.60],
        [-1.80, -2.90, 2.40],
        [4.80, 2.70, -2.10],
    ]
    for i, centre in enumerate(water_centres):
        centre = np.asarray(centre, dtype=float)
        put(4 + i, "O", centre)
        put(4 + i, "H1", centre + [0.76, 0.59, 0.0])
        put(4 + i, "H2", centre + [-0.76, 0.59, 0.0])

    coords = []
    for resindex, (_resname, atoms) in enumerate(_RESIDUES):
        for name, _element in atoms:
            coords.append(pos[(resindex, name)])
    return np.asarray(coords, dtype=np.float32)


def _bond_list() -> list[tuple[int, int]]:
    """Intra-residue bonds (enough for hydrogen/donor detection)."""
    offsets: list[int] = []
    running = 0
    for _resname, atoms in _RESIDUES:
        offsets.append(running)
        running += len(atoms)

    def idx(resindex: int, name: str) -> int:
        names = [n for n, _e in _RESIDUES[resindex][1]]
        return offsets[resindex] + names.index(name)

    bonds: list[tuple[int, int]] = [
        (idx(0, "N"), idx(0, "CA")), (idx(0, "CA"), idx(0, "CB")),
        (idx(0, "CB"), idx(0, "OG")), (idx(0, "OG"), idx(0, "HG")),
        (idx(0, "CA"), idx(0, "C")), (idx(0, "C"), idx(0, "O")),
        (idx(1, "N"), idx(1, "CA")), (idx(1, "CA"), idx(1, "CB")),
        (idx(1, "CB"), idx(1, "CG")), (idx(1, "CG"), idx(1, "ND1")),
        (idx(1, "ND1"), idx(1, "CE1")), (idx(1, "CE1"), idx(1, "NE2")),
        (idx(1, "NE2"), idx(1, "HE2")), (idx(1, "NE2"), idx(1, "CG")),
        (idx(1, "CA"), idx(1, "C")), (idx(1, "C"), idx(1, "O")),
        (idx(2, "N"), idx(2, "CA")), (idx(2, "CA"), idx(2, "CB")),
        (idx(2, "CB"), idx(2, "CG")), (idx(2, "CG"), idx(2, "OD1")),
        (idx(2, "CG"), idx(2, "OD2")), (idx(2, "CA"), idx(2, "C")),
        (idx(2, "C"), idx(2, "O")),
        (idx(3, "C1"), idx(3, "O1")), (idx(3, "C1"), idx(3, "O2")),
        (idx(3, "C1"), idx(3, "C2")), (idx(3, "C2"), idx(3, "H1")),
    ]
    for res in range(4, 8):
        bonds += [(idx(res, "O"), idx(res, "H1")), (idx(res, "O"), idx(res, "H2"))]
    # peptide bonds
    bonds += [(idx(0, "C"), idx(1, "N")), (idx(1, "C"), idx(2, "N"))]
    return bonds


def make_toy_universe(n_frames: int = 60, seed: int = 0, dt: float = 1.0) -> mda.Universe:
    """Build a small in-memory universe with an oscillating reaction geometry.

    Parameters
    ----------
    n_frames
        Number of frames of the synthetic trajectory.
    seed
        Seed of the random noise added to every frame.
    dt
        Time between frames, in ps (stored in the trajectory).
    """
    names: list[str] = []
    elements: list[str] = []
    resnames: list[str] = []
    resids: list[int] = []
    atom_resindex: list[int] = []
    for resindex, (resname, atoms) in enumerate(_RESIDUES):
        resnames.append(resname)
        resids.append(resindex + 1)
        for name, element in atoms:
            names.append(name)
            elements.append(element)
            atom_resindex.append(resindex)

    n_atoms = len(names)
    universe = mda.Universe.empty(
        n_atoms=n_atoms,
        n_residues=len(_RESIDUES),
        atom_resindex=np.asarray(atom_resindex),
        residue_segindex=np.zeros(len(_RESIDUES), dtype=int),
        trajectory=True,
    )
    universe.add_TopologyAttr("names", names)
    universe.add_TopologyAttr("types", elements)
    universe.add_TopologyAttr("elements", elements)
    universe.add_TopologyAttr("resnames", resnames)
    universe.add_TopologyAttr("resids", resids)
    universe.add_TopologyAttr("segids", ["SYS"])
    universe.add_TopologyAttr("masses", [_MASSES[e] for e in elements])
    universe.add_TopologyAttr("charges", [_CHARGES[e] for e in elements])
    universe.add_TopologyAttr(Bonds(_bond_list()))

    base = _base_positions()
    rng = np.random.default_rng(seed)
    coords = np.empty((n_frames, n_atoms, 3), dtype=np.float32)
    ligand = np.array([i for i, ri in enumerate(atom_resindex) if ri == 3])
    phase = np.linspace(0.0, 4.0 * np.pi, n_frames)

    for frame in range(n_frames):
        snapshot = base + rng.normal(0.0, 0.06, size=base.shape).astype(np.float32)
        # Slow breathing of the attack distance + a global wobble.
        snapshot[ligand, 1] += 0.55 * np.sin(phase[frame])
        snapshot[:, 0] += 0.10 * np.cos(0.5 * phase[frame])
        coords[frame] = snapshot

    box = np.array([40.0, 40.0, 40.0, 90.0, 90.0, 90.0], dtype=np.float32)
    universe.load_new(coords, order="fac", dt=dt,
                      dimensions=np.tile(box, (n_frames, 1)))
    return universe


def write_toy_system(
    directory: str | Path,
    n_frames: int = 60,
    seed: int = 0,
    traj_format: str = "dcd",
) -> tuple[Path, Path]:
    """Write the toy system to disk as ``toy.pdb`` + ``toy.<traj_format>``.

    Returns the ``(topology, trajectory)`` paths, ready to be fed to the CLI
    exactly like a real ``prmtop``/``nc`` pair.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    universe = make_toy_universe(n_frames=n_frames, seed=seed)

    topology = directory / "toy.pdb"
    trajectory = directory / f"toy.{traj_format.lower()}"

    universe.trajectory[0]
    # bonds='all' writes CONECT records, so the topology keeps the connectivity
    # a real prmtop would provide (needed by the H-bond detection).
    universe.atoms.write(str(topology), bonds="all")
    with mda.Writer(str(trajectory), n_atoms=universe.atoms.n_atoms) as writer:
        for _ts in universe.trajectory:
            writer.write(universe.atoms)
    return topology, trajectory
