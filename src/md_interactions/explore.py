"""Automatic discovery of the interactions worth measuring.

Instead of naming residues and atoms up front — which forces you to know the
numbering of every topology you touch — this module inverts the problem: point
it at a region and it reports what is actually there, ranked by how persistent
each interaction is.

Four modes, all built on the same machinery:

``contacts``
    Every polar contact, hydrogen bond and salt bridge around a region, with
    its occupancy along the trajectory.
``changes``
    What forms or breaks during the simulation, including covalent bonds and
    transferred protons — the reactive events a topology-based analysis would
    silently misreport.
``compare``
    The same table for two trajectories, joined and sorted by the change in
    occupancy (pre-reactive vs product, wild type vs mutant...).
``reaction``
    Only the reaction geometry: bonds made and broken, proton transfers and
    close nucleophile–electrophile approaches.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from MDAnalysis.lib.distances import calc_angles, calc_bonds, capped_distance

from .exceptions import AnalysisError
from .system import MDSystem

__all__ = [
    "find_contacts",
    "detect_bond_changes",
    "detect_proton_transfers",
    "compare_contacts",
    "reaction_summary",
    "DEFAULT_REGION",
]

#: Default region when the user does not give one: everything that is neither
#: protein nor solvent, i.e. the ligand/cofactor.
DEFAULT_REGION = "not protein and not resname WAT HOH SOL TIP3 T3P NA CL K MG ZN CA"

_WATER_RESNAMES = {"WAT", "HOH", "SOL", "TIP3", "T3P"}

#: Formally charged groups, used to label salt bridges.
_ANIONIC = {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2"),
            ("ASH", "OD1"), ("ASH", "OD2"), ("GLH", "OE1"), ("GLH", "OE2")}
_CATIONIC = {("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"), ("LYS", "NZ"),
             ("HIP", "ND1"), ("HIP", "NE2")}


# --------------------------------------------------------------------------- #
# atom classification
# --------------------------------------------------------------------------- #
def _element_of(masses: np.ndarray) -> np.ndarray:
    """Element symbols from masses (robust when the topology has no elements)."""
    table = [(1.5, "H"), (13.0, "C"), (14.5, "N"), (16.5, "O"),
             (19.5, "F"), (31.0, "P"), (32.6, "S"), (36.0, "CL")]
    elements = np.full(masses.shape, "X", dtype=object)
    for index, mass in enumerate(masses):
        for limit, symbol in table:
            if mass <= limit:
                elements[index] = symbol
                break
    return elements


def _polar_mask(group) -> np.ndarray:
    """N, O and S atoms — the ones that can donate or accept a hydrogen bond."""
    elements = _element_of(np.asarray(group.masses, dtype=float))
    return np.isin(elements, ["N", "O", "S"])


def _hydrogens_by_heavy(universe) -> dict[int, list[int]]:
    """``{heavy atom index: [bonded hydrogen indices]}`` from the topology."""
    mapping: dict[int, list[int]] = {}
    try:
        bonds = universe.bonds.indices
    except Exception:
        return mapping
    masses = np.asarray(universe.atoms.masses, dtype=float)
    for first, second in bonds:
        for heavy, light in ((first, second), (second, first)):
            if masses[light] < 1.5 <= masses[heavy]:
                mapping.setdefault(int(heavy), []).append(int(light))
    return mapping


def _label(atom) -> str:
    return f"{atom.resname}{atom.resid}:{atom.name}"


def _is_water(resname: str) -> bool:
    return str(resname).upper() in _WATER_RESNAMES


def _charge_class(resname: str, name: str) -> str:
    if (resname, name) in _ANIONIC or (name.startswith("O") and name.endswith("P")):
        return "-"
    if (resname, name) in _CATIONIC:
        return "+"
    return "0"


# --------------------------------------------------------------------------- #
# contacts
# --------------------------------------------------------------------------- #
@dataclass
class _Candidate:
    """A polar pair seen close together in at least one scanned frame."""

    a: int
    b: int
    label_a: str = ""
    label_b: str = ""
    kind: str = "polar"
    aggregate: str = ""          # non-empty when the partner is bulk solvent


def find_contacts(
    system: MDSystem,
    region: str = DEFAULT_REGION,
    cutoff: float = 4.0,
    scan_frames: int = 25,
    hbond_distance: float = 3.5,
    hbond_angle: float = 150.0,
    waters: str = "aggregate",
    min_occupancy: float = 1.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Find and score every polar contact around ``region``.

    Parameters
    ----------
    region
        Selection defining what to look around (default: the ligand).
    cutoff
        Heavy-atom distance used to propose candidate pairs.
    scan_frames
        How many evenly spaced frames are scanned to build the candidate list.
        Contacts that never come within ``cutoff`` in any of them are missed,
        so raise it for very mobile sites.
    waters
        ``aggregate`` reports "any water" as a single partner per region atom
        (individual waters exchange constantly), ``individual`` keeps each
        water, ``ignore`` drops the solvent.

    Returns
    -------
    DataFrame
        One row per contact: labels, kind (hbond/salt bridge/polar), occupancy
        and distance statistics, sorted by decreasing occupancy.
    """
    if waters not in {"aggregate", "individual", "ignore"}:
        raise AnalysisError("'waters' must be 'aggregate', 'individual' or 'ignore'.")

    universe = system.universe
    region_group = system.select(region, name="explore.region")
    region_polar = region_group[_polar_mask(region_group)]
    if region_polar.n_atoms == 0:
        raise AnalysisError(
            f'The region "{region}" contains no N/O/S atoms, so it cannot form '
            "polar contacts. Check the selection."
        )

    environment = universe.select_atoms(f"not ({region})")
    environment_polar = environment[_polar_mask(environment)]
    if waters == "ignore":
        keep = np.array([not _is_water(r) for r in environment_polar.resnames])
        environment_polar = environment_polar[keep]

    if verbose:
        print(f"[explore] region: {region_polar.n_atoms} polar atoms; "
              f"environment: {environment_polar.n_atoms} polar atoms")

    candidates = _scan_candidates(system, region_polar, environment_polar,
                                  cutoff, scan_frames, waters, verbose)
    if not candidates:
        return pd.DataFrame(columns=["atom_a", "atom_b", "kind", "occupancy_%",
                                     "distance_mean", "distance_min", "angle_mean"])

    return _score_candidates(system, candidates, hbond_distance, hbond_angle,
                             cutoff, min_occupancy, verbose)


def _scan_candidates(system, region_polar, environment_polar, cutoff,
                     scan_frames, waters, verbose) -> list[_Candidate]:
    """Frames scattered along the trajectory propose the pairs worth following."""
    universe = system.universe
    positions = np.linspace(0, system.n_frames - 1,
                            min(scan_frames, system.n_frames)).astype(int)
    positions = np.unique(positions)

    seen: dict[tuple[int, int], _Candidate] = {}
    for position in positions:
        system.goto(int(position))
        box = universe.dimensions if _valid_box(universe.dimensions) else None
        pairs, _distances = capped_distance(
            region_polar.positions, environment_polar.positions,
            max_cutoff=cutoff, box=box, return_distances=True,
        )
        for i, j in pairs:
            atom_a = region_polar[int(i)]
            atom_b = environment_polar[int(j)]
            aggregate = ""
            if waters == "aggregate" and _is_water(atom_b.resname):
                # all waters collapse into one "any water" partner per region atom
                key = (int(atom_a.index), -1)
                aggregate = f"{atom_b.resname} (any)"
            else:
                key = (int(atom_a.index), int(atom_b.index))
            if key in seen:
                continue
            seen[key] = _Candidate(
                a=int(atom_a.index),
                b=int(atom_b.index) if not aggregate else -1,
                label_a=_label(atom_a),
                label_b=aggregate or _label(atom_b),
                kind=_classify(atom_a, atom_b, aggregate),
                aggregate=aggregate,
            )
    if verbose:
        print(f"[explore] {len(seen)} candidate pair(s) from "
              f"{positions.size} scanned frame(s)")
    return list(seen.values())


def _classify(atom_a, atom_b, aggregate: str) -> str:
    if aggregate:
        return "water"
    charge_a = _charge_class(atom_a.resname, atom_a.name)
    charge_b = _charge_class(atom_b.resname, atom_b.name)
    if {charge_a, charge_b} == {"+", "-"}:
        return "salt bridge"
    return "polar"


def _valid_box(box) -> bool:
    if box is None:
        return False
    box = np.asarray(box, dtype=float)
    return box.shape == (6,) and np.all(np.isfinite(box)) and np.all(box[:3] > 0)


def _score_candidates(system, candidates, hbond_distance, hbond_angle,
                      cutoff, min_occupancy, verbose) -> pd.DataFrame:
    """One pass over the trajectory scoring every candidate pair."""
    universe = system.universe
    hydrogens = _hydrogens_by_heavy(universe)

    # aggregated water partners need the whole water oxygen set at every frame
    solvent = universe.select_atoms("resname WAT HOH SOL TIP3 T3P")
    water_polar = solvent[_polar_mask(solvent)] if solvent.n_atoms else solvent

    direct = [c for c in candidates if not c.aggregate]
    aggregated = [c for c in candidates if c.aggregate]

    idx_a = np.array([c.a for c in direct], dtype=int)
    idx_b = np.array([c.b for c in direct], dtype=int)
    agg_a = np.array([c.a for c in aggregated], dtype=int)

    # donor/hydrogen/acceptor triplets for the angle criterion (both directions)
    triplets, triplet_owner = _build_triplets(direct, hydrogens)

    n_frames = system.n_frames
    distances = np.full((n_frames, len(direct)), np.nan)
    angles = np.full((n_frames, len(direct)), np.nan)
    agg_distances = np.full((n_frames, len(aggregated)), np.nan)

    for step, _ts in enumerate(system.iter_frames()):
        box = universe.dimensions if _valid_box(universe.dimensions) else None
        all_positions = universe.atoms.positions

        if len(direct):
            distances[step] = calc_bonds(all_positions[idx_a], all_positions[idx_b],
                                         box=box)
        if triplets.size:
            values = np.degrees(calc_angles(
                all_positions[triplets[:, 0]], all_positions[triplets[:, 1]],
                all_positions[triplets[:, 2]], box=box,
            ))
            # best (most linear) hydrogen for each pair
            for pair_index in range(len(direct)):
                mask = triplet_owner == pair_index
                if mask.any():
                    angles[step, pair_index] = values[mask].max()

        if len(aggregated) and water_polar.n_atoms:
            water_positions = water_polar.positions
            for k, atom_index in enumerate(agg_a):
                deltas = calc_bonds(
                    np.repeat(all_positions[atom_index][None, :],
                              water_positions.shape[0], axis=0),
                    water_positions, box=box,
                )
                agg_distances[step, k] = deltas.min()

    rows: list[dict] = []
    for index, candidate in enumerate(direct):
        series = distances[:, index]
        angle_series = angles[:, index]
        rows.append(_row(candidate, series, angle_series, hbond_distance,
                         hbond_angle, cutoff))
    for index, candidate in enumerate(aggregated):
        series = agg_distances[:, index]
        rows.append(_row(candidate, series, np.full_like(series, np.nan),
                         hbond_distance, hbond_angle, cutoff))

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame[frame["occupancy_%"] >= min_occupancy]
    frame = frame.sort_values("occupancy_%", ascending=False).reset_index(drop=True)
    if verbose:
        print(f"[explore] {len(frame)} contact(s) with occupancy >= {min_occupancy:g}%")
    return frame


def _build_triplets(direct, hydrogens) -> tuple[np.ndarray, np.ndarray]:
    """(donor, H, acceptor) triplets for every candidate, in both directions."""
    triplets: list[tuple[int, int, int]] = []
    owner: list[int] = []
    for index, candidate in enumerate(direct):
        for donor, acceptor in ((candidate.a, candidate.b), (candidate.b, candidate.a)):
            for hydrogen in hydrogens.get(donor, []):
                triplets.append((donor, hydrogen, acceptor))
                owner.append(index)
    if not triplets:
        return np.empty((0, 3), dtype=int), np.empty(0, dtype=int)
    return np.asarray(triplets, dtype=int), np.asarray(owner, dtype=int)


def _row(candidate: _Candidate, distances: np.ndarray, angles: np.ndarray,
         hbond_distance: float, hbond_angle: float, cutoff: float) -> dict:
    clean = distances[~np.isnan(distances)]
    has_angle = np.any(~np.isnan(angles))
    if has_angle:
        present = (distances <= hbond_distance) & (angles >= hbond_angle)
        kind = "hbond" if present.any() else candidate.kind
    else:
        present = distances <= min(hbond_distance, cutoff)
        kind = candidate.kind
    if candidate.kind == "salt bridge" and (distances <= 4.0).any():
        kind = "salt bridge"
    return {
        "atom_a": candidate.label_a,
        "atom_b": candidate.label_b,
        "kind": kind,
        "occupancy_%": float(np.mean(present) * 100.0),
        "distance_mean": float(clean.mean()) if clean.size else np.nan,
        "distance_min": float(clean.min()) if clean.size else np.nan,
        "angle_mean": float(np.nanmean(angles)) if has_angle else np.nan,
        "index_a": candidate.a,
        "index_b": candidate.b,
    }


# --------------------------------------------------------------------------- #
# changes: covalent bonds and protons
# --------------------------------------------------------------------------- #
def detect_bond_changes(
    system: MDSystem,
    broken_factor: float = 1.6,
    formed_cutoff: float = 1.8,
    region: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Covalent bonds of the topology that break, and new ones that form.

    A bond counts as broken when its length exceeds ``broken_factor`` times the
    value it has in the first analysed frame.  Bond formation is searched among
    heavy-atom pairs that are *not* bonded in the topology and come closer than
    ``formed_cutoff``.

    This is what tells you that a trajectory is reactive: an AMBER topology
    describes the connectivity of the starting structure only, so a transferred
    proton or a cleaved bond is invisible to any analysis that trusts it.
    """
    universe = system.universe
    try:
        bond_indices = universe.bonds.indices
    except Exception as exc:
        raise AnalysisError(
            "The topology carries no bond information, so bond changes cannot "
            "be detected."
        ) from exc

    if region:
        selected = set(system.select(region, name="explore.region").indices.tolist())
        mask = np.array([a in selected or b in selected for a, b in bond_indices])
        bond_indices = bond_indices[mask]

    lengths = np.full((system.n_frames, len(bond_indices)), np.nan)
    for step, _ts in enumerate(system.iter_frames()):
        box = universe.dimensions if _valid_box(universe.dimensions) else None
        positions = universe.atoms.positions
        lengths[step] = calc_bonds(positions[bond_indices[:, 0]],
                                   positions[bond_indices[:, 1]], box=box)

    reference = lengths[0]
    maxima = np.nanmax(lengths, axis=0)
    minima = np.nanmin(lengths, axis=0)
    broken = np.flatnonzero((maxima > reference * broken_factor) | (maxima > 2.2))

    masses = np.asarray(universe.atoms.masses, dtype=float)
    rows = []
    for k in broken:
        first, second = bond_indices[k]
        series = lengths[:, k]
        # A bond already long in the first frame broke *before* this trajectory:
        # the topology describes the starting structure of the whole project,
        # not of this particular run.
        already = bool(minima[k] > 1.8)
        state = "already broken at frame 0" if already else "breaks during the run"
        note = ""
        for heavy, light in ((first, second), (second, first)):
            if masses[light] < 1.5 <= masses[heavy]:
                note = _current_host(system, int(light))
                break
        rows.append({
            "event": "bond broken",
            "atom_a": _label(universe.atoms[int(first)]),
            "atom_b": _label(universe.atoms[int(second)]),
            "state": state,
            "initial": float(reference[k]),
            "mean": float(np.nanmean(series)),
            "max": float(maxima[k]),
            "note": note,
        })

    formed = _detect_formed_bonds(system, set(map(tuple, np.sort(bond_indices, axis=1))),
                                  formed_cutoff)
    frame = pd.DataFrame(rows + formed)
    if verbose:
        print(f"[explore] {len(rows)} broken bond(s), {len(formed)} newly formed")
    return frame


def _current_host(system: MDSystem, hydrogen_index: int) -> str:
    """Which heavy atom the hydrogen is actually bonded to, at the last frame."""
    universe = system.universe
    system.goto(system.n_frames - 1)
    heavy = universe.atoms[np.asarray(universe.atoms.masses, dtype=float) >= 1.5]
    box = universe.dimensions if _valid_box(universe.dimensions) else None
    pairs, distances = capped_distance(
        universe.atoms[hydrogen_index].position[None, :], heavy.positions,
        max_cutoff=1.4, box=box, return_distances=True,
    )
    if len(pairs) == 0:
        return "no heavy atom within 1.4 A"
    best = int(np.argmin(distances))
    host = heavy[int(pairs[best][1])]
    return f"H now bonded to {_label(host)} ({distances[best]:.2f} A)"


def _detect_formed_bonds(system: MDSystem, existing: set, cutoff: float) -> list[dict]:
    """Heavy-atom pairs that approach to covalent distance without being bonded."""
    universe = system.universe
    heavy = universe.select_atoms("not type H")
    heavy = heavy[np.asarray(heavy.masses, dtype=float) > 1.5]
    found: dict[tuple[int, int], list[float]] = {}

    positions = np.linspace(0, system.n_frames - 1,
                            min(40, system.n_frames)).astype(int)
    for position in np.unique(positions):
        system.goto(int(position))
        box = universe.dimensions if _valid_box(universe.dimensions) else None
        pairs, distances = capped_distance(heavy.positions, heavy.positions,
                                           max_cutoff=cutoff, box=box,
                                           return_distances=True)
        for (i, j), distance in zip(pairs, distances):
            if i >= j:
                continue
            key = tuple(sorted((int(heavy[int(i)].index), int(heavy[int(j)].index))))
            if key in existing:
                continue
            found.setdefault(key, []).append(float(distance))

    rows = []
    for (first, second), values in found.items():
        rows.append({
            "event": "bond formed",
            "atom_a": _label(universe.atoms[first]),
            "atom_b": _label(universe.atoms[second]),
            "state": "not bonded in the topology",
            "initial": np.nan,
            "mean": float(np.mean(values)),
            "max": float(np.max(values)),
            "note": "covalent distance: the topology is out of date for this run",
        })
    return rows


def detect_proton_transfers(
    system: MDSystem,
    scan_frames: int = 40,
    verbose: bool = True,
) -> pd.DataFrame:
    """Hydrogens whose covalent host changes during the trajectory.

    For every hydrogen the nearest heavy atom is tracked; when that identity
    changes, the proton moved.  Reports where it started, where it ended and
    the fraction of the scanned frames spent on each host.
    """
    universe = system.universe
    masses = np.asarray(universe.atoms.masses, dtype=float)
    hydrogens = universe.atoms[masses < 1.5]
    heavy = universe.atoms[masses >= 1.5]
    if hydrogens.n_atoms == 0:
        raise AnalysisError("The topology has no explicit hydrogens.")

    positions = np.unique(np.linspace(0, system.n_frames - 1,
                                      min(scan_frames, system.n_frames)).astype(int))
    hosts = np.empty((positions.size, hydrogens.n_atoms), dtype=int)

    for step, position in enumerate(positions):
        system.goto(int(position))
        box = universe.dimensions if _valid_box(universe.dimensions) else None
        pairs, distances = capped_distance(hydrogens.positions, heavy.positions,
                                           max_cutoff=1.4, box=box,
                                           return_distances=True)
        best = np.full(hydrogens.n_atoms, -1, dtype=int)
        best_distance = np.full(hydrogens.n_atoms, np.inf)
        for (i, j), distance in zip(pairs, distances):
            if distance < best_distance[int(i)]:
                best_distance[int(i)] = distance
                best[int(i)] = int(heavy[int(j)].index)
        hosts[step] = best

    rows = []
    for column in range(hydrogens.n_atoms):
        series = hosts[:, column]
        valid = series[series >= 0]
        if valid.size == 0:
            continue
        unique, counts = np.unique(valid, return_counts=True)
        if unique.size < 2:
            continue
        order = np.argsort(-counts)
        rows.append({
            "event": "proton transfer",
            "hydrogen": _label(hydrogens[column]),
            "host_start": _label(universe.atoms[int(valid[0])]),
            "host_end": _label(universe.atoms[int(valid[-1])]),
            "hosts": ", ".join(
                f"{_label(universe.atoms[int(unique[k])])} "
                f"({100 * counts[k] / valid.size:.0f}%)" for k in order[:3]
            ),
            "n_hosts": int(unique.size),
        })

    frame = pd.DataFrame(rows)
    if verbose:
        print(f"[explore] {len(frame)} proton transfer(s) detected")
    return frame


# --------------------------------------------------------------------------- #
# compare / reaction
# --------------------------------------------------------------------------- #
def compare_contacts(first: pd.DataFrame, second: pd.DataFrame,
                     names: tuple[str, str] = ("A", "B"),
                     min_change: float = 10.0) -> pd.DataFrame:
    """Join two contact tables and rank by the change in occupancy."""
    key = ["atom_a", "atom_b"]
    left = first[key + ["kind", "occupancy_%", "distance_mean"]].copy()
    right = second[key + ["occupancy_%", "distance_mean"]].copy()
    left.columns = key + ["kind", f"occupancy_{names[0]}", f"distance_{names[0]}"]
    right.columns = key + [f"occupancy_{names[1]}", f"distance_{names[1]}"]

    merged = left.merge(right, on=key, how="outer")
    for column in (f"occupancy_{names[0]}", f"occupancy_{names[1]}"):
        merged[column] = merged[column].fillna(0.0)
    merged["delta_occupancy"] = (merged[f"occupancy_{names[1]}"]
                                 - merged[f"occupancy_{names[0]}"])
    merged = merged[merged["delta_occupancy"].abs() >= min_change]
    return merged.sort_values("delta_occupancy", key=abs, ascending=False) \
                 .reset_index(drop=True)


@dataclass
class ReactionReport:
    """What changed chemically during the trajectory."""

    bonds: pd.DataFrame = field(default_factory=pd.DataFrame)
    protons: pd.DataFrame = field(default_factory=pd.DataFrame)
    approaches: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def is_reactive(self) -> bool:
        return not (self.bonds.empty and self.protons.empty)


def reaction_summary(
    system: MDSystem,
    region: str = DEFAULT_REGION,
    approach_cutoff: float = 3.2,
    verbose: bool = True,
) -> ReactionReport:
    """Bond changes, proton transfers and close reactive approaches."""
    bonds = detect_bond_changes(system, verbose=verbose)
    protons = detect_proton_transfers(system, verbose=verbose)
    contacts = find_contacts(system, region=region, cutoff=approach_cutoff,
                             waters="aggregate", min_occupancy=5.0, verbose=verbose)
    approaches = contacts[contacts["distance_min"] <= approach_cutoff] \
        if not contacts.empty else contacts
    return ReactionReport(bonds=bonds, protons=protons, approaches=approaches)
