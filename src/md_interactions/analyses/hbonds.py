"""Hydrogen bonds: explicitly tracked pairs and automatic detection.

Two complementary modes, both usable at the same time:

``pairs``
    Hydrogen bonds you care about by name (substrate–catalytic residue,
    residue–water...).  For each frame the donor–acceptor distance and the
    D–H···A angle are stored, together with the occupancy (percentage of
    frames satisfying both cutoffs).  When the acceptor selection contains
    several atoms (e.g. the two oxygens of a carboxylate) the closest one is
    used at every frame; likewise the hydrogen giving the most linear
    arrangement is chosen.

``auto``
    Blind detection over a region using MDAnalysis'
    :class:`~MDAnalysis.analysis.hydrogenbonds.hbond_analysis.HydrogenBondAnalysis`,
    reported as a table of donor/acceptor pairs sorted by occupancy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.lib.distances import calc_angles, distance_array

from ..config import HBondDef, HBondsConfig
from ..exceptions import MissingTopologyInfoError
from ..io_utils import OutputPaths, write_csv
from ..plotting import color_cycle, save_figure
from ..results import AnalysisResult, summary_row
from ..system import MDSystem
from ._common import basename, current_box

__all__ = ["run", "compute_tracked_hbonds", "detect_hbonds"]

_DIST_UNIT = "Å"
_ANGLE_UNIT = "°"


# --------------------------------------------------------------------------- #
# explicit pairs
# --------------------------------------------------------------------------- #
def _donor_hydrogens(system: MDSystem, donor: mda.core.groups.AtomGroup,
                     defn: HBondDef) -> mda.core.groups.AtomGroup:
    """Hydrogens attached to the donor (explicit, bonded, or by distance)."""
    if defn.hydrogen:
        return system.select(defn.hydrogen, name=f"hbond:{defn.name}.hydrogen")

    universe = system.universe
    try:
        neighbours = [atom.bonded_atoms for atom in donor]
        if neighbours:
            bonded = neighbours[0]
            for extra in neighbours[1:]:
                bonded = bonded + extra
            hydrogens = bonded.select_atoms("mass 0.5 to 1.5")
            if hydrogens.n_atoms:
                return hydrogens
    except (mda.exceptions.NoDataError, AttributeError, TypeError):
        pass

    # Fall back to a geometric search around the donor at the first frame.
    indices = " ".join(str(i) for i in donor.indices)
    hydrogens = universe.select_atoms(
        f"(mass 0.5 to 1.5) and (around 1.35 index {indices})"
    )
    if hydrogens.n_atoms == 0:
        raise MissingTopologyInfoError(
            f"No hydrogen found on the donor of '{defn.name}' "
            f"(\"{system.config.resolve_selection(defn.donor)}\").\n"
            "  Hydrogen-bond analysis needs explicit hydrogens: check that the "
            "topology is not a united-atom/coarse-grained model, or give the "
            "hydrogen explicitly with 'hydrogen: <selection>'."
        )
    return hydrogens


def compute_tracked_hbonds(
    system: MDSystem, config: HBondsConfig | None = None
) -> pd.DataFrame:
    """Distance and angle time series for every named hydrogen bond.

    Columns per H-bond ``X``: ``X`` (D–A distance, Å), ``X_angle``
    (D–H···A, degrees) and ``X_present`` (both cutoffs satisfied).
    """
    config = config or system.config.hbonds
    tracked = []
    for defn in config.pairs:
        donor = system.select(defn.donor, name=f"hbond:{defn.name}.donor")
        acceptor = system.select(defn.acceptor, name=f"hbond:{defn.name}.acceptor")
        hydrogens = _donor_hydrogens(system, donor, defn)
        tracked.append((defn, donor, hydrogens, acceptor))

    n_frames = system.n_frames
    distances = np.full((n_frames, len(tracked)), np.nan)
    angles = np.full((n_frames, len(tracked)), np.nan)

    for i, _ts in enumerate(system.iter_frames()):
        box = current_box(system, True)
        for j, (_defn, donor, hydrogens, acceptor) in enumerate(tracked):
            d_matrix = distance_array(donor.positions, acceptor.positions, box=box)
            d_i, a_i = np.unravel_index(np.argmin(d_matrix), d_matrix.shape)
            distances[i, j] = d_matrix[d_i, a_i]

            donor_pos = donor.positions[d_i].reshape(1, 3)
            acceptor_pos = acceptor.positions[a_i].reshape(1, 3)
            candidates = np.degrees(calc_angles(
                np.repeat(donor_pos, hydrogens.n_atoms, axis=0),
                hydrogens.positions,
                np.repeat(acceptor_pos, hydrogens.n_atoms, axis=0),
                box=box,
            ))
            angles[i, j] = candidates.max()

    df = pd.DataFrame(system.time_frame_columns())
    for j, (defn, _d, _h, _a) in enumerate(tracked):
        df[defn.name] = distances[:, j]
        df[f"{defn.name}_angle"] = angles[:, j]
        df[f"{defn.name}_present"] = (
            (distances[:, j] <= config.d_a_cutoff)
            & (angles[:, j] >= config.d_h_a_angle_cutoff)
        )
    return df


# --------------------------------------------------------------------------- #
# automatic detection
# --------------------------------------------------------------------------- #
def detect_hbonds(system: MDSystem, config: HBondsConfig | None = None) -> pd.DataFrame:
    """Blind H-bond detection; returns one row per donor/acceptor pair.

    Columns: ``donor``, ``hydrogen``, ``acceptor``, ``occupancy_%``,
    ``distance_mean``, ``angle_mean``, sorted by decreasing occupancy.
    """
    from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis

    config = config or system.config.hbonds
    resolve = system.config.resolve_selection
    between = [[resolve(a), resolve(b)] for a, b in config.auto_between] or None

    kwargs = dict(
        universe=system.universe,
        between=between,
        d_a_cutoff=config.d_a_cutoff,
        d_h_a_angle_cutoff=config.d_h_a_angle_cutoff,
    )
    if config.donors_sel:
        kwargs["donors_sel"] = resolve(config.donors_sel)
    if config.hydrogens_sel:
        kwargs["hydrogens_sel"] = resolve(config.hydrogens_sel)
    if config.acceptors_sel:
        kwargs["acceptors_sel"] = resolve(config.acceptors_sel)

    # Restrict the search to the region of interest *before* running: MDAnalysis
    # applies `between` as a post-filter, so without this the neighbour search
    # scans every donor/acceptor in the box (all the solvent included), which is
    # orders of magnitude slower on a solvated system.
    if between is not None:
        kwargs.update(_restrict_to_region(system.universe, between, kwargs))

    try:
        analysis = HydrogenBondAnalysis(**kwargs)
        analysis.run(**system.run_kwargs)
    except (mda.exceptions.NoDataError, ValueError, AttributeError) as exc:
        fallback = _fallback_selections(kwargs)
        if fallback is None:
            raise MissingTopologyInfoError(
                f"Automatic hydrogen-bond detection failed: {exc}\n"
                "  MDAnalysis needs explicit hydrogens plus bond/charge information "
                "to guess donors and acceptors. With an AMBER prmtop this is "
                "normally available; otherwise set 'donors_sel', 'hydrogens_sel' "
                "and 'acceptors_sel' explicitly in the configuration."
            ) from exc
        try:
            analysis = HydrogenBondAnalysis(**{**kwargs, **fallback})
            analysis.run(**system.run_kwargs)
        except (mda.exceptions.NoDataError, ValueError, AttributeError) as exc2:
            raise MissingTopologyInfoError(
                f"Automatic hydrogen-bond detection failed: {exc2}\n"
                "  Neither MDAnalysis' guessing (which needs charges) nor the "
                "name/mass heuristics worked on this topology. Set 'donors_sel', "
                "'hydrogens_sel' and 'acceptors_sel' explicitly."
            ) from exc2

    table = np.asarray(analysis.results.hbonds)
    if table.size == 0:
        return pd.DataFrame(columns=["donor", "hydrogen", "acceptor", "occupancy_%",
                                     "distance_mean", "angle_mean", "n_frames"])

    universe = system.universe
    records: dict[tuple[int, int, int], list] = {}
    for _frame, donor_i, hydrogen_i, acceptor_i, distance, angle in table:
        key = (int(donor_i), int(hydrogen_i), int(acceptor_i))
        entry = records.setdefault(key, [0, [], []])
        entry[0] += 1
        entry[1].append(distance)
        entry[2].append(angle)

    rows = []
    for (donor_i, hydrogen_i, acceptor_i), (count, dists, angs) in records.items():
        rows.append({
            "donor": _atom_label(universe.atoms[donor_i]),
            "hydrogen": _atom_label(universe.atoms[hydrogen_i]),
            "acceptor": _atom_label(universe.atoms[acceptor_i]),
            "occupancy_%": 100.0 * count / system.n_frames,
            "distance_mean": float(np.mean(dists)),
            "angle_mean": float(np.mean(angs)),
            "n_frames": int(count),
        })
    df = pd.DataFrame(rows).sort_values("occupancy_%", ascending=False)
    df = df[df["occupancy_%"] >= config.min_occupancy]
    return df.head(config.max_reported).reset_index(drop=True)


def _restrict_to_region(universe, between: list[list[str]], kwargs: dict) -> dict[str, str]:
    """Confine donor/hydrogen/acceptor selections to the ``between`` region.

    Selections the user provided are intersected with the region; the ones left
    to MDAnalysis are guessed *within* the region using its own ``guess_*``
    helpers, falling back to name/mass heuristics if the topology lacks the
    charges those helpers need.
    """
    from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis

    selections = {sel for pair in between for sel in pair}
    region = " or ".join(f"({sel})" for sel in sorted(selections))
    guesser_names = {
        "donors_sel": "guess_donors",
        "hydrogens_sel": "guess_hydrogens",
        "acceptors_sel": "guess_acceptors",
    }

    probe = None
    if any(not kwargs.get(key) for key in guesser_names):
        # Building the probe already needs charges/bonds; if the topology has
        # none we simply fall back to the name/mass heuristics below.
        try:
            probe = HydrogenBondAnalysis(universe=universe)
        except Exception:
            probe = None

    heuristics = _fallback_selections({}) or {}
    restricted: dict[str, str] = {}
    for key, guesser in guesser_names.items():
        if kwargs.get(key):
            restricted[key] = f"({kwargs[key]}) and ({region})"
            continue
        guessed = ""
        if probe is not None:
            try:
                guessed = getattr(probe, guesser)(select=region)
            except Exception:
                guessed = ""
        restricted[key] = f"({guessed or heuristics[key]}) and ({region})"
    return restricted


def _fallback_selections(kwargs: dict) -> dict[str, str] | None:
    """Name/mass based donor-acceptor selections when guessing is impossible.

    Returns ``None`` when the user already provided every selection (there is
    nothing left to fall back to).
    """
    provided = {k for k in ("donors_sel", "hydrogens_sel", "acceptors_sel")
                if kwargs.get(k)}
    if len(provided) == 3:
        return None
    defaults = {
        "donors_sel": "name N* O* S*",
        "hydrogens_sel": "mass 0.5 to 1.5",
        "acceptors_sel": "name N* O* S*",
    }
    return {k: v for k, v in defaults.items() if k not in provided}


def _atom_label(atom) -> str:
    return f"{atom.resname}{atom.resid}:{atom.name}"


# --------------------------------------------------------------------------- #
# module entry point
# --------------------------------------------------------------------------- #
def run(
    system: MDSystem,
    paths: OutputPaths,
    config: HBondsConfig | None = None,
    verbose: bool = True,
) -> AnalysisResult:
    """Run the hydrogen-bond analysis and write CSVs and figures."""
    config = config or system.config.hbonds
    formats = system.config.output.formats
    dpi = system.config.output.dpi

    tables: dict[str, pd.DataFrame] = {}
    rows: list[dict] = []
    plots: list = []
    notes = [
        f"Criteria: d(D–A) ≤ {config.d_a_cutoff:g} {_DIST_UNIT} and "
        f"∠(D–H···A) ≥ {config.d_h_a_angle_cutoff:g}{_ANGLE_UNIT}."
    ]
    series = None
    occupancies: dict[str, float] = {}
    tracked_atom_pairs: set[tuple[str, str]] = set()

    if config.pairs:
        if verbose:
            print(f"[hbonds] {len(config.pairs)} tracked H-bond(s)")
        df = compute_tracked_hbonds(system, config)
        tables["hbonds_tracked"] = df
        write_csv(df, paths.data, "hbonds_tracked")
        series = df[["frame", "time"]
                    + [c for c in df.columns if not c.endswith("_present")
                       and c not in ("frame", "time")]]

        for defn in config.pairs:
            distance = df[defn.name].to_numpy()
            angle = df[f"{defn.name}_angle"].to_numpy()
            occupancy = 100.0 * float(np.mean(df[f"{defn.name}_present"].to_numpy()))
            occupancies[defn.label or defn.name] = occupancy
            label = defn.label or f"{defn.donor}···{defn.acceptor}"
            rows.append(summary_row("hbonds", defn.name, distance, _DIST_UNIT,
                                    definition=f"d(D–A) {label}",
                                    occupancy_pct=occupancy))
            rows.append(summary_row("hbonds", f"{defn.name}_angle", angle, _ANGLE_UNIT,
                                    definition=f"∠(D–H···A) {label}"))
            plots += _plot_tracked(system, paths, defn, df, config, formats, dpi)

            # remember which atom pairs are already tracked, so the automatic
            # detection does not report the same H-bond a second time
            donors = system.select(defn.donor, name=f"hbond:{defn.name}.donor")
            acceptors = system.select(defn.acceptor, name=f"hbond:{defn.name}.acceptor")
            for donor_atom in donors:
                for acceptor_atom in acceptors:
                    tracked_atom_pairs.add(
                        (_atom_label(donor_atom), _atom_label(acceptor_atom))
                    )

    if config.auto:
        if verbose:
            print("[hbonds] automatic detection ...")
        try:
            detected = detect_hbonds(system, config)
        except MissingTopologyInfoError as exc:
            # Do not throw away the tracked pairs because the blind search failed.
            if not config.pairs:
                raise
            notes.append(f"Automatic detection skipped: {str(exc).splitlines()[0]}")
            print(f"[hbonds] WARNING: automatic detection skipped ({exc})")
            detected = None
        if detected is not None:
            tables["hbonds_detected"] = detected
            write_csv(detected, paths.data, "hbonds_detected")
            notes.append(
                f"{len(detected)} H-bond(s) detected with occupancy ≥ "
                f"{config.min_occupancy:g}% (top {config.max_reported} reported)."
            )
            for _i, row in detected.iterrows():
                if (row["donor"], row["acceptor"]) in tracked_atom_pairs:
                    continue  # already shown as a tracked pair
                key = f"{row['donor']}–H···{row['acceptor']}"
                occupancies.setdefault(key, float(row["occupancy_%"]))

    if occupancies:
        plots += _plot_occupancy(occupancies, paths, formats, dpi)

    return AnalysisResult(
        name="hbonds",
        title="Hydrogen bonds",
        tables=tables,
        series=series,
        summary=pd.DataFrame(rows),
        plots=list(plots),
        notes=notes,
    )


def _plot_tracked(system, paths, defn: HBondDef, df, config, formats, dpi) -> list:
    """Two stacked panels: D–A distance and D–H···A angle."""
    import matplotlib.pyplot as plt

    time = df["time"].to_numpy()
    colors = color_cycle(2)
    fig, (ax_d, ax_a) = plt.subplots(
        2, 1, sharex=True, figsize=(6.6, 4.4),
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12},
    )
    ax_d.plot(time, df[defn.name].to_numpy(), color=colors[0])
    ax_d.axhline(config.d_a_cutoff, color="0.35", linestyle="--", linewidth=1.0)
    ax_d.set_ylabel(f"d(D–A) ({_DIST_UNIT})")

    ax_a.plot(time, df[f"{defn.name}_angle"].to_numpy(), color=colors[1])
    ax_a.axhline(config.d_h_a_angle_cutoff, color="0.35", linestyle="--", linewidth=1.0)
    ax_a.set_ylabel(f"∠(D–H···A) ({_ANGLE_UNIT})")
    ax_a.set_xlabel(system.time_label)
    ax_a.set_ylim(0, 180)

    occupancy = 100.0 * float(np.mean(df[f"{defn.name}_present"].to_numpy()))
    ax_d.set_title(f"{defn.label or defn.name} — occupancy {occupancy:.1f}%")
    for axis in (ax_d, ax_a):
        axis.set_xlim(float(time.min()), float(time.max()))
    fig.tight_layout()
    return save_figure(fig, paths.plots, basename("hbond", defn.name, "_timeseries"),
                       formats, dpi)[:1]


def _plot_occupancy(occupancies: dict[str, float], paths, formats, dpi) -> list:
    import matplotlib.pyplot as plt

    labels = list(occupancies)
    values = [occupancies[k] for k in labels]
    order = np.argsort(values)
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]

    height = max(2.4, 0.32 * len(labels) + 1.0)
    fig, ax = plt.subplots(figsize=(6.8, height))
    ax.barh(range(len(labels)), values, color=color_cycle(1)[0], alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Occupancy (%)")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    for i, value in enumerate(values):
        ax.text(min(value + 1.5, 97), i, f"{value:.0f}%", va="center", fontsize=8)
    fig.tight_layout()
    return save_figure(fig, paths.plots, "hbond_occupancy", formats, dpi)[:1]
