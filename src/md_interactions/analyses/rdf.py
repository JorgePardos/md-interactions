"""Radial distribution functions and solvation shells.

Built for the question that comes up most often in practice: *how is this atom
(or this residue) hydrated?*  For every pair it reports

* ``g(r)`` — the radial distribution function,
* ``n(r)`` — the running coordination number,
* the **first solvation shell**: the first minimum of ``g(r)`` and the number of
  partners inside it,
* the **hydration number over time** — how many waters sit inside that shell at
  each frame, which is what shows a shell being lost or exchanged,
* **which** molecules occupy that shell and for how long, because an average of
  two waters made of the same two molecules throughout and one made of a
  different pair every frame are the same number and completely different
  chemistry,
* a **regime**: a site can have a structured shell, look like bulk, or actively
  *exclude* solvent.  The third case has ``g(r)`` staying below 1 everywhere,
  and it is a finding in its own right — a buried or hydrogen-bonded site —
  not merely a failed measurement.

The histogram is accumulated frame by frame, so the averaged ``g(r)``, the time
series and the occupancy of the shell all come out of a single pass over the
trajectory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from MDAnalysis.lib.distances import capped_distance

from ..config import RDFConfig, RDFPair
from ..exceptions import AnalysisError
from ..io_utils import OutputPaths, write_csv
from ..plotting import color_cycle, plot_timeseries, save_figure
from ..results import AnalysisResult, summary_row
from ..system import MDSystem
from ._common import basename, current_box, safe_name

__all__ = ["run", "compute_rdf", "first_shell", "hydration_series",
           "shell_residents", "classify_shell", "bulk_tail"]

#: Radius within which the identity of the partners is tracked.  Bounded on
#: purpose: keeping every partner out to ``range[1]`` would hold hundreds of
#: molecules per frame in memory to answer a question that is only ever asked
#: about the first shell or two.
RESIDENT_RADIUS = 6.0

#: Used to report an occupancy when no shell can be detected.  The usual
#: hydrogen-bond limit for water, so "waters within 3.5 Å" stays meaningful
#: even for a site whose g(r) has no minimum to find.
FALLBACK_CUTOFF = 3.5


def compute_rdf(
    system: MDSystem, spec: RDFPair,
) -> tuple[pd.DataFrame, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """RDF, running coordination number, per-frame histogram and occupants.

    Returns ``(profile, counts, residents)`` where ``profile`` has columns
    ``r``, ``g_r`` and ``n_r``, ``counts`` is the ``(n_frames, nbins)``
    histogram — kept so the hydration time series needs no second pass — and
    ``residents`` holds, for every frame, the residue ids of the partners
    within :data:`RESIDENT_RADIUS` and their distances.

    ``spec.center`` chooses what the distances are measured from: ``atom``
    averages over every atom of ``g1`` (the usual definition), while ``com``
    uses the centre of mass of ``g1`` as a single centre, which is what "water
    around this residue" usually means for a compact group.
    """
    group1 = system.select(spec.g1, name=f"rdf:{spec.name}.g1")
    group2 = system.select(spec.g2, name=f"rdf:{spec.name}.g2")

    rmin, rmax = spec.range
    if rmax <= rmin:
        raise AnalysisError(f"'{spec.name}': range must be increasing.")
    edges = np.linspace(rmin, rmax, spec.nbins + 1)
    counts = np.zeros((system.n_frames, spec.nbins))
    volumes = np.zeros(system.n_frames)

    mode = _center_mode(spec.center)
    partner_residues = group2.resindices
    partner_resids = np.asarray(group2.resids)
    residents: list[tuple[np.ndarray, np.ndarray]] = []
    empty = (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32))

    for step, _ts in enumerate(system.iter_frames()):
        box = current_box(system, True)
        centres = (group1.center_of_mass().reshape(1, 3).astype(np.float32)
                   if mode == "com" else group1.positions)
        pairs, distances = capped_distance(
            centres, group2.positions, max_cutoff=float(rmax), box=box,
            return_distances=True,
        )
        occupants = empty
        if len(pairs):
            keep = distances > 1e-6                     # drop self-pairs
            if spec.exclude_same_residue and mode == "atom":
                same = group1.resindices[pairs[:, 0]] == partner_residues[pairs[:, 1]]
                keep &= ~same
            pairs, distances = pairs[keep], distances[keep]

            # Identity is captured before the histogram, and before the
            # proximal collapse below, so it is the same set of contacts in
            # every mode: one entry per partner *residue*, at its closest
            # approach in this frame.
            close = distances <= RESIDENT_RADIUS
            if close.any():
                occupants = _nearest_per_residue(
                    partner_resids[pairs[close, 1]], distances[close])

            if mode == "proximal":
                # one distance per partner: how far it is from the *closest*
                # atom of the group, which is what "water around this residue"
                # actually means for a non-spherical solute
                order = np.argsort(distances)
                _unique, first = np.unique(pairs[order, 1], return_index=True)
                distances = distances[order][first]
            counts[step] = np.histogram(distances, bins=edges)[0]
        residents.append(occupants)
        volumes[step] = _frame_volume(system, box, group2)

    centres_r = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    mean_counts = counts.mean(axis=0)

    if mode == "proximal":
        # No spherical shell to divide by: the accessible volume around a
        # non-spherical solute is not 4*pi*r^2*dr, and pretending otherwise
        # would produce a g(r) that cannot be compared with bulk. Report the
        # distribution itself (molecules per Å) instead of a fake g(r).
        profile = pd.DataFrame({"r": centres_r, "g_r": mean_counts / widths,
                                "n_r": np.cumsum(mean_counts)})
        return profile, counts, residents

    n_centres = 1 if mode == "com" else group1.n_atoms
    density = group2.n_atoms / float(np.mean(volumes))
    per_centre = mean_counts / n_centres
    shell_volume = 4.0 * np.pi * centres_r ** 2 * widths
    with np.errstate(divide="ignore", invalid="ignore"):
        g_r = np.nan_to_num(per_centre / (shell_volume * density))
    profile = pd.DataFrame({"r": centres_r, "g_r": g_r,
                            "n_r": np.cumsum(per_centre)})
    return profile, counts, residents


def _nearest_per_residue(resids: np.ndarray,
                         distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One entry per residue, holding its closest approach."""
    order = np.argsort(distances)
    sorted_resids = resids[order]
    unique, first = np.unique(sorted_resids, return_index=True)
    # np.unique reports the first occurrence in the array it was given, and
    # that array is sorted by increasing distance, so this picks the minimum.
    return unique.astype(np.int32), distances[order][first].astype(np.float32)


def _center_mode(center: str) -> str:
    """``atom`` | ``com`` | ``proximal`` (nearest atom of the group)."""
    center = str(center).lower()
    if center in {"residue", "proximal", "min", "nearest"}:
        return "proximal"
    if center == "com":
        return "com"
    return "atom"


def is_normalised(spec: RDFPair) -> bool:
    """True when the profile is a real ``g(r)`` comparable with bulk density."""
    return _center_mode(spec.center) != "proximal"


def _frame_volume(system: MDSystem, box, group2) -> float:
    """Box volume, or an estimate when the trajectory carries no box."""
    if box is not None:
        return float(np.prod(np.asarray(box[:3], dtype=float)))
    positions = group2.positions
    spread = positions.max(axis=0) - positions.min(axis=0)
    return float(np.prod(np.maximum(spread, 1.0)))


def first_shell(profile: pd.DataFrame, min_peak: float = 1.05,
                normalised: bool = True, dip_fraction: float = 0.75) -> dict:
    """First peak of the profile and the minimum that closes the shell.

    Returns ``{peak_r, peak_g, minimum_r, coordination}``, all NaN when there
    is no structured shell — a flat ``g(r)`` has no coordination number worth
    quoting, and reporting one anyway would be misleading.  For an
    unnormalised (proximal) profile only the shape matters, so ``min_peak``
    is not applied.
    """
    r = profile["r"].to_numpy()
    g = profile["g_r"].to_numpy()
    empty = {"peak_r": np.nan, "peak_g": np.nan, "minimum_r": np.nan,
             "coordination": np.nan}
    if g.size < 5 or not np.any(g > 0):
        return empty
    if normalised and not np.any(g > min_peak):
        return empty

    # The FIRST local maximum, not the global one: a proximal distribution
    # keeps growing with r (more accessible volume further out), so the global
    # maximum sits at the last bin and says nothing about the first shell.
    floor = min_peak if normalised else 0.15 * float(np.max(g))
    peak = None
    for index in range(1, g.size - 1):
        if g[index] >= floor and g[index] > g[index - 1] and g[index] >= g[index + 1]:
            peak = index
            break
    if peak is None:
        if not normalised:
            return empty
        peak = int(np.argmax(g))
        if g[peak] < min_peak:
            return empty

    # The shell closes at the first *real* dip, not at any wiggle: a noisy
    # profile has small local minima right after the peak, and taking one of
    # them would report a first shell holding a fraction of a molecule.
    minimum = None
    for index in range(peak + 1, g.size - 1):
        is_local_min = g[index] <= g[index + 1]
        deep_enough = g[index] <= dip_fraction * g[peak]
        if is_local_min and deep_enough:
            minimum = index
            break
    if minimum is None:
        minimum = int(np.argmin(g[peak:])) + peak

    return {
        "peak_r": float(r[peak]),
        "peak_g": float(g[peak]),
        "minimum_r": float(r[minimum]),
        "coordination": float(profile["n_r"].to_numpy()[minimum]),
    }


def classify_shell(profile: pd.DataFrame, shell: dict,
                   normalised: bool = True) -> str:
    """``structured`` | ``excluded`` | ``bulk-like`` | ``featureless``.

    The distinction that matters is the second one.  A ``g(r)`` that stays
    below 1 everywhere does not mean the measurement failed: it means the
    solvent is *kept out* of that site, which is exactly what a buried oxygen
    or one donating its proton to the backbone looks like.  Reporting that as
    "no shell found" throws away the result.
    """
    if np.isfinite(shell.get("coordination", np.nan)):
        return "structured"
    if not normalised:
        return "featureless"
    peak = float(np.max(profile["g_r"].to_numpy())) if len(profile) else 0.0
    return "excluded" if peak < 0.60 else "bulk-like"


def bulk_tail(profile: pd.DataFrame, fraction: float = 0.25) -> float:
    """Mean ``g(r)`` over the outer ``fraction`` of the range.

    A normalisation that is working returns ~1 here.  A value far from 1 means
    either the range stops before bulk is reached or the centre sits inside a
    region the solvent cannot occupy, and in both cases the absolute scale of
    ``g(r)`` should not be quoted.
    """
    r = profile["r"].to_numpy()
    if r.size == 0:
        return float("nan")
    tail = r >= (r.min() + (1.0 - fraction) * (r.max() - r.min()))
    return float(np.mean(profile["g_r"].to_numpy()[tail])) if tail.any() else float("nan")


def shell_residents(residents: list[tuple[np.ndarray, np.ndarray]],
                    cutoff: float, n_frames: int) -> pd.DataFrame | None:
    """Which partner residues occupy the shell, and for how long.

    Returns one row per residue that ever enters ``cutoff``, with the number of
    frames it spends there, its longest uninterrupted stay and its mean closest
    approach — sorted by occupancy.  ``None`` when the cutoff falls outside the
    tracked radius, in which case the identities were never recorded and
    silently returning an empty table would read as "nobody was there".
    """
    if not np.isfinite(cutoff) or cutoff > RESIDENT_RADIUS or not residents:
        return None

    frames_of: dict[int, list[int]] = {}
    distances_of: dict[int, list[float]] = {}
    for step, (resids, distances) in enumerate(residents):
        inside = distances <= cutoff
        for resid, distance in zip(resids[inside], distances[inside]):
            frames_of.setdefault(int(resid), []).append(step)
            distances_of.setdefault(int(resid), []).append(float(distance))

    if not frames_of:
        return pd.DataFrame(columns=["resid", "frames", "occupancy_pct",
                                     "longest_frames", "mean_distance"])

    rows = [{
        "resid": resid,
        "frames": len(steps),
        "occupancy_pct": 100.0 * len(steps) / max(n_frames, 1),
        "longest_frames": _longest_run(steps),
        "mean_distance": float(np.mean(distances_of[resid])),
    } for resid, steps in frames_of.items()]
    return (pd.DataFrame(rows)
            .sort_values(["frames", "resid"], ascending=[False, True])
            .reset_index(drop=True))


def _longest_run(steps: list[int]) -> int:
    """Longest run of consecutive frames in an ascending list."""
    longest = current = 1
    for previous, value in zip(steps, steps[1:]):
        current = current + 1 if value == previous + 1 else 1
        longest = max(longest, current)
    return longest


def hydration_series(counts: np.ndarray, profile: pd.DataFrame,
                     cutoff: float) -> np.ndarray:
    """Number of partners within ``cutoff`` at every frame."""
    if not np.isfinite(cutoff):
        return np.full(counts.shape[0], np.nan)
    last_bin = int(np.searchsorted(profile["r"].to_numpy(), cutoff, side="right"))
    return counts[:, :last_bin].sum(axis=1)


def run(
    system: MDSystem,
    paths: OutputPaths,
    config: RDFConfig | None = None,
    verbose: bool = True,
) -> AnalysisResult:
    """Run every configured RDF and write profiles, shells and figures."""
    config = config or system.config.rdf
    formats = system.config.output.formats
    dpi = system.config.output.dpi

    tables: dict[str, pd.DataFrame] = {}
    plots: list = []
    rows: list[dict] = []
    notes: list[str] = []
    series_frames: list[pd.DataFrame] = []

    for spec in config.pairs:
        if verbose:
            print(f"[rdf] {spec.name}: {spec.g1} -> {spec.g2} (centre: {spec.center})")
        profile, counts, residents = compute_rdf(system, spec)
        normalised = is_normalised(spec)
        shell = first_shell(profile, normalised=normalised)
        regime = classify_shell(profile, shell, normalised)
        tail = bulk_tail(profile) if normalised else float("nan")
        name = safe_name(spec.name)
        tables[name] = profile
        write_csv(profile, paths.data, basename("rdf", name))
        plots += _plot_rdf(profile, shell, spec, paths, formats, dpi, normalised,
                           regime)

        # Without a detected minimum there is still a number worth reporting:
        # fall back to a fixed distance and say so, rather than leaving the
        # site with no occupancy at all.
        detected = bool(spec.shell_cutoff) or np.isfinite(shell["minimum_r"])
        if spec.shell_cutoff:
            cutoff = float(spec.shell_cutoff)
        elif np.isfinite(shell["minimum_r"]):
            cutoff = float(shell["minimum_r"])
        else:
            cutoff = FALLBACK_CUTOFF
        hydration = hydration_series(counts, profile, cutoff)
        if np.isfinite(cutoff) and np.isfinite(hydration).any():
            frame = pd.DataFrame(system.time_frame_columns())
            column = f"n_{name}"
            frame[column] = hydration
            series_frames.append(frame)
            tables[f"{name}_hydration"] = frame
            write_csv(frame, paths.data, basename("rdf", name, "_hydration"))
            plots += _plot_hydration(system, frame, column, spec, cutoff,
                                     paths, formats, dpi)
            qualifier = "detected shell" if detected else "fixed cutoff"
            rows.append(summary_row(
                "rdf", f"{spec.name}_n_shell", hydration, "molecules",
                definition=f"partners within {cutoff:.2f} Å of {spec.g1} "
                           f"({qualifier})",
            ))

        # Who is in the shell, and whether it is always the same molecule.
        occupants = shell_residents(residents, cutoff, system.n_frames)
        top = None
        if occupants is not None and not occupants.empty:
            tables[f"{name}_residents"] = occupants
            write_csv(occupants, paths.data, basename("rdf", name, "_residents"))
            plots += _plot_residents(occupants, spec, cutoff, paths, formats, dpi)
            top = occupants.iloc[0]

        rows.append({
            "analysis": "rdf",
            "observable": spec.name,
            "unit": "Å",
            "n_frames": system.n_frames,
            "mean": shell["peak_r"],
            "std": float("nan"),
            "min": float(spec.range[0]),
            "max": float(spec.range[1]),
            "definition": f"g(r) {spec.g1} - {spec.g2}",
            "regime": regime,
            "max_g": float(np.max(profile["g_r"].to_numpy())) if len(profile) else np.nan,
            "g_bulk_tail": tail,
            "first_peak_r": shell["peak_r"],
            "first_peak_g": shell["peak_g"],
            "shell_cutoff_r": cutoff if detected else np.nan,
            "coordination_number": shell["coordination"],
            "n_distinct_residents": (int(len(occupants))
                                     if occupants is not None else np.nan),
            "top_resident": int(top["resid"]) if top is not None else np.nan,
            "top_resident_pct": float(top["occupancy_pct"]) if top is not None else np.nan,
        })

        if regime == "structured":
            notes.append(
                f"'{spec.name}': first shell out to {shell['minimum_r']:.2f} Å "
                f"holding {shell['coordination']:.1f} partners "
                f"(peak at {shell['peak_r']:.2f} Å, g = {shell['peak_g']:.2f})."
            )
        elif regime == "excluded":
            peak_g = float(np.max(profile["g_r"].to_numpy()))
            notes.append(
                f"'{spec.name}': solvent is EXCLUDED from this site — g(r) never "
                f"exceeds {peak_g:.2f}, so it stays below bulk density at every "
                f"distance. This is a result, not a failed fit: the site is "
                f"buried or its donor is already engaged. Occupancy is reported "
                f"within a fixed {cutoff:.2f} Å instead of a detected minimum."
            )
        elif regime == "bulk-like":
            notes.append(
                f"'{spec.name}': no structured first shell — g(r) stays close to "
                f"bulk, so a coordination number would be meaningless. Occupancy "
                f"is reported within a fixed {cutoff:.2f} Å."
            )
        else:
            notes.append(
                f"'{spec.name}': the proximal distribution shows no clear first "
                f"minimum; occupancy within {cutoff:.2f} Å is reported instead."
            )

        if top is not None:
            notes.append(
                f"'{spec.name}': {len(occupants)} distinct molecule(s) visit the "
                f"shell; the most persistent is residue {int(top['resid'])}, "
                f"present in {top['occupancy_pct']:.1f}% of the frames "
                f"(longest uninterrupted stay {int(top['longest_frames'])} frames, "
                f"mean approach {top['mean_distance']:.2f} Å)."
            )
        elif occupants is None and np.isfinite(cutoff):
            notes.append(
                f"'{spec.name}': the shell reaches {cutoff:.2f} Å, beyond the "
                f"{RESIDENT_RADIUS:.1f} Å within which partner identities are "
                "tracked, so no occupancy per molecule is reported."
            )

        if normalised and np.isfinite(tail) and abs(tail - 1.0) > 0.25:
            notes.append(
                f"'{spec.name}': CAUTION — g(r) averages {tail:.2f} over the "
                f"outer quarter of the range instead of 1. Either the range "
                f"stops before bulk or the centre sits where solvent cannot go; "
                f"the absolute scale of g(r) should not be quoted."
            )

    series = None
    if series_frames:
        series = series_frames[0]
        for extra in series_frames[1:]:
            columns = [c for c in extra.columns
                       if c not in ("frame", "time", "replica")]
            series = series.merge(extra[["frame", *columns]], on="frame")

    notes.append("'mean' holds the position of the first g(r) peak; n(r) is the "
                 "running coordination number.")
    return AnalysisResult(
        name="rdf",
        title="Radial distribution functions and solvation",
        tables=tables,
        series=series,
        summary=pd.DataFrame(rows),
        plots=list(plots),
        notes=notes,
    )


def _plot_rdf(profile: pd.DataFrame, shell: dict, spec: RDFPair,
              paths: OutputPaths, formats, dpi, normalised: bool = True,
              regime: str = "structured") -> list:
    """g(r) with n(r) on a twin axis and the first shell highlighted."""
    import matplotlib.pyplot as plt

    colors = color_cycle(2)
    label = "g(r)" if normalised else "molecules / Å"
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.plot(profile["r"], profile["g_r"], color=colors[0], label=label)
    ax.set_xlabel("r (Å)" if normalised
                  else "distance to the nearest atom of the group (Å)")
    ax.set_ylabel(label)
    ax.set_xlim(*spec.range)
    if normalised:
        ax.axhline(1.0, color="0.6", linewidth=0.8, linestyle=":")

    if np.isfinite(shell["minimum_r"]):
        ax.axvspan(spec.range[0], shell["minimum_r"], color=colors[0], alpha=0.08)
        ax.axvline(shell["minimum_r"], color="0.35", linestyle="--", linewidth=1.0)
        ax.annotate(
            f"1st shell: {shell['coordination']:.1f} within "
            f"{shell['minimum_r']:.2f} Å",
            xy=(shell["minimum_r"], ax.get_ylim()[1]),
            xytext=(5, -8), textcoords="offset points",
            fontsize=7.5, color="0.3", va="top",
        )
    elif regime == "excluded":
        # Say what the flat curve means, so the panel is not read as a failure.
        ax.text(0.98, 0.94,
                f"solvent excluded\n(max g = {profile['g_r'].max():.2f})",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                color="0.3")

    twin = ax.twinx()
    twin.plot(profile["r"], profile["n_r"], color=colors[1], linestyle="--",
              label="n(r)")
    twin.set_ylabel("n(r) (coordination number)")
    twin.grid(False)

    lines = ax.get_lines()[:1] + twin.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc="lower right")
    ax.set_title(spec.name)
    fig.tight_layout()
    return save_figure(fig, paths.plots, basename("rdf", spec.name), formats, dpi)[:1]


def _plot_residents(occupants: pd.DataFrame, spec: RDFPair, cutoff: float,
                    paths: OutputPaths, formats, dpi, top_n: int = 10) -> list:
    """Occupancy of the shell per molecule.

    The shape of this plot answers a question the coordination number cannot:
    a few tall bars mean the same molecules stay put, a long flat tail means
    the shell is a revolving door with the same average population.
    """
    import matplotlib.pyplot as plt

    shown = occupants.head(top_n)
    colors = color_cycle(2)
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    positions = np.arange(len(shown))
    ax.bar(positions, shown["occupancy_pct"], width=0.6, color=colors[0], alpha=0.85)
    # keep room for at least four bars: two molecules drawn at full width read
    # as a pair of slabs rather than as a measurement
    ax.set_xlim(-0.75, max(len(shown) - 0.25, 3.75))
    ax.set_xticks(positions)
    ax.set_xticklabels([str(int(v)) for v in shown["resid"]], rotation=45,
                       ha="right", fontsize=7.5)
    ax.set_xlabel("Partner residue")
    ax.set_ylabel(f"% of frames within {cutoff:.2f} Å")
    ax.set_ylim(0, 100)
    exchanging = len(occupants) - len(shown)
    title = f"{spec.name}: who occupies the shell"
    if exchanging > 0:
        title += f"  ({exchanging} further molecule(s) not shown)"
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return save_figure(fig, paths.plots, basename("rdf", spec.name, "_residents"),
                       formats, dpi)[:1]


def _plot_hydration(system: MDSystem, frame: pd.DataFrame, column: str,
                    spec: RDFPair, cutoff: float, paths: OutputPaths,
                    formats, dpi) -> list:
    """How many partners occupy the first shell along the trajectory."""
    values = frame[column].to_numpy()
    fig = plot_timeseries(
        frame["time"].to_numpy(), {f"within {cutoff:.2f} Å": values},
        xlabel=system.time_label, ylabel="Molecules in the shell",
        title=f"{spec.name}: hydration number",
    )
    axis = fig.axes[0]
    axis.axhline(values.mean(), color="0.35", linestyle="--", linewidth=1.0)
    axis.text(0.995, values.mean(), f" mean {values.mean():.1f}", fontsize=8,
              color="0.35", transform=axis.get_yaxis_transform(),
              ha="right", va="bottom")
    return save_figure(fig, paths.plots, basename("rdf", spec.name, "_hydration"),
                       formats, dpi)[:1]
