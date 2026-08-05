"""Radial distribution functions and solvation shells.

Built for the question that comes up most often in practice: *how is this atom
(or this residue) hydrated?*  For every pair it reports

* ``g(r)`` — the radial distribution function,
* ``n(r)`` — the running coordination number,
* the **first solvation shell**: the first minimum of ``g(r)`` and the number of
  partners inside it,
* the **hydration number over time** — how many waters sit inside that shell at
  each frame, which is what shows a shell being lost or exchanged.

The histogram is accumulated frame by frame, so the averaged ``g(r)`` and the
time series come out of a single pass over the trajectory.
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

__all__ = ["run", "compute_rdf", "first_shell", "hydration_series"]


def compute_rdf(system: MDSystem, spec: RDFPair) -> tuple[pd.DataFrame, np.ndarray]:
    """RDF, running coordination number and the per-frame histogram.

    Returns ``(profile, counts)`` where ``profile`` has columns ``r``, ``g_r``
    and ``n_r``, and ``counts`` is the ``(n_frames, nbins)`` histogram, kept so
    that the hydration time series needs no second pass.

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

    for step, _ts in enumerate(system.iter_frames()):
        box = current_box(system, True)
        centres = (group1.center_of_mass().reshape(1, 3).astype(np.float32)
                   if mode == "com" else group1.positions)
        pairs, distances = capped_distance(
            centres, group2.positions, max_cutoff=float(rmax), box=box,
            return_distances=True,
        )
        if len(pairs):
            keep = distances > 1e-6                     # drop self-pairs
            if spec.exclude_same_residue and mode == "atom":
                same = group1.resindices[pairs[:, 0]] == partner_residues[pairs[:, 1]]
                keep &= ~same
            pairs, distances = pairs[keep], distances[keep]

            if mode == "proximal":
                # one distance per partner: how far it is from the *closest*
                # atom of the group, which is what "water around this residue"
                # actually means for a non-spherical solute
                order = np.argsort(distances)
                _unique, first = np.unique(pairs[order, 1], return_index=True)
                distances = distances[order][first]
            counts[step] = np.histogram(distances, bins=edges)[0]
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
        return profile, counts

    n_centres = 1 if mode == "com" else group1.n_atoms
    density = group2.n_atoms / float(np.mean(volumes))
    per_centre = mean_counts / n_centres
    shell_volume = 4.0 * np.pi * centres_r ** 2 * widths
    with np.errstate(divide="ignore", invalid="ignore"):
        g_r = np.nan_to_num(per_centre / (shell_volume * density))
    profile = pd.DataFrame({"r": centres_r, "g_r": g_r,
                            "n_r": np.cumsum(per_centre)})
    return profile, counts


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
        profile, counts = compute_rdf(system, spec)
        normalised = is_normalised(spec)
        shell = first_shell(profile, normalised=normalised)
        name = safe_name(spec.name)
        tables[name] = profile
        write_csv(profile, paths.data, basename("rdf", name))
        plots += _plot_rdf(profile, shell, spec, paths, formats, dpi, normalised)

        cutoff = spec.shell_cutoff if spec.shell_cutoff else shell["minimum_r"]
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
            rows.append(summary_row(
                "rdf", f"{spec.name}_n_shell", hydration, "molecules",
                definition=f"partners within {cutoff:.2f} Å of {spec.g1}",
            ))

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
            "first_peak_r": shell["peak_r"],
            "first_peak_g": shell["peak_g"],
            "shell_cutoff_r": shell["minimum_r"],
            "coordination_number": shell["coordination"],
        })
        if np.isfinite(shell["coordination"]):
            notes.append(
                f"'{spec.name}': first shell out to {shell['minimum_r']:.2f} Å "
                f"holding {shell['coordination']:.1f} partners "
                f"(peak at {shell['peak_r']:.2f} Å, g = {shell['peak_g']:.2f})."
            )
        else:
            notes.append(
                f"'{spec.name}': no structured first shell — g(r) never rises "
                "clearly above 1, so a coordination number would be meaningless."
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
              paths: OutputPaths, formats, dpi, normalised: bool = True) -> list:
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
