"""Solvent molecules bridging two groups.

Two radial distribution functions can both show a well-populated first shell
without a single molecule ever touching *both* groups at once.  When the
question is whether a water can relay a proton between a donor and an acceptor,
what matters is the intersection: the molecules that are simultaneously within
reach of the two.

For every pair this reports how often such a bridge exists, how many there are,
which molecule it is, the two distances that define it, and how long the same
molecule stays in place — a bridge that survives tens of picoseconds is a very
different thing from one that is remade by a different water every frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from MDAnalysis.lib.distances import capped_distance

from ..config import BridgeConfig, BridgePair
from ..exceptions import AnalysisError
from ..io_utils import OutputPaths, write_csv
from ..plotting import color_cycle, plot_timeseries, save_figure
from ..results import AnalysisResult, summary_row
from ..system import MDSystem
from ._common import basename, current_box, safe_name

__all__ = ["run", "compute_bridges"]


def compute_bridges(system: MDSystem, spec: BridgePair) -> pd.DataFrame:
    """Per-frame description of the bridges between the two groups.

    Columns: ``n_bridges`` (how many solvent molecules bridge at that frame),
    ``resid`` of the best one, ``d_a``/``d_b`` (its distances to each group)
    and ``d_direct`` (the direct group-to-group distance, for context).
    """
    group_a = system.select(spec.group_a, name=f"bridge:{spec.name}.a")
    group_b = system.select(spec.group_b, name=f"bridge:{spec.name}.b")
    solvent = system.select(spec.solvent, name=f"bridge:{spec.name}.solvent")
    if solvent.n_atoms == 0:
        raise AnalysisError(f"'{spec.name}': the solvent selection is empty.")

    resids = np.asarray(solvent.resids)
    n_frames = system.n_frames
    counts = np.zeros(n_frames, dtype=int)
    best_resid = np.full(n_frames, -1, dtype=int)
    d_a = np.full(n_frames, np.nan)
    d_b = np.full(n_frames, np.nan)
    d_direct = np.full(n_frames, np.nan)

    for step, _ts in enumerate(system.iter_frames()):
        box = current_box(system, True)
        near_a = _within(group_a.positions, solvent.positions, spec.cutoff, box)
        if not near_a:
            continue
        near_b = _within(group_b.positions, solvent.positions, spec.cutoff, box)
        shared = set(near_a) & set(near_b)
        if not shared:
            continue

        counts[step] = len({int(resids[i]) for i in shared})
        # the "best" bridge is the one with the shortest total path A-W-B
        index = min(shared, key=lambda i: near_a[i] + near_b[i])
        best_resid[step] = int(resids[index])
        d_a[step] = near_a[index]
        d_b[step] = near_b[index]
        d_direct[step] = _min_distance(group_a.positions, group_b.positions, box)

    frame = pd.DataFrame(system.time_frame_columns())
    frame["n_bridges"] = counts
    frame["resid"] = best_resid
    frame["d_a"] = d_a
    frame["d_b"] = d_b
    frame["d_direct"] = d_direct
    return frame


def _within(reference: np.ndarray, candidates: np.ndarray, cutoff: float,
            box) -> dict[int, float]:
    """``{candidate index: distance}`` for candidates within ``cutoff``."""
    pairs, distances = capped_distance(reference, candidates, max_cutoff=cutoff,
                                       box=box, return_distances=True)
    best: dict[int, float] = {}
    for (_i, j), distance in zip(pairs, distances):
        j = int(j)
        if j not in best or distance < best[j]:
            best[j] = float(distance)
    return best


def _min_distance(first: np.ndarray, second: np.ndarray, box) -> float:
    pairs, distances = capped_distance(first, second, max_cutoff=99.0, box=box,
                                       return_distances=True)
    return float(distances.min()) if len(distances) else np.nan


def residence_statistics(frame: pd.DataFrame) -> dict:
    """How long the same molecule keeps bridging, in frames.

    Reports the longest uninterrupted stretch and how many distinct molecules
    take turns, which is what separates a structural water from a position that
    merely stays solvated.
    """
    resids = frame["resid"].to_numpy()
    present = resids >= 0
    if not present.any():
        return {"longest_frames": 0, "n_distinct": 0, "most_common": None,
                "most_common_frames": 0}

    longest = current = 0
    previous = -1
    for value, is_present in zip(resids, present):
        if is_present and value == previous:
            current += 1
        elif is_present:
            current = 1
        else:
            current = 0
        previous = value if is_present else -1
        longest = max(longest, current)

    values, counts = np.unique(resids[present], return_counts=True)
    order = int(np.argmax(counts))
    return {
        "longest_frames": int(longest),
        "n_distinct": int(values.size),
        "most_common": int(values[order]),
        "most_common_frames": int(counts[order]),
    }


def run(
    system: MDSystem,
    paths: OutputPaths,
    config: BridgeConfig | None = None,
    verbose: bool = True,
) -> AnalysisResult:
    """Run every configured bridge analysis and write tables and figures."""
    config = config or system.config.bridges
    formats = system.config.output.formats
    dpi = system.config.output.dpi

    tables: dict[str, pd.DataFrame] = {}
    rows: list[dict] = []
    plots: list = []
    notes: list[str] = []
    series_frames: list[pd.DataFrame] = []

    for spec in config.pairs:
        if verbose:
            print(f"[bridges] {spec.name}: {spec.group_a} <-> {spec.group_b} "
                  f"(cutoff {spec.cutoff:g} Å)")
        frame = compute_bridges(system, spec)
        name = safe_name(spec.name)
        tables[name] = frame
        write_csv(frame, paths.data, basename("bridge", name))

        occupancy = 100.0 * float(np.mean(frame["n_bridges"] > 0))
        residence = residence_statistics(frame)
        bridged = frame[frame["n_bridges"] > 0]

        series = pd.DataFrame(system.time_frame_columns())
        series[f"nbridge_{name}"] = frame["n_bridges"].to_numpy()
        series_frames.append(series)

        rows.append(summary_row(
            "bridges", spec.name, frame["n_bridges"].to_numpy(), "molecules",
            definition=f"{spec.group_a} <-> {spec.group_b} within {spec.cutoff:g} Å",
            occupancy_pct=occupancy,
        ))
        if not bridged.empty:
            rows.append(summary_row("bridges", f"{spec.name}_d_a",
                                    bridged["d_a"].to_numpy(), "Å",
                                    definition=f"{spec.group_a} ··· bridging solvent"))
            rows.append(summary_row("bridges", f"{spec.name}_d_b",
                                    bridged["d_b"].to_numpy(), "Å",
                                    definition=f"solvent ··· {spec.group_b}"))
            rows.append(summary_row("bridges", f"{spec.name}_d_direct",
                                    frame["d_direct"].to_numpy(), "Å",
                                    definition=f"{spec.group_a} ··· {spec.group_b} direct"))

        plots += _plot_bridges(system, frame, spec, paths, formats, dpi)

        if occupancy == 0:
            notes.append(
                f"'{spec.name}': no solvent molecule is ever within "
                f"{spec.cutoff:g} Å of both groups at the same time."
            )
        else:
            notes.append(
                f"'{spec.name}': a bridge exists in {occupancy:.1f}% of the frames; "
                f"{residence['n_distinct']} distinct molecule(s) take part, the "
                f"longest uninterrupted one lasting {residence['longest_frames']} "
                f"frames (most frequent: residue {residence['most_common']}, "
                f"{residence['most_common_frames']} frames)."
            )

    series = None
    if series_frames:
        series = series_frames[0]
        for extra in series_frames[1:]:
            columns = [c for c in extra.columns
                       if c not in ("frame", "time", "replica")]
            series = series.merge(extra[["frame", *columns]], on="frame")

    notes.append("A bridge is a solvent molecule within the cutoff of both "
                 "groups at the same frame; the reported distances belong to "
                 "the one with the shortest total path.")
    return AnalysisResult(
        name="bridges",
        title="Bridging solvent molecules",
        tables=tables,
        series=series,
        summary=pd.DataFrame(rows),
        plots=list(plots),
        notes=notes,
    )


def _plot_bridges(system: MDSystem, frame: pd.DataFrame, spec: BridgePair,
                  paths: OutputPaths, formats, dpi) -> list:
    """Number of bridges over time, and the geometry of the best one."""
    import matplotlib.pyplot as plt

    plots: list = []
    time = frame["time"].to_numpy()
    colors = color_cycle(3)

    figure = plot_timeseries(
        time, {"bridging molecules": frame["n_bridges"].to_numpy()},
        xlabel=system.time_label, ylabel="Molecules bridging",
        title=f"{spec.name}: bridges over time",
    )
    axis = figure.axes[0]
    axis.set_ylim(bottom=0)
    occupancy = 100.0 * float(np.mean(frame["n_bridges"] > 0))
    axis.text(0.01, 0.95, f"present in {occupancy:.1f}% of frames",
              transform=axis.transAxes, fontsize=8, va="top", color="0.35")
    plots += save_figure(figure, paths.plots, basename("bridge", spec.name, "_count"),
                         formats, dpi)[:1]

    bridged = frame[frame["n_bridges"] > 0]
    if bridged.empty:
        return plots

    figure, (ax_time, ax_scatter) = plt.subplots(
        1, 2, figsize=(8.6, 3.4), gridspec_kw={"width_ratios": [3, 2]})
    ax_time.plot(bridged["time"], bridged["d_a"], ".", markersize=2.5,
                 color=colors[0], label=f"{spec.group_a} ··· W")
    ax_time.plot(bridged["time"], bridged["d_b"], ".", markersize=2.5,
                 color=colors[1], label=f"W ··· {spec.group_b}")
    ax_time.plot(frame["time"], frame["d_direct"], "-", linewidth=0.8,
                 color=colors[2], alpha=0.7, label="direct")
    ax_time.set_xlabel(system.time_label)
    ax_time.set_ylabel("Distance (Å)")
    ax_time.legend(fontsize=7, loc="best")

    ax_scatter.plot(bridged["d_a"], bridged["d_b"], ".", markersize=2.5,
                    color=colors[0], alpha=0.5)
    ax_scatter.set_xlabel(f"d({spec.group_a} ··· W) (Å)")
    ax_scatter.set_ylabel(f"d(W ··· {spec.group_b}) (Å)")
    ax_scatter.axhline(spec.cutoff, color="0.5", linestyle=":", linewidth=0.8)
    ax_scatter.axvline(spec.cutoff, color="0.5", linestyle=":", linewidth=0.8)
    figure.suptitle(f"{spec.name}: geometry of the shortest bridge", fontsize=10)
    figure.tight_layout()
    plots += save_figure(figure, paths.plots,
                         basename("bridge", spec.name, "_geometry"), formats, dpi)[:1]
    return plots
