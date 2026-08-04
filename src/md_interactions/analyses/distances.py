"""Interatomic distances along the trajectory.

Produces, for every distance defined in the configuration:

* a time series (CSV + figure),
* a normalised histogram with optional Gaussian KDE,
* summary statistics (mean ± std, min, max) and, if a ``threshold`` was given,
  the percentage of frames below it (population of near-attack conformations).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from MDAnalysis.lib.distances import calc_bonds, distance_array

from ..config import DistancesConfig
from ..io_utils import OutputPaths, write_csv
from ..plotting import (
    plot_distribution,
    plot_facet_distributions,
    plot_timeseries,
    save_figure,
)
from ..results import AnalysisResult, summary_row
from ..system import MDSystem
from ._common import (
    GeometryGroup,
    basename,
    current_box,
    replica_groups,
    resolve_group,
)

__all__ = ["run", "compute_distances"]

_UNIT = "Å"


def compute_distances(
    system: MDSystem, config: DistancesConfig | None = None
) -> pd.DataFrame:
    """Compute every configured distance and return a wide DataFrame.

    The returned frame has columns ``frame``, ``time`` and one column per
    distance (in ångström), which is exactly the CSV written to
    ``results/data/distances.csv``.
    """
    config = config or system.config.distances
    pairs: list[tuple[str, GeometryGroup, GeometryGroup, str]] = []
    for defn in config.pairs:
        group_a = resolve_group(system, defn.atoms[0], defn.mode, defn.name)
        group_b = resolve_group(system, defn.atoms[1], defn.mode, defn.name)
        pairs.append((defn.name, group_a, group_b, defn.mode))

    values = np.full((system.n_frames, len(pairs)), np.nan, dtype=float)
    for i, _ts in enumerate(system.iter_frames()):
        box = current_box(system, config.pbc)
        for j, (_name, group_a, group_b, mode) in enumerate(pairs):
            if mode == "min":
                values[i, j] = distance_array(
                    group_a.atoms.positions, group_b.atoms.positions, box=box
                ).min()
            else:
                pos_a = np.asarray(group_a.position(), dtype=np.float32).reshape(1, 3)
                pos_b = np.asarray(group_b.position(), dtype=np.float32).reshape(1, 3)
                values[i, j] = calc_bonds(pos_a, pos_b, box=box)[0]

    data = system.time_frame_columns()
    for j, (name, _a, _b, _m) in enumerate(pairs):
        data[name] = values[:, j]
    return pd.DataFrame(data)


def run(
    system: MDSystem,
    paths: OutputPaths,
    config: DistancesConfig | None = None,
    verbose: bool = True,
) -> AnalysisResult:
    """Run the distance analysis and write CSVs and figures."""
    config = config or system.config.distances
    if verbose:
        print(f"[distances] {len(config.pairs)} distance(s) over {system.n_frames} frames")

    df = compute_distances(system, config)
    write_csv(df, paths.data, "distances")

    labels = {d.name: (d.label or _auto_label(system, d)) for d in config.pairs}
    time = df["time"].to_numpy()
    plots: list[str] = []
    rows: list[dict] = []

    for defn in config.pairs:
        values = df[defn.name].to_numpy()
        base = basename("dist", defn.name)
        hlines = [(defn.threshold, f"{defn.threshold:g} {_UNIT}")] if defn.threshold else []

        fig = plot_timeseries(
            time,
            {labels[defn.name]: values},
            xlabel=system.time_label,
            ylabel=f"Distance ({_UNIT})",
            title=defn.name,
            smooth_window=config.running_average,
            hlines=hlines,
        )
        plots += save_figure(fig, paths.plots, f"{base}_timeseries",
                             system.config.output.formats, system.config.output.dpi)[:1]

        fig = plot_distribution(
            {labels[defn.name]: values},
            xlabel=f"Distance ({_UNIT})",
            bins=config.bins,
            kde=config.kde,
            title=defn.name,
            normalization=config.normalization,
            unit=_UNIT,
        )
        plots += save_figure(fig, paths.plots, f"{base}_histogram",
                             system.config.output.formats, system.config.output.dpi)[:1]

        extra: dict[str, float | str] = {"definition": labels[defn.name]}
        if defn.threshold is not None:
            below = float(np.mean(values < defn.threshold) * 100.0)
            extra["threshold"] = defn.threshold
            extra["frames_below_threshold_%"] = below
        rows.append(summary_row("distances", defn.name, values, _UNIT, **extra))

    if len(config.pairs) > 1:
        series = {labels[d.name]: df[d.name].to_numpy() for d in config.pairs}
        fig = plot_timeseries(
            time, series, xlabel=system.time_label, ylabel=f"Distance ({_UNIT})",
            title="Key distances", smooth_window=config.running_average,
            figsize=(7.2, 4.0),
        )
        plots += save_figure(fig, paths.plots, "dist_all_timeseries",
                             system.config.output.formats, system.config.output.dpi)[:1]

        if config.facet:
            # one panel per distance: the figure that goes into the paper
            fig = plot_facet_distributions(
                {d.name: df[d.name].to_numpy() for d in config.pairs},
                xlabel=f"Distance ({_UNIT})", unit=_UNIT,
                bins=min(config.bins, 40), kde=config.kde,
                normalization=config.normalization,
                groups=replica_groups(system, df, [d.name for d in config.pairs]),
            )
            plots += save_figure(fig, paths.plots, "dist_distributions",
                                 system.config.output.formats,
                                 system.config.output.dpi)[:1]
        else:
            fig = plot_distribution(series, xlabel=f"Distance ({_UNIT})",
                                    bins=config.bins, kde=config.kde,
                                    title="Key distances", figsize=(6.0, 4.0),
                                    normalization=config.normalization, unit=_UNIT)
            plots += save_figure(fig, paths.plots, "dist_all_histogram",
                                 system.config.output.formats,
                                 system.config.output.dpi)[:1]

    return AnalysisResult(
        name="distances",
        title="Key distances",
        tables={"distances": df},
        series=df,
        summary=pd.DataFrame(rows),
        plots=list(plots),
        notes=[
            f"Minimum-image convention: {'on' if config.pbc else 'off'}.",
        ],
    )


def _auto_label(system: MDSystem, defn) -> str:
    """Build a readable label such as ``SER145:OG – LIG:C1``."""
    try:
        left = system.describe_selection(defn.atoms[0])
        right = system.describe_selection(defn.atoms[1])
    except Exception:  # pragma: no cover - selection errors surface earlier
        left, right = defn.atoms
    return f"{left} – {right}"
