"""Radius of gyration of one or more selections along the trajectory.

Useful to monitor compaction/opening of a domain or of the active-site pocket
during the simulation.
"""

from __future__ import annotations

import MDAnalysis as mda
import numpy as np
import pandas as pd

from ..config import RgyrConfig
from ..exceptions import MissingTopologyInfoError
from ..io_utils import OutputPaths, write_csv
from ..plotting import plot_distribution, plot_timeseries, save_figure
from ..results import AnalysisResult, summary_row
from ..system import MDSystem
from ._common import basename

__all__ = ["run", "compute_rgyr"]

_UNIT = "Å"


def compute_rgyr(system: MDSystem, config: RgyrConfig | None = None) -> pd.DataFrame:
    """Mass-weighted radius of gyration of every configured group (Å)."""
    config = config or system.config.rgyr
    groups = [(g.name, system.select(g.selection, name=f"rgyr:{g.name}"))
              for g in config.groups]

    for name, atoms in groups:
        try:
            if not np.isfinite(atoms.masses).all() or atoms.masses.sum() <= 0:
                raise mda.exceptions.NoDataError
        except (mda.exceptions.NoDataError, AttributeError) as exc:
            raise MissingTopologyInfoError(
                f"The radius of gyration of '{name}' needs atomic masses, which "
                "this topology does not provide. An AMBER prmtop always carries "
                "them; a bare PDB may not."
            ) from exc

    values = np.full((system.n_frames, len(groups)), np.nan)
    for i, _ts in enumerate(system.iter_frames()):
        for j, (_name, atoms) in enumerate(groups):
            values[i, j] = atoms.radius_of_gyration()

    df = pd.DataFrame(system.time_frame_columns())
    for j, (name, _atoms) in enumerate(groups):
        df[name] = values[:, j]
    return df


def run(
    system: MDSystem,
    paths: OutputPaths,
    config: RgyrConfig | None = None,
    verbose: bool = True,
) -> AnalysisResult:
    """Run the radius-of-gyration analysis and write CSVs and figures."""
    config = config or system.config.rgyr
    if verbose:
        print(f"[radius_of_gyration] {len(config.groups)} group(s)")

    df = compute_rgyr(system, config)
    write_csv(df, paths.data, "radius_of_gyration")
    formats = system.config.output.formats
    dpi = system.config.output.dpi
    time = df["time"].to_numpy()

    plots: list = []
    rows: list[dict] = []
    for group in config.groups:
        values = df[group.name].to_numpy()
        fig = plot_timeseries(time, {group.name: values}, xlabel=system.time_label,
                              ylabel=f"$R_g$ ({_UNIT})", title=group.name)
        plots += save_figure(fig, paths.plots, basename("rgyr", group.name, "_timeseries"),
                             formats, dpi)[:1]
        fig = plot_distribution({group.name: values}, xlabel=f"$R_g$ ({_UNIT})",
                                bins=50, kde=True, title=group.name)
        plots += save_figure(fig, paths.plots, basename("rgyr", group.name, "_histogram"),
                             formats, dpi)[:1]
        rows.append(summary_row("radius_of_gyration", group.name, values, _UNIT,
                                definition=group.selection))

    if len(config.groups) > 1:
        fig = plot_timeseries(time, {g.name: df[g.name].to_numpy() for g in config.groups},
                              xlabel=system.time_label, ylabel=f"$R_g$ ({_UNIT})",
                              figsize=(7.2, 4.0))
        plots += save_figure(fig, paths.plots, "rgyr_all_timeseries", formats, dpi)[:1]

    return AnalysisResult(
        name="radius_of_gyration",
        title="Radius of gyration",
        tables={"radius_of_gyration": df},
        series=df,
        summary=pd.DataFrame(rows),
        plots=list(plots),
    )
