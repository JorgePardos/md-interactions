"""Distributions of pre-computed observables (``data:`` mode).

Same figures as the trajectory-based modules — multi-panel distributions, time
series and a summary table — but reading a :class:`~md_interactions.tabular.Dataset`
instead of a topology and a trajectory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..io_utils import OutputPaths, write_csv
from ..plotting import (
    plot_distribution,
    plot_facet_distributions,
    plot_timeseries,
    save_figure,
)
from ..results import AnalysisResult, summary_row
from ..tabular import Dataset
from ._common import basename, safe_name

__all__ = ["run"]


def run(
    dataset: Dataset,
    paths: OutputPaths,
    verbose: bool = True,
) -> AnalysisResult:
    """Plot and summarise every column of the dataset."""
    config = dataset.config.data
    output = dataset.config.output
    formats, dpi = output.formats, output.dpi
    unit = dataset.unit
    xlabel = f"Distance ({unit})" if unit else "Value"

    if verbose:
        print(f"[distributions] {len(dataset.columns)} column(s), "
              f"{dataset.n_frames} rows")

    write_csv(dataset.data, paths.data, "observables")
    plots: list = []
    rows: list[dict] = []

    groups = {name: {column: dataset.data.loc[mask, column].to_numpy()
                     for column in dataset.columns}
              for name, mask in dataset.replica_masks().items()} or None

    # -- one panel per observable ----------------------------------------- #
    if config.facet and len(dataset.columns) > 1:
        fig = plot_facet_distributions(
            {column: dataset.values(column) for column in dataset.columns},
            xlabel=xlabel, unit=unit, bins=config.bins, kde=config.kde,
            normalization=config.normalization, groups=groups,
        )
        plots += save_figure(fig, paths.plots, "distributions", formats, dpi)[:1]

    # -- individual panels, for single-figure use -------------------------- #
    for column in dataset.columns:
        values = dataset.values(column)
        series = {column: values}
        if groups:
            series = {f"{column} ({name})": groups[name][column] for name in groups}
        fig = plot_distribution(series, xlabel=xlabel, bins=config.bins,
                                kde=config.kde, title=column,
                                normalization=config.normalization, unit=unit)
        plots += save_figure(fig, paths.plots, basename("dist", column, "_histogram"),
                             formats, dpi)[:1]

        threshold = config.thresholds.get(column)
        extra: dict[str, float] = {}
        if threshold is not None:
            extra["threshold"] = threshold
            extra["frames_below_threshold_%"] = float(
                np.mean(values < threshold) * 100.0
            )
        rows.append(summary_row("distributions", column, values, unit, **extra))

    # -- time series ------------------------------------------------------- #
    if config.timeseries:
        for column in dataset.columns:
            fig = plot_timeseries(
                dataset.times, {column: dataset.values(column)},
                xlabel=dataset.time_label, ylabel=f"{column} ({unit})" if unit else column,
                title=column,
            )
            plots += save_figure(fig, paths.plots,
                                 basename("dist", column, "_timeseries"),
                                 formats, dpi)[:1]

    return AnalysisResult(
        name="distributions",
        title="Distributions of the measured observables",
        tables={"observables": dataset.data},
        series=dataset.data,
        summary=pd.DataFrame(rows),
        plots=list(plots),
        notes=[
            f"Source files: {', '.join(p.name for p in dataset.sources)}.",
            "Histograms are normalised as "
            f"'{config.normalization}'.",
        ],
    )
