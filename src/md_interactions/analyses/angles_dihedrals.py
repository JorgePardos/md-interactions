"""User-defined angles and dihedrals along the trajectory.

Angles are reported in [0, 180]°, dihedrals in (-180, 180]°.  Dihedral
statistics use circular averages (``scipy.stats.circmean``/``circstd``), which
is what you want for an angle that visits both sides of ±180°.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from MDAnalysis.lib.distances import calc_angles, calc_dihedrals

from ..config import AnglesConfig, GeometryDef
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

__all__ = ["run", "compute_angles"]

_UNIT = "°"


def compute_angles(
    system: MDSystem, config: AnglesConfig | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute all configured angles and dihedrals.

    Returns
    -------
    (angles, dihedrals)
        Two wide DataFrames with ``frame``/``time`` plus one column per
        observable, in degrees.  Either can be empty.
    """
    config = config or system.config.angles

    angle_groups = [
        (defn.name, [resolve_group(system, tok, defn.mode, defn.name) for tok in defn.atoms])
        for defn in config.angles
    ]
    dihedral_groups = [
        (defn.name, [resolve_group(system, tok, defn.mode, defn.name) for tok in defn.atoms])
        for defn in config.dihedrals
    ]

    n_frames = system.n_frames
    angle_values = np.full((n_frames, len(angle_groups)), np.nan)
    dihedral_values = np.full((n_frames, len(dihedral_groups)), np.nan)

    for i, _ts in enumerate(system.iter_frames()):
        box = current_box(system, config.pbc)
        for j, (_name, groups) in enumerate(angle_groups):
            a, b, c = (_point(g) for g in groups)
            angle_values[i, j] = np.degrees(calc_angles(a, b, c, box=box)[0])
        for j, (_name, groups) in enumerate(dihedral_groups):
            a, b, c, d = (_point(g) for g in groups)
            dihedral_values[i, j] = np.degrees(calc_dihedrals(a, b, c, d, box=box)[0])

    angles_df = pd.DataFrame(system.time_frame_columns())
    for j, (name, _g) in enumerate(angle_groups):
        angles_df[name] = angle_values[:, j]
    dihedrals_df = pd.DataFrame(system.time_frame_columns())
    for j, (name, _g) in enumerate(dihedral_groups):
        dihedrals_df[name] = dihedral_values[:, j]
    return angles_df, dihedrals_df


def _point(group: GeometryGroup) -> np.ndarray:
    return np.asarray(group.position(), dtype=np.float32).reshape(1, 3)


def run(
    system: MDSystem,
    paths: OutputPaths,
    config: AnglesConfig | None = None,
    verbose: bool = True,
) -> AnalysisResult:
    """Run the angle/dihedral analysis and write CSVs and figures."""
    config = config or system.config.angles
    if verbose:
        print(f"[angles_dihedrals] {len(config.angles)} angle(s), "
              f"{len(config.dihedrals)} dihedral(s) over {system.n_frames} frames")

    angles_df, dihedrals_df = compute_angles(system, config)
    tables: dict[str, pd.DataFrame] = {}
    plots: list = []
    rows: list[dict] = []
    formats = system.config.output.formats
    dpi = system.config.output.dpi

    if config.angles:
        tables["angles"] = angles_df
        write_csv(angles_df, paths.data, "angles")
        plots += _plot_group(system, paths, config, angles_df, config.angles,
                             prefix="angle", periodic=(0.0, 180.0),
                             ylabel=f"Angle ({_UNIT})", formats=formats, dpi=dpi)
        for defn in config.angles:
            values = angles_df[defn.name].to_numpy()
            rows.append(summary_row("angles", defn.name, values, _UNIT,
                                    definition=_label(system, defn)))

    if config.dihedrals:
        tables["dihedrals"] = dihedrals_df
        write_csv(dihedrals_df, paths.data, "dihedrals")
        plots += _plot_group(system, paths, config, dihedrals_df, config.dihedrals,
                             prefix="dihedral", periodic=(-180.0, 180.0),
                             ylabel=f"Dihedral ({_UNIT})", formats=formats, dpi=dpi)
        for defn in config.dihedrals:
            values = dihedrals_df[defn.name].to_numpy()
            rows.append(_circular_summary(defn.name, values, _label(system, defn)))

    series = angles_df
    if config.dihedrals:
        extra = [c for c in dihedrals_df.columns if c not in ("frame", "time")]
        series = (angles_df.merge(dihedrals_df[["frame", *extra]], on="frame")
                  if config.angles else dihedrals_df)

    return AnalysisResult(
        name="angles_dihedrals",
        title="Angles and dihedrals",
        tables=tables,
        series=series,
        summary=pd.DataFrame(rows),
        plots=list(plots),
        notes=["Dihedral statistics are circular (circmean / circstd)."],
    )


def _label(system: MDSystem, defn: GeometryDef) -> str:
    if defn.label:
        return defn.label
    try:
        return " – ".join(system.describe_selection(tok) for tok in defn.atoms)
    except Exception:  # pragma: no cover
        return " – ".join(defn.atoms)


def _plot_group(system, paths, config, df, definitions, prefix, periodic, ylabel,
                formats, dpi) -> list:
    plots: list = []
    time = df["time"].to_numpy()
    labels = {d.name: _label(system, d) for d in definitions}

    for defn in definitions:
        values = df[defn.name].to_numpy()
        base = basename(prefix, defn.name)
        fig = plot_timeseries(time, {labels[defn.name]: values},
                              xlabel=system.time_label, ylabel=ylabel, title=defn.name)
        plots += save_figure(fig, paths.plots, f"{base}_timeseries", formats, dpi)[:1]
        fig = plot_distribution({labels[defn.name]: values}, xlabel=ylabel,
                                bins=config.bins, kde=config.kde, title=defn.name,
                                periodic=periodic, normalization=config.normalization,
                                unit=_UNIT)
        plots += save_figure(fig, paths.plots, f"{base}_histogram", formats, dpi)[:1]

    if len(definitions) > 1:
        series = {labels[d.name]: df[d.name].to_numpy() for d in definitions}
        fig = plot_timeseries(time, series, xlabel=system.time_label, ylabel=ylabel,
                              title=None, figsize=(7.2, 4.0))
        plots += save_figure(fig, paths.plots, f"{prefix}_all_timeseries", formats, dpi)[:1]
        if config.facet:
            names = [d.name for d in definitions]
            fig = plot_facet_distributions(
                {name: df[name].to_numpy() for name in names},
                xlabel=ylabel, unit=_UNIT, bins=min(config.bins, 40), kde=config.kde,
                normalization=config.normalization, periodic=periodic,
                groups=replica_groups(system, df, names),
            )
            plots += save_figure(fig, paths.plots, f"{prefix}_distributions",
                                 formats, dpi)[:1]
        else:
            fig = plot_distribution(series, xlabel=ylabel, bins=config.bins,
                                    kde=config.kde, periodic=periodic,
                                    figsize=(6.0, 4.0),
                                    normalization=config.normalization, unit=_UNIT)
            plots += save_figure(fig, paths.plots, f"{prefix}_all_histogram",
                                 formats, dpi)[:1]
    return plots


def _circular_summary(name: str, values: np.ndarray, definition: str) -> dict:
    """Summary row for a dihedral, using circular mean/std."""
    from scipy.stats import circmean, circstd

    clean = values[~np.isnan(values)]
    if clean.size == 0:  # pragma: no cover - guarded upstream
        return summary_row("dihedrals", name, clean, _UNIT, definition=definition)
    mean = float(circmean(clean, low=-180.0, high=180.0))
    std = float(circstd(clean, low=-180.0, high=180.0))
    return {
        "analysis": "dihedrals",
        "observable": name,
        "unit": _UNIT,
        "n_frames": int(clean.size),
        "mean": mean,
        "std": std,
        "min": float(clean.min()),
        "max": float(clean.max()),
        "definition": definition,
        "circular": True,
    }
