"""RMSD (global and local) and per-residue RMSF.

RMSD
    One curve per configured group.  A group may declare a ``superposition``
    selection different from the measured one, which is the usual way of
    reporting a *local* RMSD (e.g. fit on the protein backbone, measure the
    ligand or the active site).

RMSF
    Computed on the selection requested by the user after aligning every frame
    to the average structure (two-pass iterative fit, the standard recipe).
    Only the coordinates of the involved atoms are held in memory, so this
    works on large systems as long as the selection is reasonable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import align, rms

from ..config import RMSDConfig, RMSFConfig
from ..exceptions import AnalysisError
from ..io_utils import OutputPaths, write_csv
from ..plotting import color_cycle, plot_timeseries, save_figure
from ..results import AnalysisResult, summary_row
from ..system import MDSystem
from ._common import basename

__all__ = ["run", "compute_rmsd", "compute_rmsf"]

_UNIT = "Å"


# --------------------------------------------------------------------------- #
# RMSD
# --------------------------------------------------------------------------- #
def compute_rmsd(system: MDSystem, config: RMSDConfig | None = None) -> pd.DataFrame:
    """RMSD time series (one column per group), in ångström."""
    config = config or system.config.rmsd
    reference = _reference_universe(system, config)

    # Groups sharing the same superposition selection are computed in one pass.
    by_fit: dict[str, list] = {}
    for group in config.groups:
        fit = group.superposition or group.selection
        by_fit.setdefault(fit, []).append(group)

    df = pd.DataFrame(system.time_frame_columns())
    for fit_token, groups in by_fit.items():
        fit_sel = system.config.resolve_selection(fit_token)
        system.select(fit_token, name="rmsd.superposition")  # validate on mobile
        group_sels = []
        for group in groups:
            system.select(group.selection, name=f"rmsd:{group.name}")
            group_sels.append(system.config.resolve_selection(group.selection))

        analysis = rms.RMSD(
            system.universe,
            reference,
            select=fit_sel,
            groupselections=group_sels,
        ).run(**system.run_kwargs)
        # columns: frame, time, rmsd(select), rmsd(groupselections...)
        values = analysis.results.rmsd
        for j, group in enumerate(groups):
            df[group.name] = values[:, 3 + j]
    return df


def _reference_universe(system: MDSystem, config: RMSDConfig) -> mda.Universe:
    """Reference structure: an external file or a frame of the trajectory."""
    topology = str(system.config.system.topology)
    if config.reference is not None:
        try:
            return mda.Universe(topology, str(config.reference))
        except Exception as exc:
            raise AnalysisError(
                f"Could not read the RMSD reference '{config.reference}': {exc}"
            ) from exc
    reference = mda.Universe(topology, *[str(p) for p in system.config.system.trajectory])
    n_frames = len(reference.trajectory)
    if not 0 <= config.ref_frame < n_frames:
        raise AnalysisError(
            f"'analyses.rmsd.ref_frame' = {config.ref_frame} is out of range "
            f"(trajectory has {n_frames} frames)."
        )
    reference.trajectory[config.ref_frame]
    return reference


# --------------------------------------------------------------------------- #
# RMSF
# --------------------------------------------------------------------------- #
def compute_rmsf(
    system: MDSystem, config: RMSFConfig | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-atom and per-residue RMSF of the configured selection.

    Returns
    -------
    (per_atom, per_residue)
        ``per_atom`` has columns ``index, resid, resname, name, rmsf``;
        ``per_residue`` has ``resid, resname, n_atoms, rmsf`` where the
        residue value is the mass-weighted RMS of its atoms.
    """
    config = config or system.config.rmsf
    target = system.select(config.selection, name="rmsf.selection")
    fit = (system.select(config.align_selection, name="rmsf.align_selection")
           if config.align_selection else target)

    combined = (target + fit).unique
    lookup = {index: i for i, index in enumerate(combined.indices)}
    target_idx = np.array([lookup[i] for i in target.indices])
    fit_idx = np.array([lookup[i] for i in fit.indices])

    coords = np.empty((system.n_frames, combined.n_atoms, 3), dtype=np.float64)
    for i, _ts in enumerate(system.iter_frames()):
        coords[i] = combined.positions

    try:
        weights = np.asarray(fit.masses, dtype=float)
        if not np.all(np.isfinite(weights)) or weights.sum() <= 0:
            weights = None
    except (mda.exceptions.NoDataError, AttributeError):
        weights = None

    if config.align:
        coords = _iterative_fit(coords, fit_idx, weights)

    positions = coords[:, target_idx, :]
    fluctuations = positions - positions.mean(axis=0, keepdims=True)
    rmsf = np.sqrt((fluctuations ** 2).sum(axis=2).mean(axis=0))

    per_atom = pd.DataFrame({
        "index": target.indices,
        "resid": target.resids,
        "resname": target.resnames,
        "name": target.names,
        "rmsf": rmsf,
    })

    try:
        masses = np.asarray(target.masses, dtype=float)
        if not np.all(np.isfinite(masses)) or masses.sum() <= 0:
            masses = np.ones(target.n_atoms)
    except (mda.exceptions.NoDataError, AttributeError):
        masses = np.ones(target.n_atoms)
    per_atom["_mass"] = masses

    grouped = per_atom.groupby(["resid", "resname"], sort=True)
    per_residue = grouped.apply(
        lambda g: pd.Series({
            "n_atoms": len(g),
            "rmsf": float(np.sqrt(np.average(g["rmsf"] ** 2, weights=g["_mass"]))),
        }),
        include_groups=False,
    ).reset_index()
    per_residue["n_atoms"] = per_residue["n_atoms"].astype(int)
    return per_atom.drop(columns="_mass"), per_residue


def _iterative_fit(
    coords: np.ndarray, fit_idx: np.ndarray, weights: np.ndarray | None,
    n_passes: int = 2,
) -> np.ndarray:
    """Superpose every frame onto the average structure (iteratively).

    Uses MDAnalysis' QCP rotation matrix; the first pass fits on frame 0, the
    following ones on the running average, which is the standard way of
    removing global motion before computing fluctuations.
    """
    coords = coords.copy()
    reference = coords[0, fit_idx]
    for _pass in range(n_passes):
        ref_com = _centre(reference, weights)
        ref_centred = reference - ref_com
        for i in range(coords.shape[0]):
            mobile = coords[i, fit_idx]
            com = _centre(mobile, weights)
            rotation, _rmsd = align.rotation_matrix(mobile - com, ref_centred,
                                                    weights=weights)
            coords[i] = (coords[i] - com) @ rotation.T + ref_com
        reference = coords[:, fit_idx, :].mean(axis=0)
    return coords


def _centre(positions: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    if weights is None:
        return positions.mean(axis=0)
    return np.average(positions, axis=0, weights=weights)


# --------------------------------------------------------------------------- #
# module entry point
# --------------------------------------------------------------------------- #
def run(
    system: MDSystem,
    paths: OutputPaths,
    config=None,
    verbose: bool = True,
) -> AnalysisResult:
    """Run RMSD and/or RMSF according to the configuration."""
    rmsd_cfg: RMSDConfig = system.config.rmsd
    rmsf_cfg: RMSFConfig = system.config.rmsf
    formats = system.config.output.formats
    dpi = system.config.output.dpi

    tables: dict[str, pd.DataFrame] = {}
    rows: list[dict] = []
    plots: list = []
    notes: list[str] = []
    series = None

    if rmsd_cfg.enabled:
        if verbose:
            print(f"[rmsd] {len(rmsd_cfg.groups)} group(s) over {system.n_frames} frames")
        df = compute_rmsd(system, rmsd_cfg)
        tables["rmsd"] = df
        series = df
        write_csv(df, paths.data, "rmsd")

        time = df["time"].to_numpy()
        for group in rmsd_cfg.groups:
            values = df[group.name].to_numpy()
            fig = plot_timeseries(time, {group.name: values}, xlabel=system.time_label,
                                  ylabel=f"RMSD ({_UNIT})", title=group.name)
            plots += save_figure(fig, paths.plots,
                                 basename("rmsd", group.name, "_timeseries"),
                                 formats, dpi)[:1]
            rows.append(summary_row(
                "rmsd", group.name, values, _UNIT,
                definition=f"{group.selection}"
                           + (f" (fit: {group.superposition})" if group.superposition else ""),
            ))
        if len(rmsd_cfg.groups) > 1:
            fig = plot_timeseries(
                time, {g.name: df[g.name].to_numpy() for g in rmsd_cfg.groups},
                xlabel=system.time_label, ylabel=f"RMSD ({_UNIT})",
                title="RMSD", figsize=(7.2, 4.0),
            )
            plots += save_figure(fig, paths.plots, "rmsd_all_timeseries", formats, dpi)[:1]
        reference = (str(rmsd_cfg.reference) if rmsd_cfg.reference
                     else f"frame {rmsd_cfg.ref_frame} of the trajectory")
        notes.append(f"RMSD reference: {reference}.")

    if rmsf_cfg.enabled:
        if verbose:
            print(f"[rmsf] selection \"{rmsf_cfg.selection}\"")
        per_atom, per_residue = compute_rmsf(system, rmsf_cfg)
        tables["rmsf_per_atom"] = per_atom
        tables["rmsf_per_residue"] = per_residue
        write_csv(per_atom, paths.data, "rmsf_per_atom")
        write_csv(per_residue, paths.data, "rmsf_per_residue")
        plots += _plot_rmsf(per_residue, rmsf_cfg, paths, formats, dpi)
        rows.append(summary_row("rmsf", "rmsf_per_residue",
                                per_residue["rmsf"].to_numpy(), _UNIT,
                                definition=rmsf_cfg.selection))
        peak = per_residue.loc[per_residue["rmsf"].idxmax()]
        notes.append(
            f"Most flexible residue: {peak['resname']}{int(peak['resid'])} "
            f"({peak['rmsf']:.2f} {_UNIT})."
        )

    return AnalysisResult(
        name="rmsd_rmsf",
        title="RMSD and RMSF",
        tables=tables,
        series=series,
        summary=pd.DataFrame(rows),
        plots=list(plots),
        notes=notes,
    )


def _plot_rmsf(per_residue: pd.DataFrame, config: RMSFConfig, paths: OutputPaths,
               formats, dpi) -> list:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    resids = per_residue["resid"].to_numpy()
    values = per_residue["rmsf"].to_numpy()
    ax.plot(resids, values, color=color_cycle(1)[0])
    ax.fill_between(resids, 0, values, color=color_cycle(1)[0], alpha=0.20)

    highlighted = {int(h) for h in config.highlight if str(h).lstrip("-").isdigit()}
    if highlighted:
        mask = np.isin(resids, list(highlighted))
        if mask.any():
            ax.scatter(resids[mask], values[mask], color="#D55E00", zorder=5, s=28,
                       label="highlighted residues")
            for resid, value in zip(resids[mask], values[mask]):
                ax.annotate(str(int(resid)), (resid, value), textcoords="offset points",
                            xytext=(0, 6), ha="center", fontsize=8, color="#D55E00")
            ax.legend(loc="best")
    ax.set_xlabel("Residue")
    ax.set_ylabel(f"RMSF ({_UNIT})")
    ax.set_xlim(resids.min(), resids.max())
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return save_figure(fig, paths.plots, "rmsf_per_residue", formats, dpi)[:1]
