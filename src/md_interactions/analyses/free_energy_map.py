"""2D probability-density / apparent free-energy maps.

Projects the trajectory onto a pair of observables computed by the other
modules (distances, angles, RMSD, Rg...) and turns the 2D histogram (or KDE)
into an apparent free energy ``ΔG = -kT ln P``, referenced to the most
populated bin.

.. warning::
   For unbiased classical MD this is a *conformational population* map, not a
   converged free-energy surface: barriers are only meaningful if the relevant
   states are reversibly sampled.  For biased simulations (metadynamics, umbrella
   sampling) reweight the data before feeding it here, or use the native
   analysis of the biasing code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FreeEnergyConfig, FreeEnergyMap
from ..exceptions import AnalysisError
from ..io_utils import OutputPaths, write_csv
from ..plotting import save_figure
from ..results import AnalysisResult
from ..system import MDSystem
from ._common import basename, safe_name

__all__ = ["run", "compute_map", "effective_bins"]

#: Boltzmann constant in the supported energy units.
_KB = {"kcal/mol": 0.0019872041, "kJ/mol": 0.008314462618, "kT": None}


def effective_bins(requested: int, n_samples: int) -> int:
    """Cap the number of bins so that the map is not mostly empty.

    With too many bins for the number of frames every occupied bin holds one
    count, the map degenerates into confetti and ``-kT ln P`` becomes
    meaningless.  The cap follows the usual ``sqrt(n/2)`` rule of thumb per
    axis, with a floor of 10 bins.
    """
    cap = max(10, int(np.sqrt(max(n_samples, 1) / 2.0)))
    return int(max(5, min(requested, cap)))


def compute_map(
    x: np.ndarray,
    y: np.ndarray,
    spec: FreeEnergyMap,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the 2D grid for one map.

    Returns
    -------
    (x_edges_or_grid, y_edges_or_grid, density, energy)
        ``density`` is the normalised probability density and ``energy`` the
        apparent free energy (NaN where the density is zero).  Both are shaped
        ``(n_y, n_x)`` so they can be passed straight to ``pcolormesh``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if x.size < 10:
        raise AnalysisError(
            f"Not enough valid frames ({x.size}) to build the map '{spec.name}'."
        )

    bins = effective_bins(spec.bins, x.size)

    if spec.method == "kde":
        from scipy.stats import gaussian_kde

        x_grid = np.linspace(x.min(), x.max(), bins)
        y_grid = np.linspace(y.min(), y.max(), bins)
        mesh_x, mesh_y = np.meshgrid(x_grid, y_grid)
        try:
            kernel = gaussian_kde(np.vstack([x, y]))
        except Exception as exc:  # singular covariance
            raise AnalysisError(
                f"KDE failed for map '{spec.name}' ({exc}); try method: histogram."
            ) from exc
        density = kernel(np.vstack([mesh_x.ravel(), mesh_y.ravel()])).reshape(mesh_x.shape)
        axes = (x_grid, y_grid)
    else:
        counts, x_edges, y_edges = np.histogram2d(x, y, bins=bins, density=True)
        density = counts.T  # (n_y, n_x)
        if spec.smooth > 0:
            from scipy.ndimage import gaussian_filter

            density = gaussian_filter(density, sigma=spec.smooth, mode="nearest")
        axes = (x_edges, y_edges)

    energy = _to_free_energy(density, spec)
    return axes[0], axes[1], density, energy


def _to_free_energy(density: np.ndarray, spec: FreeEnergyMap) -> np.ndarray:
    """``-kT ln P`` shifted so that the global minimum is 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(density > 0, density / density.max(), np.nan)
        kb = _KB[spec.energy_unit]
        kt = 1.0 if kb is None else kb * spec.temperature
        energy = -kt * np.log(relative)
    return energy


def run(
    system: MDSystem | None,
    paths: OutputPaths,
    config: FreeEnergyConfig | None = None,
    observables: pd.DataFrame | None = None,
    units: dict[str, str] | None = None,
    output_config=None,
    verbose: bool = True,
) -> AnalysisResult:
    """Build every configured 2D map from the collected observables.

    Works both on a trajectory run (``system`` given) and on the ``data:``
    mode, where the observables come from a pre-computed table and
    ``output_config`` carries the figure formats.
    """
    config = config or (system.config.free_energy if system else None)
    if config is None:  # pragma: no cover - guarded by the caller
        raise AnalysisError("No free-energy map configuration was provided.")
    output_config = output_config or (system.config.output if system else None)
    formats = output_config.formats if output_config else ("png",)
    dpi = output_config.dpi if output_config else 300
    units = units or {}
    if observables is None or observables.empty:
        raise AnalysisError(
            "Free-energy maps need observables computed by other analyses "
            "(distances, angles, RMSD, Rg). None were available."
        )
    available = [c for c in observables.columns if c not in ("frame", "time")]

    tables: dict[str, pd.DataFrame] = {}
    plots: list = []
    notes: list[str] = []

    for spec in config.maps:
        for axis_name in (spec.x, spec.y):
            if axis_name not in observables.columns:
                raise AnalysisError(
                    f"Map '{spec.name}' references the unknown observable "
                    f"'{axis_name}'.\n  Available observables: {', '.join(available)}"
                )
        if verbose:
            print(f"[free_energy_maps] {spec.name}: {spec.x} vs {spec.y} ({spec.method})")

        x = observables[spec.x].to_numpy()
        y = observables[spec.y].to_numpy()
        x_axis, y_axis, density, energy = compute_map(x, y, spec)

        n_valid = int(np.sum(~(np.isnan(x) | np.isnan(y))))
        used_bins = effective_bins(spec.bins, n_valid)
        if used_bins < spec.bins:
            message = (
                f"'{spec.name}': {used_bins} bins used instead of {spec.bins} — "
                f"{n_valid} frames are not enough to fill a finer grid."
            )
            notes.append(message)
            if verbose:
                print(f"[free_energy_maps] {message}")

        grid = _grid_table(x_axis, y_axis, density, energy, spec)
        name = basename("fes", spec.name)
        tables[name] = grid
        write_csv(grid, paths.data, name)

        xlabel = spec.xlabel or _axis_label(spec.x, units)
        ylabel = spec.ylabel or _axis_label(spec.y, units)
        fig = _plot_map(x_axis, y_axis, density, energy, spec, xlabel, ylabel)
        plots += save_figure(fig, paths.plots, name, formats, dpi)[:1]

        if spec.free_energy:
            notes.append(
                f"'{spec.name}': ΔG = -kT ln P at T = {spec.temperature:g} K, "
                f"relative to the most populated bin ({spec.energy_unit})."
            )

    return AnalysisResult(
        name="free_energy_maps",
        title="Free-energy / density maps",
        tables=tables,
        summary=None,
        plots=list(plots),
        notes=notes + [
            "Unbiased MD: these maps reflect the sampled populations, not a "
            "converged free-energy surface.",
        ],
    )


def _axis_label(name: str, units: dict[str, str]) -> str:
    unit = units.get(name)
    return f"{name} ({unit})" if unit else name


def _grid_table(x_axis, y_axis, density, energy, spec: FreeEnergyMap) -> pd.DataFrame:
    """Long-format table of the grid, so the map can be re-plotted elsewhere."""
    if spec.method == "kde":
        x_centres, y_centres = x_axis, y_axis
    else:
        x_centres = 0.5 * (x_axis[:-1] + x_axis[1:])
        y_centres = 0.5 * (y_axis[:-1] + y_axis[1:])
    mesh_x, mesh_y = np.meshgrid(x_centres, y_centres)
    return pd.DataFrame({
        spec.x: mesh_x.ravel(),
        spec.y: mesh_y.ravel(),
        "density": density.ravel(),
        "free_energy": energy.ravel(),
    })


def _plot_map(x_axis, y_axis, density, energy, spec: FreeEnergyMap,
              xlabel: str, ylabel: str):
    import matplotlib.pyplot as plt

    values = energy if spec.free_energy else density
    if spec.free_energy:
        colorbar_label = (f"ΔG ({spec.energy_unit})" if spec.energy_unit != "kT"
                          else "ΔG (kT)")
        cmap = "viridis"
        vmax = spec.max_energy
    else:
        colorbar_label = "Probability density"
        cmap = "magma"
        vmax = None

    if spec.method == "kde":
        x_edges = _edges_from_centres(x_axis)
        y_edges = _edges_from_centres(y_axis)
    else:
        x_edges, y_edges = x_axis, y_axis

    colormap = plt.get_cmap(cmap).copy()
    colormap.set_bad("white")          # unsampled bins stay blank
    masked = np.ma.masked_invalid(values)

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    mesh = ax.pcolormesh(x_edges, y_edges, masked, cmap=colormap, vmin=0, vmax=vmax,
                         shading="auto", rasterized=True)
    if spec.free_energy and np.isfinite(values).sum() > 20:
        top = vmax if vmax is not None else float(np.nanmax(values))
        levels = np.arange(1.0, top + 0.5, 1.0)
        if levels.size:
            x_centres = (x_axis if spec.method == "kde"
                         else 0.5 * (np.asarray(x_edges)[:-1] + np.asarray(x_edges)[1:]))
            y_centres = (y_axis if spec.method == "kde"
                         else 0.5 * (np.asarray(y_edges)[:-1] + np.asarray(y_edges)[1:]))
            # NaN (unsampled) bins are ignored by contour, no spurious edges.
            ax.contour(x_centres, y_centres, values, levels=levels,
                       colors="white", linewidths=0.5, alpha=0.55)

    colorbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    colorbar.set_label(colorbar_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(spec.name)
    ax.grid(False)
    fig.tight_layout()
    return fig


def _edges_from_centres(centres: np.ndarray) -> np.ndarray:
    centres = np.asarray(centres, dtype=float)
    step = centres[1] - centres[0]
    return np.concatenate([centres - 0.5 * step, [centres[-1] + 0.5 * step]])
