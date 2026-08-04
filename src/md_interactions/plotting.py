"""Shared plotting style and figure helpers.

Every module builds its figures through these helpers so that fonts, colours,
sizes and file naming stay consistent across the whole report (and across
papers).  Figures are always saved in every format requested in
``output.formats`` (PNG for drafts, PDF/SVG for publication).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402

__all__ = [
    "PALETTE",
    "apply_style",
    "color_cycle",
    "save_figure",
    "plot_timeseries",
    "plot_distribution",
    "plot_facet_distributions",
    "normalization_label",
    "running_mean",
]

#: Usable height of a journal page (inches); figures are laid out to fit it.
PAGE_HEIGHT_IN = 9.0
#: Width of a double-column figure (inches).
PAGE_WIDTH_IN = 7.3

#: Line styles used to tell replicas apart inside a panel.
REPLICA_LINESTYLES: list = [
    (0, (4.5, 1.5)),            # dashed
    (0, (1, 1.4)),              # dotted
    (0, (6, 1.4, 1.2, 1.4)),    # dash-dot
    (0, (3, 1.2, 1, 1.2, 1, 1.2)),  # dash-dot-dot
]
#: Neutral shades paired with the line styles (dark on top of a coloured bar).
REPLICA_SHADES: list[str] = ["#111111", "#5A5A5A", "#9A9A9A", "#3D3D3D"]

#: Colour-blind safe qualitative palette (Okabe & Ito).
PALETTE: list[str] = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#8C564B",  # brown
    "#332288",  # indigo
]

_RC_PARAMS: dict[str, Any] = {
    "figure.figsize": (6.0, 4.0),
    "figure.dpi": 110,
    "savefig.bbox": "tight",
    "savefig.transparent": False,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "pdf.fonttype": 42,   # editable text in Illustrator/Inkscape
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def apply_style(style: str = "md_interactions") -> None:
    """Apply the package-wide matplotlib style.

    ``style`` may also be the name of any matplotlib/seaborn style, which is
    applied *before* the package overrides (so the palette and font sizes stay
    consistent).
    """
    if style and style != "md_interactions":
        try:
            plt.style.use(style)
        except (OSError, ValueError):
            pass
    matplotlib.rcParams.update(_RC_PARAMS)
    matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=PALETTE)


def color_cycle(n: int) -> list[str]:
    """Return ``n`` colours from the package palette (cycling if needed)."""
    return [PALETTE[i % len(PALETTE)] for i in range(n)]


def save_figure(
    fig: plt.Figure,
    directory: Path,
    basename: str,
    formats: Sequence[str] = ("png",),
    dpi: int = 300,
    close: bool = True,
) -> list[Path]:
    """Save ``fig`` as ``directory/basename.<fmt>`` for every requested format.

    Returns the list of written paths; the first one is always the raster
    (PNG) version when PNG was requested, which is what the report embeds.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ordered = sorted(formats, key=lambda f: 0 if f == "png" else 1)
    paths: list[Path] = []
    for fmt in ordered:
        path = directory / f"{basename}.{fmt}"
        fig.savefig(path, dpi=dpi, format=fmt)
        paths.append(path)
    if close:
        plt.close(fig)
    return paths


def running_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Centred running average; returns NaN where the window does not fit."""
    values = np.asarray(values, dtype=float)
    if window <= 1 or window > values.size:
        return values.copy()
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="same")
    half = window // 2
    smoothed[:half] = np.nan
    if half:
        smoothed[-half:] = np.nan
    return smoothed


def plot_timeseries(
    time: np.ndarray,
    series: Mapping[str, np.ndarray],
    xlabel: str,
    ylabel: str,
    title: str | None = None,
    smooth_window: int = 0,
    hlines: Iterable[tuple[float, str]] = (),
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Standard time-series panel (one line per entry of ``series``)."""
    fig, ax = plt.subplots(figsize=figsize or (6.4, 3.6))
    colors = color_cycle(len(series))
    raw_alpha = 0.35 if smooth_window > 1 else 1.0
    for (label, values), color in zip(series.items(), colors):
        values = np.asarray(values, dtype=float)
        ax.plot(time, values, color=color, alpha=raw_alpha,
                label=None if smooth_window > 1 else label)
        if smooth_window > 1:
            ax.plot(time, running_mean(values, smooth_window), color=color,
                    linewidth=1.8, label=f"{label} (avg. {smooth_window} frames)")
    for value, label in hlines:
        ax.axhline(value, color="0.35", linestyle="--", linewidth=1.0)
        ax.text(0.995, value, f" {label}", color="0.35", fontsize=8,
                transform=ax.get_yaxis_transform(), ha="right", va="bottom")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if len(series) > 1 or smooth_window > 1:
        if len(series) > 5:
            # too many curves for an inset legend: move it out of the axes
            ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                      ncol=1 if len(series) <= 12 else 2)
        else:
            ax.legend(loc="best")
    ax.set_xlim(float(np.min(time)), float(np.max(time)))
    fig.tight_layout()
    return fig


def normalization_label(normalization: str, unit: str = "Å") -> str:
    """Y-axis label matching the requested histogram normalisation."""
    if normalization == "density":
        return f"Probability density ({unit}$^{{-1}}$)" if unit else "Probability density"
    if normalization == "percent":
        return "Percentage of frames (%)"
    return "Counts"


def _histogram_weights(values: np.ndarray, normalization: str):
    """``(density, weights)`` pair for :meth:`matplotlib.axes.Axes.hist`."""
    if normalization == "density":
        return True, None
    if normalization == "percent":
        return False, np.full(values.shape, 100.0 / values.size)
    return False, None


def plot_distribution(
    series: Mapping[str, np.ndarray],
    xlabel: str,
    bins: int = 60,
    kde: bool = True,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    periodic: tuple[float, float] | None = None,
    normalization: str = "density",
    unit: str = "Å",
) -> plt.Figure:
    """Normalised histogram (+ optional Gaussian KDE) of one or more series.

    ``normalization`` is ``density`` (probability density, integrates to 1 and
    is therefore comparable between panels of different width), ``percent``
    (percentage of frames per bin) or ``counts``.
    """
    fig, ax = plt.subplots(figsize=figsize or (5.2, 3.6))
    colors = color_cycle(len(series))
    hist_range = periodic
    for (label, values), color in zip(series.items(), colors):
        values = np.asarray(values, dtype=float)
        values = values[~np.isnan(values)]
        if values.size == 0:
            continue
        density, weights = _histogram_weights(values, normalization)
        ax.hist(values, bins=bins, range=hist_range, density=density, weights=weights,
                color=color, alpha=0.45, label=label, edgecolor="none")
        if kde and normalization == "density" and values.size > 5 and np.ptp(values) > 0:
            grid, kde_values = _kde(values, hist_range)
            if grid is not None:
                ax.plot(grid, kde_values, color=color, linewidth=1.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(normalization_label(normalization, unit))
    if periodic:
        ax.set_xlim(*periodic)
    if title:
        ax.set_title(title)
    if len(series) > 1:
        ax.legend(loc="best")
    fig.tight_layout()
    return fig


def facet_layout(n_panels: int, panel_height: float = 1.85,
                 max_height: float = PAGE_HEIGHT_IN) -> tuple[int, int]:
    """Choose ``(nrows, ncols)`` so that the grid fits on a printed page.

    Starts at two columns (the widest, most readable panels) and adds columns
    until the figure is no taller than ``max_height``.
    """
    for ncols in (2, 3, 4, 5):
        nrows = int(np.ceil(n_panels / ncols))
        if nrows * panel_height <= max_height or ncols == 5:
            return nrows, ncols
    return int(np.ceil(n_panels / 2)), 2  # pragma: no cover


def plot_facet_distributions(
    series: Mapping[str, np.ndarray],
    xlabel: str = "Distance (Å)",
    unit: str = "Å",
    bins: int = 30,
    kde: bool = True,
    normalization: str = "density",
    annotate: bool = True,
    groups: Mapping[str, Mapping[str, np.ndarray]] | None = None,
    periodic: tuple[float, float] | None = None,
) -> plt.Figure:
    """One histogram panel per observable, laid out to fit a journal page.

    Parameters
    ----------
    series
        ``{observable: values}`` — the pooled data plotted as bars.
    groups
        Optional ``{group: {observable: values}}`` (typically one entry per
        replica).  Each group is drawn as a dashed KDE line on top of the
        pooled histogram, which is the quickest way to see whether the
        replicas agree or whether a population comes from a single one.
    normalization
        ``density`` (default), ``percent`` or ``counts``.  Density is the only
        one that stays comparable between panels when each panel has its own
        range, because it integrates to 1 regardless of bin width.
    annotate
        Print ``mean ± sd`` and ``n`` inside every panel.
    """
    names = list(series)
    n = len(names)
    if n == 0:
        raise ValueError("plot_facet_distributions needs at least one observable.")

    nrows, ncols = facet_layout(n)
    panel_height = 1.85 if ncols > 2 else 2.1
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(PAGE_WIDTH_IN, panel_height * nrows))
    axes = np.atleast_1d(axes).flatten()
    colors = color_cycle(n)
    group_names = list(groups) if groups else []
    # long names do not fit inside a panel; keep the placement uniform across
    # the whole figure instead of mixing in-panel and above-panel labels
    use_titles = max(len(str(name)) for name in names) > 12

    for index, (name, color) in enumerate(zip(names, colors)):
        ax = axes[index]
        values = np.asarray(series[name], dtype=float)
        values = values[~np.isnan(values)]
        if values.size == 0:
            ax.set_axis_off()
            continue

        density, weights = _histogram_weights(values, normalization)
        ax.hist(values, bins=bins, range=periodic, density=density, weights=weights,
                color=mpl_to_rgba(color, 0.75), edgecolor=_darken(color),
                linewidth=0.6, zorder=2)
        if kde and normalization == "density" and values.size > 5 and np.ptp(values) > 0:
            grid, kde_values = _kde(values, periodic)
            if grid is not None:
                ax.plot(grid, kde_values, color=_darken(color), linewidth=1.2, zorder=3)

        for position, group in enumerate(group_names):
            group_values = groups[group].get(name)
            if group_values is None:
                continue
            group_values = np.asarray(group_values, dtype=float)
            group_values = group_values[~np.isnan(group_values)]
            if group_values.size < 6 or np.ptp(group_values) == 0:
                continue
            grid, kde_values = _kde(group_values, periodic or
                                    (float(values.min()), float(values.max())))
            if grid is not None:
                # each replica gets its own dash pattern *and* its own shade,
                # otherwise the curves are impossible to tell apart
                ax.plot(grid, kde_values, linewidth=1.1,
                        linestyle=REPLICA_LINESTYLES[position % len(REPLICA_LINESTYLES)],
                        color=REPLICA_SHADES[position % len(REPLICA_SHADES)],
                        alpha=0.9, zorder=4)

        if use_titles:
            ax.set_title(name, fontsize=8, fontweight="bold", color=color,
                         loc="right", pad=3)
            label_top = 0.93
        else:
            ax.text(0.95, 0.93, name, transform=ax.transAxes, ha="right", va="top",
                    fontsize=8.5, fontweight="bold", color=color)
            label_top = 0.79
        if annotate:
            ax.text(0.95, label_top,
                    f"{values.mean():.2f} ± {values.std(ddof=1 if values.size > 1 else 0):.2f}"
                    f"{' ' + unit if unit else ''}\nn = {values.size}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=6.5, color="0.35")

        if periodic:
            ax.set_xlim(*periodic)
        else:
            ax.set_xlim(float(values.min()), float(values.max()))
        ax.locator_params(axis="x", nbins=4)
        ax.grid(False)

    for ax in axes[n:]:
        fig.delaxes(ax)

    # x label under the last panel of every column, y label once on the left
    for col in range(ncols):
        column_indices = [i for i in range(col, n, ncols)]
        if column_indices:
            axes[max(column_indices)].set_xlabel(xlabel)
    left_column = [i for i in range(0, n, ncols)]
    axes[left_column[len(left_column) // 2]].set_ylabel(
        normalization_label(normalization, unit)
    )

    fig.tight_layout(h_pad=1.0, w_pad=1.0)
    if group_names:
        # one figure-level legend for the per-replica curves, below the grid
        handles = [
            matplotlib.lines.Line2D(
                [], [], linewidth=1.1,
                linestyle=REPLICA_LINESTYLES[i % len(REPLICA_LINESTYLES)],
                color=REPLICA_SHADES[i % len(REPLICA_SHADES)], label=name,
            )
            for i, name in enumerate(group_names)
        ]
        handles.append(matplotlib.lines.Line2D([], [], linewidth=1.4, color="0.45",
                                               label="pooled"))
        fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6),
                   fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.015),
                   handlelength=3.0, columnspacing=1.6)
    return fig


def mpl_to_rgba(color: str, alpha: float):
    """``color`` with the given alpha, as an RGBA tuple."""
    return matplotlib.colors.to_rgba(color, alpha)


def _darken(color: str, factor: float = 0.6):
    """Darker version of ``color``, used for bar outlines and KDE lines."""
    r, g, b = matplotlib.colors.to_rgb(color)
    return (r * factor, g * factor, b * factor)


def _kde(values: np.ndarray, hist_range: tuple[float, float] | None):
    """Gaussian KDE on a regular grid; ``(None, None)`` if scipy fails."""
    try:
        from scipy.stats import gaussian_kde
    except ImportError:  # pragma: no cover - scipy is a hard dependency
        return None, None
    try:
        kernel = gaussian_kde(values)
    except Exception:  # singular covariance (constant data), etc.
        return None, None
    if hist_range is not None:
        lo, hi = hist_range
    else:
        span = float(np.ptp(values))
        lo = float(values.min()) - 0.1 * span
        hi = float(values.max()) + 0.1 * span
    grid = np.linspace(lo, hi, 400)
    return grid, kernel(grid)
