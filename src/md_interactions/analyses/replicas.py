"""Agreement between replicas.

Pooling replicas into a single histogram is the usual way to hide the most
interesting thing in a set of simulations: that one run visits a state the
others never see.  This module compares, observable by observable, the
distribution sampled by each replica and flags the ones that disagree.

The statistic used is the **overlap coefficient** — the area shared by two
normalised histograms, 1.0 for identical distributions and 0.0 for disjoint
ones.  It is reported instead of a p-value on purpose: consecutive MD frames
are strongly correlated, so tests such as Kolmogorov–Smirnov assume an
effective sample size that a trajectory does not have and end up calling
everything "significantly different".
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from ..io_utils import OutputPaths, write_csv
from ..plotting import color_cycle, save_figure
from ..results import AnalysisResult

__all__ = ["run", "compare_replicas", "overlap_coefficient"]

#: Below this overlap two replicas are considered to sample different states.
DEFAULT_WARN_OVERLAP = 0.70


def overlap_coefficient(a: np.ndarray, b: np.ndarray, bins: int = 60) -> float:
    """Shared area of the two normalised distributions (1 = identical)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    if hi <= lo:
        return 1.0
    edges = np.linspace(lo, hi, bins + 1)
    width = edges[1] - edges[0]
    da, _ = np.histogram(a, bins=edges, density=True)
    db, _ = np.histogram(b, bins=edges, density=True)
    return float(np.minimum(da, db).sum() * width)


def compare_replicas(
    data: pd.DataFrame,
    columns: list[str],
    warn_overlap: float = DEFAULT_WARN_OVERLAP,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-replica statistics and pairwise overlaps.

    Returns ``(per_replica, pairwise)``; ``per_replica`` has one row per
    (observable, replica) with mean/sd/min/max, and ``pairwise`` one row per
    (observable, replica pair) with the overlap coefficient and a
    ``diverges`` flag.
    """
    names = list(dict.fromkeys(data["replica"].tolist()))
    per_replica_rows: list[dict] = []
    pairwise_rows: list[dict] = []

    for column in columns:
        values = {name: data.loc[data["replica"] == name, column].to_numpy(dtype=float)
                  for name in names}
        for name, series in values.items():
            clean = series[~np.isnan(series)]
            per_replica_rows.append({
                "observable": column,
                "replica": name,
                "n": int(clean.size),
                "mean": float(clean.mean()) if clean.size else np.nan,
                "std": float(clean.std(ddof=1)) if clean.size > 1 else 0.0,
                "min": float(clean.min()) if clean.size else np.nan,
                "max": float(clean.max()) if clean.size else np.nan,
            })
        for first, second in itertools.combinations(names, 2):
            overlap = overlap_coefficient(values[first], values[second])
            a, b = values[first], values[second]
            identical = bool(a.size == b.size and a.size > 0 and np.array_equal(a, b))
            pairwise_rows.append({
                "observable": column,
                "replica_a": first,
                "replica_b": second,
                "overlap": overlap,
                "delta_mean": float(np.nanmean(a) - np.nanmean(b)),
                "diverges": bool(overlap < warn_overlap),
                "identical": identical,
            })

    return pd.DataFrame(per_replica_rows), pd.DataFrame(pairwise_rows)


def run(
    data: pd.DataFrame,
    columns: list[str],
    paths: OutputPaths,
    formats=("png",),
    dpi: int = 300,
    unit: str = "Å",
    warn_overlap: float = DEFAULT_WARN_OVERLAP,
    verbose: bool = True,
) -> AnalysisResult:
    """Compare replicas and write the convergence tables and figure."""
    per_replica, pairwise = compare_replicas(data, columns, warn_overlap)
    if verbose:
        print(f"[replicas] comparing {data['replica'].nunique()} replicas "
              f"over {len(columns)} observable(s)")

    write_csv(per_replica, paths.summary, "replica_statistics")
    write_csv(pairwise, paths.summary, "replica_overlap")

    plots = _plot_overlap(pairwise, columns, paths, formats, dpi)

    diverging = pairwise[pairwise["diverges"]]
    notes = [
        "Overlap coefficient: shared area of the two normalised distributions "
        "(1 = identical). No p-values are reported because consecutive frames "
        "are correlated.",
    ]

    # Independent runs never produce bit-identical values: this means the same
    # trajectory was read twice, not that the sampling converged.
    identical = pairwise[pairwise["identical"]]
    if not identical.empty:
        pairs = sorted({(row["replica_a"], row["replica_b"])
                        for _i, row in identical.iterrows()})
        listed = ", ".join(f"{a} = {b}" for a, b in pairs)
        message = (
            f"CRITICAL: {listed} contain numerically identical values for "
            f"{len(identical)} of the {len(pairwise)} observable/pair "
            "combinations. Independent simulations never coincide bit for bit: "
            "the same trajectory file is almost certainly being read more than "
            "once (duplicated path, copied file, or runs restarted from the "
            "same seed). The replica comparison below is meaningless until this "
            "is fixed."
        )
        notes.append(message)
        if verbose:
            print(f"[replicas] {message}")
    if diverging.empty:
        notes.append(
            f"All replica pairs overlap by at least {warn_overlap:.0%}: pooling "
            "them into a single distribution is defensible."
        )
    else:
        worst = diverging.sort_values("overlap").head(5)
        listed = "; ".join(
            f"{row['observable']} ({row['replica_a']} vs {row['replica_b']}: "
            f"{row['overlap']:.2f})" for _i, row in worst.iterrows()
        )
        notes.append(
            f"WARNING: {len(diverging)} replica pair(s) sample different "
            f"distributions (overlap < {warn_overlap:.0%}): {listed}. "
            "Do not pool them without saying so — the extra population may come "
            "from a single replica."
        )
        if verbose:
            print(f"[replicas] WARNING: {len(diverging)} pair(s) below "
                  f"{warn_overlap:.0%} overlap")

    return AnalysisResult(
        name="replicas",
        title="Agreement between replicas",
        tables={"replica_statistics": per_replica, "replica_overlap": pairwise},
        summary=None,
        plots=list(plots),
        notes=notes,
    )


def _plot_overlap(pairwise: pd.DataFrame, columns: list[str], paths: OutputPaths,
                  formats, dpi) -> list:
    """Heat map of the pairwise overlap, one row per observable."""
    import matplotlib.pyplot as plt

    if pairwise.empty:
        return []
    pairs = list(dict.fromkeys(
        zip(pairwise["replica_a"], pairwise["replica_b"])
    ))
    matrix = np.full((len(columns), len(pairs)), np.nan)
    for i, column in enumerate(columns):
        for j, (a, b) in enumerate(pairs):
            row = pairwise[(pairwise["observable"] == column)
                           & (pairwise["replica_a"] == a)
                           & (pairwise["replica_b"] == b)]
            if not row.empty:
                matrix[i, j] = float(row["overlap"].iloc[0])

    height = max(2.4, 0.32 * len(columns) + 1.2)
    fig, ax = plt.subplots(figsize=(max(3.2, 1.5 * len(pairs) + 2.0), height))
    mesh = ax.imshow(matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([f"{a}\nvs {b}" for a, b in pairs], fontsize=7)
    ax.set_yticks(range(len(columns)))
    ax.set_yticklabels(columns, fontsize=7)
    for i in range(len(columns)):
        for j in range(len(pairs)):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=6.5,
                        color="black" if matrix[i, j] > 0.35 else "white")
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    colorbar.set_label("Distribution overlap")
    ax.set_title("Replica agreement", fontsize=10)
    ax.grid(False)
    fig.tight_layout()
    return save_figure(fig, paths.plots, "replica_overlap", formats, dpi)[:1]
