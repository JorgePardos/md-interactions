"""Radial distribution functions between two selections.

Wraps :class:`MDAnalysis.analysis.rdf.InterRDF` and adds the running
coordination number ``n(r)``, which is what you usually quote when discussing
the solvation shell of a catalytic atom.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from MDAnalysis.analysis.rdf import InterRDF

from ..config import RDFConfig, RDFPair
from ..io_utils import OutputPaths, write_csv
from ..plotting import color_cycle, save_figure
from ..results import AnalysisResult
from ..system import MDSystem
from ._common import basename, safe_name

__all__ = ["run", "compute_rdf"]


def compute_rdf(system: MDSystem, spec: RDFPair) -> pd.DataFrame:
    """RDF and running coordination number for one pair of selections.

    Returns a DataFrame with columns ``r`` (Å), ``g_r`` and ``n_r``.
    """
    g1 = system.select(spec.g1, name=f"rdf:{spec.name}.g1")
    g2 = system.select(spec.g2, name=f"rdf:{spec.name}.g2")

    kwargs = {}
    if spec.exclude_same_residue:
        kwargs["exclude_same"] = "residue"
    analysis = InterRDF(g1, g2, nbins=spec.nbins, range=spec.range, **kwargs)
    analysis.run(**system.run_kwargs)

    bins = np.asarray(analysis.results.bins, dtype=float)
    rdf_values = np.asarray(analysis.results.rdf, dtype=float)
    counts = np.asarray(analysis.results.count, dtype=float)
    # Average number of g2 atoms within r of a g1 atom.
    coordination = np.cumsum(counts) / (system.n_frames * g1.n_atoms)

    return pd.DataFrame({"r": bins, "g_r": rdf_values, "n_r": coordination})


def run(
    system: MDSystem,
    paths: OutputPaths,
    config: RDFConfig | None = None,
    verbose: bool = True,
) -> AnalysisResult:
    """Run every configured RDF and write CSVs and figures."""
    config = config or system.config.rdf
    formats = system.config.output.formats
    dpi = system.config.output.dpi

    tables: dict[str, pd.DataFrame] = {}
    plots: list = []
    rows: list[dict] = []

    for spec in config.pairs:
        if verbose:
            print(f"[rdf] {spec.name}: {spec.g1} -> {spec.g2}")
        df = compute_rdf(system, spec)
        name = safe_name(spec.name)
        tables[name] = df
        write_csv(df, paths.data, basename("rdf", name))
        plots += _plot_rdf(df, spec, paths, formats, dpi)

        peak_index = int(np.argmax(df["g_r"].to_numpy()))
        rows.append({
            "analysis": "rdf",
            "observable": spec.name,
            "unit": "Å",
            "n_frames": system.n_frames,
            "mean": float(df["r"].iloc[peak_index]),
            "std": float("nan"),
            "min": float(spec.range[0]),
            "max": float(spec.range[1]),
            "definition": f"g(r) {spec.g1} – {spec.g2}",
            "first_peak_r": float(df["r"].iloc[peak_index]),
            "first_peak_g": float(df["g_r"].iloc[peak_index]),
        })

    return AnalysisResult(
        name="rdf",
        title="Radial distribution functions",
        tables=tables,
        summary=pd.DataFrame(rows),
        plots=list(plots),
        notes=["'mean' holds the position of the highest g(r) peak; n(r) is the "
               "running coordination number."],
    )


def _plot_rdf(df: pd.DataFrame, spec: RDFPair, paths: OutputPaths, formats, dpi) -> list:
    import matplotlib.pyplot as plt

    colors = color_cycle(2)
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.plot(df["r"], df["g_r"], color=colors[0], label="g(r)")
    ax.set_xlabel("r (Å)")
    ax.set_ylabel("g(r)")
    ax.set_xlim(spec.range)
    ax.axhline(1.0, color="0.6", linewidth=0.8, linestyle=":")

    twin = ax.twinx()
    twin.plot(df["r"], df["n_r"], color=colors[1], linestyle="--", label="n(r)")
    twin.set_ylabel("n(r) (coordination number)")
    twin.grid(False)

    lines = ax.get_lines()[:1] + twin.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc="best")
    ax.set_title(spec.name)
    fig.tight_layout()
    return save_figure(fig, paths.plots, basename("rdf", spec.name), formats, dpi)[:1]
