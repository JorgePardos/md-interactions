"""Conformational clustering with extraction of representative structures.

Workflow
--------
1. The coordinates of the clustering selection (typically the active site or
   the ligand) are collected for the analysed frames, subsampling if there are
   more than ``max_frames``.
2. Frames are superposed onto their average structure, so the pairwise RMSD
   reduces to a Euclidean distance in the 3N-dimensional space
   (``RMSD_ij = |x_i - x_j| / sqrt(N)``); this is what makes clustering of
   thousands of frames tractable.
3. Hierarchical (``scipy.cluster.hierarchy``) or k-means (``scipy.cluster.vq``)
   clustering is applied.
4. For every cluster the medoid — the frame with the smallest average RMSD to
   the rest of its cluster — is written as a PDB, ready to be used as the
   starting point of a QM/MM calculation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

from ..config import ClusteringConfig
from ..exceptions import AnalysisError
from ..io_utils import OutputPaths, write_csv
from ..plotting import color_cycle, save_figure
from ..results import AnalysisResult
from ..system import MDSystem
from .rmsd_rmsf import _iterative_fit

__all__ = ["run", "compute_clusters"]

_UNIT = "Å"


def compute_clusters(
    system: MDSystem, config: ClusteringConfig | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Cluster the trajectory and locate the representative frame of each cluster.

    Returns
    -------
    (assignment, summary, coordinates)
        ``assignment`` has one row per clustered frame (``frame``, ``time``,
        ``cluster``, ``rmsd_to_medoid``); ``summary`` one row per cluster;
        ``coordinates`` are the aligned coordinates used, shape
        ``(n_used, n_atoms, 3)`` (handy for custom projections).
    """
    config = config or system.config.clustering
    atoms = system.select(config.selection, name="clustering.selection")
    if atoms.n_atoms < 3:
        raise AnalysisError(
            f"Clustering needs at least 3 atoms, the selection matched {atoms.n_atoms}."
        )

    positions = np.arange(system.n_frames)
    if system.n_frames > config.max_frames:
        positions = np.unique(
            np.linspace(0, system.n_frames - 1, config.max_frames).astype(int)
        )

    coords = np.empty((positions.size, atoms.n_atoms, 3), dtype=np.float64)
    wanted = set(positions.tolist())
    cursor = 0
    for i, _ts in enumerate(system.iter_frames()):
        if i in wanted:
            coords[cursor] = atoms.positions
            cursor += 1

    if config.align:
        try:
            weights = np.asarray(atoms.masses, dtype=float)
            if not np.all(np.isfinite(weights)) or weights.sum() <= 0:
                weights = None
        except Exception:
            weights = None
        coords = _iterative_fit(coords, np.arange(atoms.n_atoms), weights)

    flat = coords.reshape(coords.shape[0], -1)
    condensed = pdist(flat, metric="euclidean") / np.sqrt(atoms.n_atoms)
    distances = squareform(condensed)

    labels = _cluster_labels(flat, condensed, config)

    frame_indices = system.frame_indices[positions]
    times = system.times[positions]
    assignment = pd.DataFrame({
        "frame": frame_indices,
        "time": times,
        "cluster": labels,
        "rmsd_to_medoid": np.nan,
    })

    rows = []
    for cluster_id in sorted(set(labels)):
        members = np.flatnonzero(labels == cluster_id)
        block = distances[np.ix_(members, members)]
        medoid_local = int(members[np.argmin(block.mean(axis=1))])
        rmsd_to_medoid = distances[medoid_local, members]
        assignment.loc[members, "rmsd_to_medoid"] = rmsd_to_medoid
        rows.append({
            "cluster": int(cluster_id),
            "population": int(members.size),
            "population_%": 100.0 * members.size / labels.size,
            "representative_frame": int(frame_indices[medoid_local]),
            "representative_time": float(times[medoid_local]),
            "mean_rmsd_to_medoid": float(rmsd_to_medoid.mean()),
            "max_rmsd_to_medoid": float(rmsd_to_medoid.max()),
            "_position": int(positions[medoid_local]),
        })

    summary = pd.DataFrame(rows).sort_values("population", ascending=False)
    summary = summary.reset_index(drop=True)
    return assignment, summary, coords


def _cluster_labels(flat: np.ndarray, condensed: np.ndarray,
                    config: ClusteringConfig) -> np.ndarray:
    """Cluster labels (1-based) for the requested method."""
    n_clusters = min(config.n_clusters, flat.shape[0])
    if n_clusters < 1:
        raise AnalysisError("'analyses.clustering.n_clusters' must be >= 1.")

    if config.method == "kmeans":
        from scipy.cluster.vq import kmeans2

        _centroids, labels = kmeans2(flat, n_clusters, minit="++", seed=0, missing="warn")
        return np.asarray(labels, dtype=int) + 1

    tree = linkage(condensed, method=config.linkage)
    return np.asarray(fcluster(tree, t=n_clusters, criterion="maxclust"), dtype=int)


def run(
    system: MDSystem,
    paths: OutputPaths,
    config: ClusteringConfig | None = None,
    verbose: bool = True,
) -> AnalysisResult:
    """Run the clustering and write tables, figures and representative PDBs."""
    config = config or system.config.clustering
    if verbose:
        print(f"[clustering] {config.method} on \"{config.selection}\" "
              f"({config.n_clusters} clusters)")

    assignment, summary, coords = compute_clusters(system, config)
    write_csv(assignment, paths.data, "cluster_assignment")
    public_summary = summary.drop(columns="_position")
    write_csv(public_summary, paths.summary, "clusters")

    formats = system.config.output.formats
    dpi = system.config.output.dpi
    plots = _plot_clusters(system, assignment, public_summary, coords, paths, formats, dpi)

    notes = [
        f"Method: {config.method}"
        + (f" (linkage: {config.linkage})" if config.method == "hierarchical" else "")
        + f", selection \"{config.selection}\", {len(assignment)} frames clustered.",
        "Pairwise RMSD computed after superposition onto the average structure.",
    ]
    if config.write_structures:
        written = _write_structures(system, summary, paths,
                                    write_selection=config.write_selection,
                                    verbose=verbose)
        scope = (f"selection \"{config.write_selection}\"" if config.write_selection
                 else "whole system")
        notes.append(f"Representative structures ({scope}): "
                     + ", ".join(p.name for p in written))

    rows = [{
        "analysis": "clustering",
        "observable": f"cluster_{int(row['cluster'])}",
        "unit": _UNIT,
        "n_frames": int(row["population"]),
        "mean": float(row["mean_rmsd_to_medoid"]),
        "std": float("nan"),
        "min": 0.0,
        "max": float(row["max_rmsd_to_medoid"]),
        "definition": f"{row['population_%']:.1f}% of frames, "
                      f"representative frame {int(row['representative_frame'])}",
    } for _i, row in public_summary.iterrows()]

    return AnalysisResult(
        name="clustering",
        title="Conformational clustering",
        tables={"cluster_assignment": assignment, "clusters": public_summary},
        summary=pd.DataFrame(rows),
        plots=list(plots),
        notes=notes,
    )


def _write_structures(system: MDSystem, summary: pd.DataFrame, paths: OutputPaths,
                      write_selection: str | None = None,
                      verbose: bool = True) -> list:
    """Write the medoid of every cluster as a PDB.

    The whole system is written by default (that is what you need to restart a
    QM/MM calculation or to keep the bridging waters).  Set
    ``analyses.clustering.write_selection`` to write only a region — the file
    then opens instantly in a viewer, at the cost of losing the environment.
    """
    paths.structures.mkdir(parents=True, exist_ok=True)
    atoms = (system.select(write_selection, name="clustering.write_selection")
             if write_selection else system.universe.atoms)
    written = []
    for _i, row in summary.iterrows():
        cluster = int(row["cluster"])
        frame = int(row["representative_frame"])
        system.goto(int(row["_position"]))
        path = paths.structures / f"cluster{cluster}_frame{frame}.pdb"
        remark = (f"cluster {cluster} medoid, frame {frame}, "
                  f"t = {row['representative_time']:g} "
                  f"{system.time_unit}, {int(row['population'])} frames "
                  f"({row['population_%']:.1f}%)")
        try:
            atoms.write(str(path), remarks=remark)
        except TypeError:  # pragma: no cover - writer without 'remarks'
            atoms.write(str(path))
        written.append(path)
        if verbose:
            print(f"[clustering] cluster {cluster} -> {path.name} "
                  f"({atoms.n_atoms} atoms)")
    return written


def _plot_clusters(system, assignment, summary, coords, paths, formats, dpi) -> list:
    import matplotlib.pyplot as plt

    plots: list = []
    labels = assignment["cluster"].to_numpy()
    unique = sorted(set(labels.tolist()))
    colors = {c: col for c, col in zip(unique, color_cycle(len(unique)))}

    # 1) cluster membership along time + populations
    fig, (ax_time, ax_pop) = plt.subplots(
        1, 2, figsize=(8.4, 3.4), gridspec_kw={"width_ratios": [3, 1]}
    )
    for cluster_id in unique:
        mask = labels == cluster_id
        ax_time.scatter(assignment["time"].to_numpy()[mask],
                        np.full(mask.sum(), cluster_id),
                        s=8, color=colors[cluster_id], label=f"C{cluster_id}")
    ax_time.set_xlabel(system.time_label)
    ax_time.set_ylabel("Cluster")
    ax_time.set_yticks(unique)
    ax_time.set_yticklabels([f"C{c}" for c in unique])

    populations = summary.sort_values("cluster")
    ax_pop.bar([f"C{int(c)}" for c in populations["cluster"]],
               populations["population_%"],
               color=[colors[int(c)] for c in populations["cluster"]], alpha=0.85)
    ax_pop.set_ylabel("Population (%)")
    ax_pop.grid(axis="x", visible=False)
    fig.tight_layout()
    plots += save_figure(fig, paths.plots, "cluster_populations", formats, dpi)[:1]

    # 2) PCA projection of the aligned coordinates, coloured by cluster
    flat = coords.reshape(coords.shape[0], -1)
    centred = flat - flat.mean(axis=0, keepdims=True)
    try:
        _u, singular, vectors = np.linalg.svd(centred, full_matrices=False)
        projection = centred @ vectors[:2].T
        variance = (singular ** 2) / max((centred.shape[0] - 1), 1)
        explained = 100.0 * variance[:2] / variance.sum()
    except np.linalg.LinAlgError:  # pragma: no cover
        return plots

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for cluster_id in unique:
        mask = labels == cluster_id
        ax.scatter(projection[mask, 0], projection[mask, 1], s=14, alpha=0.75,
                   color=colors[cluster_id], label=f"C{cluster_id}")
    ax.set_xlabel(f"PC1 ({explained[0]:.0f}% of variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.0f}% of variance)")
    ax.legend(loc="best")
    fig.tight_layout()
    plots += save_figure(fig, paths.plots, "cluster_pca", formats, dpi)[:1]
    return plots
