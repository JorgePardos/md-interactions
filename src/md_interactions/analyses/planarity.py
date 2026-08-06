"""Planarity of a centre and its three substituents.

For an atom X bonded to A, B and C, planarity is how far X sits from the plane
those three define.  ``0`` means perfectly planar — sp2 — and the value grows as
the centre pyramidalises towards sp3.

Two numbers are reported per definition:

``<name>``
    the **out-of-plane distance** in Å, the direct measure of the question.
    It is *signed*: the sign follows the right-hand rule on A → B → C, so a
    centre that crosses the plane changes sign.  For a glycosyl transfer or any
    reaction that inverts configuration at a centre, that sign change is the
    event, and an absolute value would hide it.
``<name>_angle_sum``
    the sum of the three angles at the centre, in degrees.  ``360`` is planar,
    ``328.4`` is ideal tetrahedral.  Independent of bond lengths, which makes
    it the quantity to quote when comparing centres with different substituents.

A note on averaging the signed distance: a centre oscillating symmetrically
about the plane averages to ~0 while never actually being planar.  The run
therefore also reports ``<name>_abs``, the mean of ``|d|``, which is the
pyramidalisation proper.  The two together tell the whole story — a small mean
with a large spread is an oscillation, a small mean with a small spread is a
genuinely flat centre.

The improper dihedral used by force fields is not duplicated here: it can be
written directly in the ``[dihedrals]`` section, which already takes any four
atoms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from MDAnalysis.lib.distances import minimize_vectors

from ..config import PlanarityConfig
from ..exceptions import AnalysisError
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

__all__ = ["run", "compute_planarity", "out_of_plane"]

_UNIT = "Å"

#: Sum of the three angles at an ideal tetrahedral centre, for reference.
TETRAHEDRAL_ANGLE_SUM = 328.4


def out_of_plane(centre: np.ndarray, a: np.ndarray, b: np.ndarray,
                 c: np.ndarray, box=None) -> tuple[float, float]:
    """Signed distance from ``centre`` to the plane of ``a``, ``b``, ``c``.

    Returns ``(distance, angle_sum)`` in Å and degrees.  All vectors are taken
    through the minimum image when a box is given: three atoms bonded to the
    same centre can end up on opposite sides of a wrapped trajectory, and the
    plane through them would then be meaningless.
    """
    ab, ac = b - a, c - a
    if box is not None:
        ab = minimize_vectors(ab.reshape(1, 3), box)[0]
        ac = minimize_vectors(ac.reshape(1, 3), box)[0]
    normal = np.cross(ab, ac)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9:
        # Collinear substituents: every plane through them is equally valid,
        # so there is no out-of-plane distance to report.
        return float("nan"), float("nan")

    to_centre = centre - a
    if box is not None:
        to_centre = minimize_vectors(to_centre.reshape(1, 3), box)[0]
    distance = float(np.dot(to_centre, normal) / norm)

    # The angles are measured at the centre, so the vectors point outwards
    # from it; this is the bond-length-independent view of the same geometry.
    arms = []
    for point in (a, b, c):
        arm = point - centre
        if box is not None:
            arm = minimize_vectors(arm.reshape(1, 3), box)[0]
        arms.append(arm / max(float(np.linalg.norm(arm)), 1e-9))
    angle_sum = float(sum(
        np.degrees(np.arccos(np.clip(np.dot(arms[i], arms[j]), -1.0, 1.0)))
        for i, j in ((0, 1), (1, 2), (0, 2))
    ))
    return distance, angle_sum


def compute_planarity(
    system: MDSystem, config: PlanarityConfig | None = None
) -> pd.DataFrame:
    """One column per definition plus ``<name>_angle_sum``, frame by frame."""
    config = config or system.config.planarity
    if not config.definitions:
        raise AnalysisError("No planarity definitions were configured.")

    groups = [
        (defn.name,
         [resolve_group(system, token, defn.mode, defn.name) for token in defn.atoms])
        for defn in config.definitions
    ]

    n_frames = system.n_frames
    distances = np.full((n_frames, len(groups)), np.nan)
    angle_sums = np.full((n_frames, len(groups)), np.nan)

    for i, _ts in enumerate(system.iter_frames()):
        box = current_box(system, config.pbc)
        for j, (_name, members) in enumerate(groups):
            centre, a, b, c = (_point(g) for g in members)
            distances[i, j], angle_sums[i, j] = out_of_plane(centre, a, b, c, box)

    frame = pd.DataFrame(system.time_frame_columns())
    for j, (name, _g) in enumerate(groups):
        frame[name] = distances[:, j]
        frame[f"{name}_angle_sum"] = angle_sums[:, j]
    return frame


def _point(group: GeometryGroup) -> np.ndarray:
    return np.asarray(group.position(), dtype=np.float64).reshape(3)


def run(
    system: MDSystem,
    paths: OutputPaths,
    config: PlanarityConfig | None = None,
    verbose: bool = True,
) -> AnalysisResult:
    """Run every configured planarity and write tables and figures."""
    config = config or system.config.planarity
    if verbose:
        print(f"[planarity] {len(config.definitions)} centre(s) over "
              f"{system.n_frames} frames")

    frame = compute_planarity(system, config)
    formats = system.config.output.formats
    dpi = system.config.output.dpi
    rows: list[dict] = []
    notes: list[str] = []
    plots: list = []

    for defn in config.definitions:
        values = frame[defn.name].to_numpy()
        angle_sum = frame[f"{defn.name}_angle_sum"].to_numpy()
        label = _label(system, defn)

        rows.append(summary_row("planarity", defn.name, values, _UNIT,
                                definition=f"{label} (signed, 0 = planar)"))
        rows.append(summary_row("planarity", f"{defn.name}_abs",
                                np.abs(values), _UNIT,
                                definition=f"{label} (magnitude)"))
        rows.append(summary_row("planarity", f"{defn.name}_angle_sum",
                                angle_sum, "°",
                                definition=f"{label} (360 = planar, "
                                           f"{TETRAHEDRAL_ANGLE_SUM:g} = tetrahedral)"))

        clean = values[~np.isnan(values)]
        if clean.size == 0:
            notes.append(f"'{defn.name}': the three substituents are collinear in "
                         "every frame, so no plane is defined.")
            continue
        mean_abs = float(np.mean(np.abs(clean)))
        crossings = int(np.count_nonzero(np.diff(np.sign(clean)) != 0))
        notes.append(
            f"'{defn.name}': |d| = {mean_abs:.3f} Å on average "
            f"(signed {clean.mean():+.3f} ± {clean.std():.3f} Å), angles summing "
            f"to {np.nanmean(angle_sum):.1f}°. "
            + _interpret(mean_abs, float(np.nanmean(angle_sum)), crossings)
        )

    plots += _plot(system, paths, config, frame, formats, dpi)

    series_columns = [c for c in frame.columns if c not in ("frame", "time", "replica")]
    write_csv(frame, paths.data, "planarity")
    return AnalysisResult(
        name="planarity",
        title="Planarity of a centre and its three substituents",
        tables={"planarity": frame},
        series=frame[["frame", "time", *series_columns]],
        summary=pd.DataFrame(rows),
        plots=list(plots),
        notes=notes + [
            "The out-of-plane distance is signed by the right-hand rule on the "
            "three substituents in the order given, so a centre that crosses "
            "the plane changes sign; '_abs' holds the magnitude, which is what "
            "a mean should be taken over.",
        ],
    )


def _interpret(mean_abs: float, angle_sum: float, crossings: int) -> str:
    """One sentence saying what the numbers mean chemically."""
    if crossings > 0 and mean_abs < 0.15:
        return (f"The centre crosses the plane {crossings} time(s): it is "
                "essentially planar and fluctuating through it.")
    if mean_abs < 0.10 or angle_sum > 355.0:
        return "Planar within thermal fluctuation — sp2."
    if angle_sum < 340.0 or mean_abs > 0.30:
        return "Clearly pyramidal — sp3 character."
    return "Intermediate: partially pyramidalised."


def _label(system: MDSystem, defn) -> str:
    if defn.label:
        return defn.label
    try:
        described = [system.describe_selection(token) for token in defn.atoms]
    except Exception:                                    # pragma: no cover
        described = list(defn.atoms)
    return f"{described[0]} out of the plane of {', '.join(described[1:])}"


def _plot(system, paths, config, frame, formats, dpi) -> list:
    """Time series and distribution of every centre, plus the angle sums."""
    plots: list = []
    time = frame["time"].to_numpy()
    names = [defn.name for defn in config.definitions]
    labels = {defn.name: _label(system, defn) for defn in config.definitions}

    for defn in config.definitions:
        values = frame[defn.name].to_numpy()
        base = basename("planarity", defn.name)
        fig = plot_timeseries(time, {labels[defn.name]: values},
                              xlabel=system.time_label,
                              ylabel=f"Out-of-plane distance ({_UNIT})",
                              title=defn.name)
        # the planar reference is the whole point of the figure
        axis = fig.axes[0]
        axis.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
        axis.text(0.995, 0.0, " planar", fontsize=8, color="0.35",
                  transform=axis.get_yaxis_transform(), ha="right", va="bottom")
        plots += save_figure(fig, paths.plots, f"{base}_timeseries", formats, dpi)[:1]

        fig = plot_distribution({labels[defn.name]: values},
                                xlabel=f"Out-of-plane distance ({_UNIT})",
                                bins=config.bins, kde=config.kde, title=defn.name,
                                normalization=config.normalization, unit=_UNIT)
        fig.axes[0].axvline(0.0, color="0.35", linestyle="--", linewidth=1.0)
        plots += save_figure(fig, paths.plots, f"{base}_histogram", formats, dpi)[:1]

    if len(names) > 1:
        series = {labels[name]: frame[name].to_numpy() for name in names}
        fig = plot_timeseries(time, series, xlabel=system.time_label,
                              ylabel=f"Out-of-plane distance ({_UNIT})",
                              title=None, figsize=(7.2, 4.0))
        fig.axes[0].axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
        plots += save_figure(fig, paths.plots, "planarity_all_timeseries",
                             formats, dpi)[:1]
        if config.facet:
            fig = plot_facet_distributions(
                {name: frame[name].to_numpy() for name in names},
                xlabel=f"Out-of-plane distance ({_UNIT})", unit=_UNIT,
                bins=min(config.bins, 40), kde=config.kde,
                normalization=config.normalization,
                groups=replica_groups(system, frame, names),
            )
            plots += save_figure(fig, paths.plots, "planarity_distributions",
                                 formats, dpi)[:1]
    return plots
