"""Analysis modules.

Every module exposes a ``run(system, paths, config=None, verbose=True)``
function returning an :class:`~md_interactions.results.AnalysisResult`, plus a
lower level ``compute_*`` function that returns plain
:class:`pandas.DataFrame` objects for interactive use in notebooks.
"""

from __future__ import annotations

__all__ = [
    "distances",
    "angles_dihedrals",
    "rmsd_rmsf",
    "hbonds",
    "free_energy_map",
    "radius_of_gyration",
    "rdf",
    "clustering",
]
