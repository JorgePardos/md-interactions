"""Containers used to move results between analysis modules and the report."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

__all__ = ["AnalysisResult", "summary_row"]


@dataclass
class AnalysisResult:
    """Standard return value of every analysis module.

    Attributes
    ----------
    name
        Short module identifier, e.g. ``"distances"``.
    title
        Human readable title used in the report.
    tables
        Mapping ``basename -> DataFrame``.  Each entry is written to
        ``results/data/<basename>.csv``.
    series
        Optional wide-format time series (one column per observable, indexed by
        ``time``).  Collected by the runner so that later modules (e.g. the 2D
        free-energy maps) can reference observables computed elsewhere.
    summary
        Long-format summary rows (one per observable) with columns
        ``analysis, observable, unit, n_frames, mean, std, min, max`` plus any
        extra module-specific column.
    plots
        Paths of every figure produced (only the first format is listed per
        figure, i.e. the PNG).
    notes
        Free-form remarks shown in the report (warnings, parameters used...).
    """

    name: str
    title: str
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    series: pd.DataFrame | None = None
    summary: pd.DataFrame | None = None
    plots: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def summary_row(
    analysis: str,
    observable: str,
    values,
    unit: str,
    **extra,
) -> dict:
    """Build one summary record (mean ± std, min, max) from a 1-D array."""
    import numpy as np

    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    row = {
        "analysis": analysis,
        "observable": observable,
        "unit": unit,
        "n_frames": int(arr.size),
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()) if arr.size else float("nan"),
        "max": float(arr.max()) if arr.size else float("nan"),
    }
    row.update(extra)
    return row
