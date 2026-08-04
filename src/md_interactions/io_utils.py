"""Output directory layout, CSV writing and Markdown table formatting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

__all__ = ["OutputPaths", "prepare_output", "write_csv", "to_markdown_table"]


@dataclass(frozen=True)
class OutputPaths:
    """The ``results/`` directory tree used by every run."""

    root: Path
    plots: Path
    data: Path
    summary: Path
    structures: Path

    def relative(self, path: Path) -> str:
        """POSIX-style path relative to :attr:`root` (for report links)."""
        try:
            return Path(path).resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:  # pragma: no cover - path outside results/
            return Path(path).as_posix()


def prepare_output(directory: str | Path) -> OutputPaths:
    """Create ``results/{plots,data,summary,structures}`` and return the paths."""
    root = Path(directory)
    paths = OutputPaths(
        root=root,
        plots=root / "plots",
        data=root / "data",
        summary=root / "summary",
        structures=root / "structures",
    )
    for path in (paths.root, paths.plots, paths.data, paths.summary):
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_csv(df: pd.DataFrame, directory: Path, basename: str,
              float_format: str = "%.4f") -> Path:
    """Write ``df`` to ``directory/basename.csv`` (index excluded)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{basename}.csv"
    df.to_csv(path, index=False, float_format=float_format)
    return path


def to_markdown_table(df: pd.DataFrame, float_format: str = "{:.3f}") -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table.

    Implemented locally to avoid a hard dependency on ``tabulate``.
    """
    if df.empty:
        return "_(no data)_"

    def fmt(value) -> str:
        if isinstance(value, float):
            if pd.isna(value):
                return "-"
            return float_format.format(value)
        return "" if value is None else str(value)

    columns = [str(c) for c in df.columns]
    rows = [[fmt(v) for v in row] for row in df.itertuples(index=False, name=None)]
    widths = [max(len(columns[i]), *(len(r[i]) for r in rows)) if rows else len(columns[i])
              for i in range(len(columns))]
    header = "| " + " | ".join(c.ljust(w) for c, w in zip(columns, widths)) + " |"
    rule = "| " + " | ".join("-" * w for w in widths) + " |"
    body = ["| " + " | ".join(v.ljust(w) for v, w in zip(row, widths)) + " |"
            for row in rows]
    return "\n".join([header, rule, *body])
