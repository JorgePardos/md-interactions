"""Reading pre-computed per-frame tables (cpptraj ``.dat``, CSV, ...).

Lets the package work on data you already have — typically the output of
``cpptraj``'s ``distance``/``angle``/``dihedral`` commands, or of any code that
writes one row per frame — without a topology or a trajectory.  This is the
route for Monte Carlo runs, for analyses done elsewhere, and for re-plotting
years-old data in the same style as the rest of the figures.

Expected format (cpptraj's default)::

    #Frame     D20:OD1-R32:HH12   D20:OD2-R32:HH22
         1               2.5962             2.3542
         2               1.8416             1.9695

Any whitespace- or comma-separated table works; the header line may be absent
(columns are then named ``col1``, ``col2``, ...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, DataConfig
from .exceptions import ConfigError, MDInteractionsError

__all__ = ["Dataset", "read_table", "load_dataset"]

#: Header names understood as "this column is the frame/step index".
_INDEX_NAMES = {"frame", "index", "step", "time", "#frame", "n"}


@dataclass
class Dataset:
    """A table of per-frame observables, with the same API the plots expect."""

    data: pd.DataFrame
    columns: list[str]
    unit: str = "Å"
    time_unit: str = "frame"
    sources: list[Path] = field(default_factory=list)
    config: Config | None = None

    @property
    def n_frames(self) -> int:
        return int(len(self.data))

    @property
    def times(self) -> np.ndarray:
        return self.data["time"].to_numpy()

    @property
    def time_label(self) -> str:
        return "Frame" if self.time_unit == "frame" else f"Time ({self.time_unit})"

    @property
    def replica_labels(self) -> np.ndarray | None:
        if "replica" not in self.data.columns:
            return None
        return self.data["replica"].to_numpy()

    @property
    def replica_names(self) -> list[str]:
        labels = self.replica_labels
        if labels is None:
            return []
        seen: list[str] = []
        for label in labels:
            if label not in seen:
                seen.append(str(label))
        return seen

    def replica_masks(self) -> dict[str, np.ndarray]:
        labels = self.replica_labels
        if labels is None:
            return {}
        return {name: labels == name for name in self.replica_names}

    def values(self, column: str) -> np.ndarray:
        return self.data[column].to_numpy(dtype=float)


def read_table(path: str | Path) -> pd.DataFrame:
    """Read one table into a DataFrame with an ``index`` column plus the data.

    The first column is treated as the frame/step index when the header says so
    (``#Frame``, ``#Index``, ...); otherwise every column is data and the index
    is generated.
    """
    path = Path(path)
    if not path.is_file():
        raise MDInteractionsError(f"Data file not found: {path}")

    header: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                header = stripped.lstrip("#").replace(",", " ").split()
            break

    try:
        values = np.loadtxt(path, comments="#", ndmin=2)
    except ValueError:
        try:
            values = np.loadtxt(path, comments="#", delimiter=",", ndmin=2)
        except ValueError as exc:
            raise MDInteractionsError(
                f"Could not parse '{path}' as a numeric table: {exc}\n"
                "  Expected one row per frame with whitespace- or "
                "comma-separated numeric columns."
            ) from exc
    if values.size == 0:
        raise MDInteractionsError(f"Data file '{path}' contains no data rows.")

    n_columns = values.shape[1]
    if header:
        has_index = header[0].lower() in _INDEX_NAMES
    else:
        # no header: assume the leading column is the frame counter when it
        # looks like one (integers in a regular ascending sequence)
        has_index = _looks_like_index(values[:, 0])
    names = header[1:] if (header and has_index) else header
    if len(names) == n_columns - 1 and not has_index:
        # header omits the leading index column although the data has one
        has_index = True
    if len(names) != n_columns - (1 if has_index else 0):
        names = [f"col{i + 1}" for i in range(n_columns - (1 if has_index else 0))]

    if has_index:
        index = values[:, 0]
        data = values[:, 1:]
    else:
        index = np.arange(1, values.shape[0] + 1, dtype=float)
        data = values

    table = pd.DataFrame(data, columns=names)
    table.insert(0, "index", index)
    return table


def _looks_like_index(column: np.ndarray) -> bool:
    """True when the column is a regular ascending integer sequence."""
    if column.size < 2:
        return False
    if not np.allclose(column, np.round(column)):
        return False
    steps = np.diff(column)
    return bool(np.all(steps > 0) and np.allclose(steps, steps[0]))


def load_dataset(config: Config, verbose: bool = True) -> Dataset:
    """Build a :class:`Dataset` from the ``data:`` section of a configuration."""
    data_cfg: DataConfig | None = config.data
    if data_cfg is None:  # pragma: no cover - guarded by the caller
        raise ConfigError("No 'data:' section in the configuration.")

    sources: list[tuple[str | None, Path]] = []
    if data_cfg.replicas:
        for name, paths in data_cfg.replicas.items():
            sources += [(name, path) for path in paths]
    else:
        sources += [(None, path) for path in data_cfg.files]

    frames: list[pd.DataFrame] = []
    reference: list[str] | None = None
    for replica, path in sources:
        table = read_table(path)
        names = [c for c in table.columns if c != "index"]
        if reference is None:
            reference = names
        elif names != reference:
            missing = sorted(set(reference) - set(names))
            extra = sorted(set(names) - set(reference))
            raise MDInteractionsError(
                f"Column mismatch in '{path}':\n"
                f"  expected: {reference}\n"
                f"  found   : {names}\n"
                + (f"  missing: {missing}\n" if missing else "")
                + (f"  unexpected: {extra}\n" if extra else "")
                + "  All files must share the same columns in the same order; "
                "otherwise the observables would be silently mixed."
            )
        if replica is not None:
            table["replica"] = replica
        frames.append(table)

    table = pd.concat(frames, ignore_index=True)

    # optional column subset / renaming
    available = [c for c in table.columns if c not in ("index", "replica")]
    selected = data_cfg.columns or available
    unknown = [c for c in selected if c not in available]
    if unknown:
        raise ConfigError(
            f"'data.columns' lists unknown column(s) {unknown}.\n"
            f"  Available: {available}"
        )
    renamed = {c: data_cfg.labels.get(c, c) for c in selected}
    duplicates = [name for name in renamed.values()
                  if list(renamed.values()).count(name) > 1]
    if duplicates:
        raise ConfigError(f"'data.labels' produces duplicated names: {sorted(set(duplicates))}")

    ordered = ["index"] + (["replica"] if "replica" in table.columns else []) + selected
    table = table[ordered].rename(columns=renamed)
    columns = [renamed[c] for c in selected]

    # frame range / stride, applied per replica so every run is trimmed alike
    frames_cfg = data_cfg.frames
    if "replica" in table.columns:
        pieces = [group.iloc[frames_cfg.start:frames_cfg.stop:frames_cfg.stride]
                  for _name, group in table.groupby("replica", sort=False)]
        table = pd.concat(pieces, ignore_index=True)
    else:
        table = table.iloc[frames_cfg.start:frames_cfg.stop:frames_cfg.stride]
        table = table.reset_index(drop=True)
    if table.empty:
        raise MDInteractionsError(
            f"The requested frame range (start={frames_cfg.start}, "
            f"stop={frames_cfg.stop}, stride={frames_cfg.stride}) selects no rows."
        )

    # time axis
    dt = data_cfg.time.dt
    unit = data_cfg.time.unit
    frame_index = table["index"].to_numpy(dtype=float)
    table.insert(0, "frame", frame_index.astype(int))
    table["time"] = frame_index * dt if dt is not None else frame_index
    table = table.drop(columns="index")
    leading = ["frame", "time"] + (["replica"] if "replica" in table.columns else [])
    table = table[leading + columns]

    dataset = Dataset(
        data=table.reset_index(drop=True),
        columns=columns,
        unit=data_cfg.unit,
        time_unit=unit if dt is not None else "frame",
        sources=[path for _replica, path in sources],
        config=config,
    )
    if verbose:
        print(f"[md_interactions] {len(dataset.data)} rows, "
              f"{len(columns)} column(s) from {len(sources)} file(s)")
        if dataset.replica_labels is not None:
            counts = {name: int(mask.sum())
                      for name, mask in dataset.replica_masks().items()}
            listed = ", ".join(f"{k}: {v}" for k, v in counts.items())
            print(f"[md_interactions] replicas ({listed} rows)")
    return dataset
