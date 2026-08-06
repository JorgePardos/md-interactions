"""Typed configuration objects and YAML loading for :mod:`md_interactions`.

The whole tool is driven by a single YAML file.  Every section is optional
except ``system.topology`` and ``system.trajectory``; analyses that are not
declared (or declared with ``enabled: false``) are simply skipped.

Unknown keys raise :class:`~md_interactions.exceptions.ConfigError` on purpose:
a silent typo in a YAML key is one of the easiest ways to lose an afternoon.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .exceptions import ConfigError

__all__ = [
    "Config",
    "DataConfig",
    "SystemConfig",
    "TimeConfig",
    "FrameConfig",
    "AlignConfig",
    "OutputConfig",
    "DistancesConfig",
    "DistanceDef",
    "AnglesConfig",
    "GeometryDef",
    "RMSDConfig",
    "RMSDGroup",
    "RMSFConfig",
    "HBondsConfig",
    "HBondDef",
    "FreeEnergyConfig",
    "FreeEnergyMap",
    "RgyrConfig",
    "NamedSelection",
    "RDFConfig",
    "RDFPair",
    "BridgeConfig",
    "BridgePair",
    "ClusteringConfig",
    "ReportConfig",
    "load_config",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_mapping(value: Any, where: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"'{where}' must be a mapping, got {type(value).__name__}.")
    return dict(value)


def _check_keys(data: Mapping[str, Any], allowed: Sequence[str], where: str) -> None:
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ConfigError(
            f"Unknown key(s) {unknown} in '{where}'. Allowed keys: {sorted(allowed)}."
        )


def _as_list(value: Any, where: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or isinstance(value, Mapping):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    raise ConfigError(f"'{where}' must be a list.")


def _require(data: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in data or data[key] is None:
        raise ConfigError(f"'{where}' requires the key '{key}'.")
    return data[key]


def _atoms(data: Mapping[str, Any], where: str, n: int) -> list[str]:
    atoms = _as_list(_require(data, "atoms", where), f"{where}.atoms")
    if len(atoms) != n:
        raise ConfigError(
            f"'{where}.atoms' must contain exactly {n} entries, got {len(atoms)}."
        )
    return [str(a) for a in atoms]


def _normalization(value: Any, where: str, default: str = "density") -> str:
    """Validate the histogram normalisation of a section."""
    norm = str(value or default).lower()
    if norm not in {"density", "percent", "counts"}:
        raise ConfigError(
            f"'{where}.normalization' must be 'density', 'percent' or 'counts' "
            f"(got '{norm}'). Only 'density' stays comparable between panels "
            "whose bin widths differ."
        )
    return norm


def _mode(data: Mapping[str, Any], where: str, default: str = "atom") -> str:
    mode = str(data.get("mode", default)).lower()
    if mode not in {"atom", "com", "min"}:
        raise ConfigError(
            f"'{where}.mode' must be one of 'atom', 'com', 'min' (got '{mode}')."
        )
    return mode


# --------------------------------------------------------------------------- #
# system / IO
# --------------------------------------------------------------------------- #
@dataclass
class TimeConfig:
    """Time axis definition.

    ``dt`` is the time between *saved* frames of the input trajectory, expressed
    in ``unit``.  If ``dt`` is ``None`` the value stored in the trajectory
    (``universe.trajectory.dt``, in ps) is used instead.
    """

    dt: float | None = None
    unit: str = "ns"

    @classmethod
    def from_dict(cls, data: Any) -> "TimeConfig":
        data = _as_mapping(data, "system.time")
        _check_keys(data, ["dt", "unit"], "system.time")
        unit = str(data.get("unit", "ns")).lower()
        if unit not in {"ps", "ns", "frame"}:
            raise ConfigError("'system.time.unit' must be 'ps', 'ns' or 'frame'.")
        dt = data.get("dt")
        return cls(dt=float(dt) if dt is not None else None, unit=unit)


@dataclass
class FrameConfig:
    """Frame range and stride applied to every analysis."""

    start: int = 0
    stop: int | None = None
    stride: int = 1

    @classmethod
    def from_dict(cls, data: Any) -> "FrameConfig":
        data = _as_mapping(data, "system.frames")
        _check_keys(data, ["start", "stop", "stride", "step"], "system.frames")
        stride = data.get("stride", data.get("step", 1))
        stop = data.get("stop")
        cfg = cls(
            start=int(data.get("start", 0)),
            stop=int(stop) if stop is not None else None,
            stride=int(stride),
        )
        if cfg.stride < 1:
            raise ConfigError("'system.frames.stride' must be >= 1.")
        if cfg.start < 0:
            raise ConfigError("'system.frames.start' must be >= 0.")
        return cfg

    def as_slice_kwargs(self) -> dict[str, Any]:
        return {"start": self.start, "stop": self.stop, "step": self.stride}


@dataclass
class AlignConfig:
    """Optional global RMSD fit applied before measuring anything.

    Distances, angles and dihedrals are invariant under rigid-body motion, so
    this is only needed for visual inspection or when writing aligned
    structures.  RMSF and clustering perform their own internal alignment.
    """

    enabled: bool = False
    selection: str = "protein and name CA"
    reference: Path | None = None

    @classmethod
    def from_dict(cls, data: Any) -> "AlignConfig":
        data = _as_mapping(data, "system.align")
        _check_keys(data, ["enabled", "selection", "reference"], "system.align")
        ref = data.get("reference")
        return cls(
            enabled=bool(data.get("enabled", False)),
            selection=str(data.get("selection", "protein and name CA")),
            reference=Path(ref) if ref else None,
        )


@dataclass
class SystemConfig:
    """Topology, trajectory (possibly several files) and frame handling.

    Trajectories can be declared either as a flat list (they are concatenated
    and treated as one continuous run) or as ``replicas``, a mapping
    ``name -> [files]``.  With ``replicas`` the frames keep the identity of the
    run they come from, so distributions can be compared replica by replica
    instead of being silently pooled.
    """

    topology: Path
    trajectory: list[Path]
    replicas: dict[str, list[Path]] = field(default_factory=dict)
    time: TimeConfig = field(default_factory=TimeConfig)
    frames: FrameConfig = field(default_factory=FrameConfig)
    align: AlignConfig = field(default_factory=AlignConfig)

    @classmethod
    def from_dict(cls, data: Any) -> "SystemConfig":
        data = _as_mapping(data, "system")
        _check_keys(
            data,
            ["topology", "trajectory", "replicas", "time", "frames", "align"],
            "system",
        )
        replicas: dict[str, list[Path]] = {}
        raw_replicas = _as_mapping(data.get("replicas"), "system.replicas")
        for name, files in raw_replicas.items():
            paths = [Path(str(p)) for p in
                     _as_list(files, f"system.replicas.{name}")]
            if not paths:
                raise ConfigError(f"'system.replicas.{name}' lists no file.")
            replicas[str(name)] = paths

        if replicas and data.get("trajectory"):
            raise ConfigError(
                "Use either 'system.trajectory' or 'system.replicas', not both."
            )
        if replicas:
            traj = [path for paths in replicas.values() for path in paths]
        else:
            traj = [Path(str(p)) for p in
                    _as_list(_require(data, "trajectory", "system"),
                             "system.trajectory")]
        if not traj:
            raise ConfigError("'system.trajectory' must list at least one file.")
        return cls(
            topology=Path(str(_require(data, "topology", "system"))),
            trajectory=traj,
            replicas=replicas,
            time=TimeConfig.from_dict(data.get("time")),
            frames=FrameConfig.from_dict(data.get("frames")),
            align=AlignConfig.from_dict(data.get("align")),
        )


@dataclass
class DataConfig:
    """Pre-computed per-frame tables used instead of a trajectory.

    Point it at ``cpptraj`` output (or any table with one row per frame) and
    the package will produce the same distributions, summaries and 2D maps
    without reading a topology.  Declaring ``replicas`` keeps every run
    identifiable instead of pooling them.
    """

    files: list[Path] = field(default_factory=list)
    replicas: dict[str, list[Path]] = field(default_factory=dict)
    columns: list[str] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    unit: str = "Å"
    time: TimeConfig = field(default_factory=lambda: TimeConfig(dt=None, unit="frame"))
    frames: FrameConfig = field(default_factory=FrameConfig)
    bins: int = 30
    kde: bool = True
    facet: bool = True
    normalization: str = "density"
    timeseries: bool = True

    @classmethod
    def from_dict(cls, data: Any) -> "DataConfig | None":
        if data is None:
            return None
        where = "data"
        data = _as_mapping(data, where)
        _check_keys(
            data,
            ["files", "replicas", "columns", "labels", "thresholds", "unit",
             "time", "frames", "bins", "kde", "facet", "normalization", "timeseries"],
            where,
        )
        replicas: dict[str, list[Path]] = {}
        for name, files in _as_mapping(data.get("replicas"), f"{where}.replicas").items():
            paths = [Path(str(p)) for p in _as_list(files, f"{where}.replicas.{name}")]
            if not paths:
                raise ConfigError(f"'{where}.replicas.{name}' lists no file.")
            replicas[str(name)] = paths

        files = [Path(str(p)) for p in _as_list(data.get("files"), f"{where}.files")]
        if files and replicas:
            raise ConfigError(f"Use either '{where}.files' or '{where}.replicas', not both.")
        if not files and not replicas:
            raise ConfigError(f"'{where}' requires 'files' or 'replicas'.")

        thresholds = {str(k): float(v) for k, v in
                      _as_mapping(data.get("thresholds"), f"{where}.thresholds").items()}
        return cls(
            files=files,
            replicas=replicas,
            columns=[str(c) for c in _as_list(data.get("columns"), f"{where}.columns")],
            labels={str(k): str(v) for k, v in
                    _as_mapping(data.get("labels"), f"{where}.labels").items()},
            thresholds=thresholds,
            unit=str(data.get("unit", "Å")),
            time=TimeConfig.from_dict(data.get("time")) if data.get("time")
            else TimeConfig(dt=None, unit="frame"),
            frames=FrameConfig.from_dict(data.get("frames")),
            bins=int(data.get("bins", 30)),
            kde=bool(data.get("kde", True)),
            facet=bool(data.get("facet", True)),
            normalization=_normalization(data.get("normalization"), where),
            timeseries=bool(data.get("timeseries", True)),
        )


@dataclass
class OutputConfig:
    """Where results go and in which formats."""

    directory: Path = Path("results")
    formats: list[str] = field(default_factory=lambda: ["png", "pdf"])
    dpi: int = 300
    style: str = "md_interactions"

    @classmethod
    def from_dict(cls, data: Any) -> "OutputConfig":
        data = _as_mapping(data, "output")
        _check_keys(data, ["directory", "formats", "dpi", "style"], "output")
        formats = [str(f).lower().lstrip(".")
                   for f in _as_list(data.get("formats", ["png", "pdf"]), "output.formats")]
        bad = sorted(set(formats) - {"png", "pdf", "svg", "eps", "tiff", "jpg"})
        if bad:
            raise ConfigError(f"Unsupported output format(s): {bad}.")
        return cls(
            directory=Path(str(data.get("directory", "results"))),
            formats=formats or ["png"],
            dpi=int(data.get("dpi", 300)),
            style=str(data.get("style", "md_interactions")),
        )


# --------------------------------------------------------------------------- #
# geometric observables
# --------------------------------------------------------------------------- #
@dataclass
class DistanceDef:
    """A single distance between two selections.

    ``threshold`` is optional and purely diagnostic: when given, the summary
    reports the fraction of frames below it (e.g. the population of
    near-attack conformations) and the time series shows it as a dashed line.
    """

    name: str
    atoms: list[str]
    mode: str = "atom"
    label: str | None = None
    threshold: float | None = None

    @classmethod
    def from_dict(cls, data: Any, index: int) -> "DistanceDef":
        where = f"analyses.distances.pairs[{index}]"
        data = _as_mapping(data, where)
        _check_keys(data, ["name", "atoms", "mode", "label", "threshold"], where)
        threshold = data.get("threshold")
        return cls(
            name=str(_require(data, "name", where)),
            atoms=_atoms(data, where, 2),
            mode=_mode(data, where),
            label=data.get("label"),
            threshold=float(threshold) if threshold is not None else None,
        )


@dataclass
class DistancesConfig:
    enabled: bool = True
    pairs: list[DistanceDef] = field(default_factory=list)
    bins: int = 60
    kde: bool = True
    running_average: int = 0
    pbc: bool = True
    facet: bool = True
    normalization: str = "density"

    @classmethod
    def from_dict(cls, data: Any) -> "DistancesConfig":
        where = "analyses.distances"
        data = _as_mapping(data, where)
        _check_keys(
            data,
            ["enabled", "pairs", "bins", "kde", "running_average", "pbc",
             "facet", "normalization"],
            where,
        )
        pairs = [DistanceDef.from_dict(d, i)
                 for i, d in enumerate(_as_list(data.get("pairs"), f"{where}.pairs"))]
        return cls(
            enabled=bool(data.get("enabled", True)) and bool(pairs),
            pairs=pairs,
            bins=int(data.get("bins", 60)),
            kde=bool(data.get("kde", True)),
            running_average=int(data.get("running_average", 0)),
            pbc=bool(data.get("pbc", True)),
            facet=bool(data.get("facet", True)),
            normalization=_normalization(data.get("normalization"), where),
        )


@dataclass
class GeometryDef:
    """An angle (3 selections) or a dihedral (4 selections)."""

    name: str
    atoms: list[str]
    mode: str = "atom"
    label: str | None = None

    @classmethod
    def from_dict(cls, data: Any, index: int, n_atoms: int, where_root: str) -> "GeometryDef":
        where = f"{where_root}[{index}]"
        data = _as_mapping(data, where)
        _check_keys(data, ["name", "atoms", "mode", "label"], where)
        return cls(
            name=str(_require(data, "name", where)),
            atoms=_atoms(data, where, n_atoms),
            mode=_mode(data, where),
            label=data.get("label"),
        )


@dataclass
class AnglesConfig:
    """Configuration shared by the ``angles`` and ``dihedrals`` sections."""

    enabled: bool = True
    angles: list[GeometryDef] = field(default_factory=list)
    dihedrals: list[GeometryDef] = field(default_factory=list)
    bins: int = 60
    kde: bool = True
    pbc: bool = True
    facet: bool = True
    normalization: str = "density"

    @classmethod
    def from_dict(cls, angles_data: Any, dihedrals_data: Any) -> "AnglesConfig":
        angles_data = _as_mapping(angles_data, "analyses.angles")
        dihedrals_data = _as_mapping(dihedrals_data, "analyses.dihedrals")
        allowed = ["enabled", "definitions", "bins", "kde", "pbc", "facet",
                   "normalization"]
        _check_keys(angles_data, allowed, "analyses.angles")
        _check_keys(dihedrals_data, allowed, "analyses.dihedrals")

        angles: list[GeometryDef] = []
        if angles_data.get("enabled", True):
            angles = [
                GeometryDef.from_dict(d, i, 3, "analyses.angles.definitions")
                for i, d in enumerate(
                    _as_list(angles_data.get("definitions"), "analyses.angles.definitions")
                )
            ]
        dihedrals: list[GeometryDef] = []
        if dihedrals_data.get("enabled", True):
            dihedrals = [
                GeometryDef.from_dict(d, i, 4, "analyses.dihedrals.definitions")
                for i, d in enumerate(
                    _as_list(dihedrals_data.get("definitions"),
                             "analyses.dihedrals.definitions")
                )
            ]
        return cls(
            enabled=bool(angles or dihedrals),
            angles=angles,
            dihedrals=dihedrals,
            bins=int(angles_data.get("bins", dihedrals_data.get("bins", 60))),
            kde=bool(angles_data.get("kde", dihedrals_data.get("kde", True))),
            pbc=bool(angles_data.get("pbc", dihedrals_data.get("pbc", True))),
            facet=bool(angles_data.get("facet", dihedrals_data.get("facet", True))),
            normalization=_normalization(
                angles_data.get("normalization", dihedrals_data.get("normalization")),
                "analyses.angles",
            ),
        )


# --------------------------------------------------------------------------- #
# planarity
# --------------------------------------------------------------------------- #
@dataclass
class PlanarityConfig:
    """Centres whose planarity is followed.

    Each definition is four selections: **the centre first**, then its three
    substituents.  The reported value is the distance from the centre to the
    plane of the other three, zero for a planar (sp2) centre.
    """

    enabled: bool = False
    definitions: list[GeometryDef] = field(default_factory=list)
    bins: int = 60
    kde: bool = True
    pbc: bool = True
    facet: bool = True
    normalization: str = "density"

    @classmethod
    def from_dict(cls, data: Any) -> "PlanarityConfig":
        where = "analyses.planarity"
        data = _as_mapping(data, where)
        _check_keys(data, ["enabled", "definitions", "bins", "kde", "pbc",
                           "facet", "normalization"], where)
        definitions: list[GeometryDef] = []
        if data.get("enabled", True):
            definitions = [
                GeometryDef.from_dict(d, i, 4, f"{where}.definitions")
                for i, d in enumerate(
                    _as_list(data.get("definitions"), f"{where}.definitions"))
            ]
        return cls(
            enabled=bool(definitions),
            definitions=definitions,
            bins=int(data.get("bins", 60)),
            kde=bool(data.get("kde", True)),
            pbc=bool(data.get("pbc", True)),
            facet=bool(data.get("facet", True)),
            normalization=_normalization(data.get("normalization"), where),
        )


# --------------------------------------------------------------------------- #
# RMSD / RMSF
# --------------------------------------------------------------------------- #
@dataclass
class RMSDGroup:
    name: str
    selection: str
    superposition: str | None = None

    @classmethod
    def from_dict(cls, data: Any, index: int) -> "RMSDGroup":
        where = f"analyses.rmsd.groups[{index}]"
        data = _as_mapping(data, where)
        _check_keys(data, ["name", "selection", "superposition"], where)
        return cls(
            name=str(_require(data, "name", where)),
            selection=str(_require(data, "selection", where)),
            superposition=data.get("superposition"),
        )


@dataclass
class RMSDConfig:
    enabled: bool = False
    groups: list[RMSDGroup] = field(default_factory=list)
    reference: Path | None = None
    ref_frame: int = 0

    @classmethod
    def from_dict(cls, data: Any) -> "RMSDConfig":
        where = "analyses.rmsd"
        data = _as_mapping(data, where)
        _check_keys(data, ["enabled", "groups", "reference", "ref_frame"], where)
        groups = [RMSDGroup.from_dict(g, i)
                  for i, g in enumerate(_as_list(data.get("groups"), f"{where}.groups"))]
        ref = data.get("reference")
        return cls(
            enabled=bool(data.get("enabled", True)) and bool(groups),
            groups=groups,
            reference=Path(str(ref)) if ref else None,
            ref_frame=int(data.get("ref_frame", 0)),
        )


@dataclass
class RMSFConfig:
    enabled: bool = False
    selection: str = "protein and name CA"
    align: bool = True
    align_selection: str | None = None
    highlight: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Any) -> "RMSFConfig":
        where = "analyses.rmsf"
        data = _as_mapping(data, where)
        _check_keys(
            data, ["enabled", "selection", "align", "align_selection", "highlight"], where
        )
        return cls(
            enabled=bool(data.get("enabled", False)),
            selection=str(data.get("selection", "protein and name CA")),
            align=bool(data.get("align", True)),
            align_selection=data.get("align_selection"),
            highlight=[str(h) for h in _as_list(data.get("highlight"), f"{where}.highlight")],
        )


# --------------------------------------------------------------------------- #
# hydrogen bonds
# --------------------------------------------------------------------------- #
@dataclass
class HBondDef:
    """An explicitly tracked hydrogen bond (donor heavy atom / acceptor)."""

    name: str
    donor: str
    acceptor: str
    hydrogen: str | None = None
    label: str | None = None

    @classmethod
    def from_dict(cls, data: Any, index: int) -> "HBondDef":
        where = f"analyses.hbonds.pairs[{index}]"
        data = _as_mapping(data, where)
        _check_keys(data, ["name", "donor", "acceptor", "hydrogen", "label"], where)
        return cls(
            name=str(_require(data, "name", where)),
            donor=str(_require(data, "donor", where)),
            acceptor=str(_require(data, "acceptor", where)),
            hydrogen=data.get("hydrogen"),
            label=data.get("label"),
        )


@dataclass
class HBondsConfig:
    """Explicit (named) H-bonds and/or automatic detection in a region."""

    enabled: bool = False
    pairs: list[HBondDef] = field(default_factory=list)
    auto: bool = False
    auto_between: list[list[str]] = field(default_factory=list)
    donors_sel: str | None = None
    hydrogens_sel: str | None = None
    acceptors_sel: str | None = None
    d_a_cutoff: float = 3.5
    d_h_a_angle_cutoff: float = 150.0
    min_occupancy: float = 5.0
    max_reported: int = 25

    @classmethod
    def from_dict(cls, data: Any) -> "HBondsConfig":
        where = "analyses.hbonds"
        data = _as_mapping(data, where)
        _check_keys(
            data,
            [
                "enabled", "pairs", "auto", "auto_between", "donors_sel",
                "hydrogens_sel", "acceptors_sel", "d_a_cutoff",
                "d_h_a_angle_cutoff", "min_occupancy", "max_reported",
            ],
            where,
        )
        pairs = [HBondDef.from_dict(p, i)
                 for i, p in enumerate(_as_list(data.get("pairs"), f"{where}.pairs"))]
        between: list[list[str]] = []
        for i, entry in enumerate(_as_list(data.get("auto_between"), f"{where}.auto_between")):
            pair = _as_list(entry, f"{where}.auto_between[{i}]")
            if len(pair) != 2:
                raise ConfigError(
                    f"'{where}.auto_between[{i}]' must contain exactly 2 selections."
                )
            between.append([str(pair[0]), str(pair[1])])
        auto = bool(data.get("auto", False))
        return cls(
            enabled=bool(data.get("enabled", True)) and bool(pairs or auto),
            pairs=pairs,
            auto=auto,
            auto_between=between,
            donors_sel=data.get("donors_sel"),
            hydrogens_sel=data.get("hydrogens_sel"),
            acceptors_sel=data.get("acceptors_sel"),
            d_a_cutoff=float(data.get("d_a_cutoff", 3.5)),
            d_h_a_angle_cutoff=float(data.get("d_h_a_angle_cutoff", 150.0)),
            min_occupancy=float(data.get("min_occupancy", 5.0)),
            max_reported=int(data.get("max_reported", 25)),
        )


# --------------------------------------------------------------------------- #
# 2D maps
# --------------------------------------------------------------------------- #
@dataclass
class FreeEnergyMap:
    name: str
    x: str
    y: str
    method: str = "histogram"
    bins: int = 60
    free_energy: bool = True
    temperature: float = 300.0
    energy_unit: str = "kcal/mol"
    max_energy: float | None = None
    xlabel: str | None = None
    ylabel: str | None = None
    smooth: float = 0.0

    @classmethod
    def from_dict(cls, data: Any, index: int, defaults: Mapping[str, Any]) -> "FreeEnergyMap":
        where = f"analyses.free_energy_maps.maps[{index}]"
        data = _as_mapping(data, where)
        _check_keys(
            data,
            ["name", "x", "y", "method", "bins", "free_energy", "temperature",
             "energy_unit", "max_energy", "xlabel", "ylabel", "smooth"],
            where,
        )
        method = str(data.get("method", defaults.get("method", "histogram"))).lower()
        if method not in {"histogram", "kde"}:
            raise ConfigError(f"'{where}.method' must be 'histogram' or 'kde'.")
        energy_unit = str(data.get("energy_unit", defaults.get("energy_unit", "kcal/mol")))
        if energy_unit not in {"kcal/mol", "kJ/mol", "kT"}:
            raise ConfigError(
                f"'{where}.energy_unit' must be 'kcal/mol', 'kJ/mol' or 'kT'."
            )
        return cls(
            name=str(_require(data, "name", where)),
            x=str(_require(data, "x", where)),
            y=str(_require(data, "y", where)),
            method=method,
            bins=int(data.get("bins", defaults.get("bins", 60))),
            free_energy=bool(data.get("free_energy", defaults.get("free_energy", True))),
            temperature=float(data.get("temperature", defaults.get("temperature", 300.0))),
            energy_unit=energy_unit,
            max_energy=(float(data["max_energy"]) if data.get("max_energy") is not None
                        else defaults.get("max_energy")),
            xlabel=data.get("xlabel"),
            ylabel=data.get("ylabel"),
            smooth=float(data.get("smooth", defaults.get("smooth", 0.0))),
        )


@dataclass
class FreeEnergyConfig:
    enabled: bool = False
    maps: list[FreeEnergyMap] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Any) -> "FreeEnergyConfig":
        where = "analyses.free_energy_maps"
        data = _as_mapping(data, where)
        _check_keys(
            data,
            ["enabled", "maps", "method", "bins", "free_energy", "temperature",
             "energy_unit", "max_energy", "smooth"],
            where,
        )
        defaults = {k: data[k] for k in
                    ("method", "bins", "free_energy", "temperature", "energy_unit",
                     "max_energy", "smooth") if k in data}
        maps = [FreeEnergyMap.from_dict(m, i, defaults)
                for i, m in enumerate(_as_list(data.get("maps"), f"{where}.maps"))]
        return cls(enabled=bool(data.get("enabled", True)) and bool(maps), maps=maps)


# --------------------------------------------------------------------------- #
# Rg / RDF / clustering
# --------------------------------------------------------------------------- #
@dataclass
class NamedSelection:
    name: str
    selection: str

    @classmethod
    def from_dict(cls, data: Any, index: int, where_root: str) -> "NamedSelection":
        where = f"{where_root}[{index}]"
        data = _as_mapping(data, where)
        _check_keys(data, ["name", "selection"], where)
        return cls(
            name=str(_require(data, "name", where)),
            selection=str(_require(data, "selection", where)),
        )


@dataclass
class RgyrConfig:
    enabled: bool = False
    groups: list[NamedSelection] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Any) -> "RgyrConfig":
        where = "analyses.radius_of_gyration"
        data = _as_mapping(data, where)
        _check_keys(data, ["enabled", "groups"], where)
        groups = [NamedSelection.from_dict(g, i, f"{where}.groups")
                  for i, g in enumerate(_as_list(data.get("groups"), f"{where}.groups"))]
        return cls(enabled=bool(data.get("enabled", True)) and bool(groups), groups=groups)


@dataclass
class RDFPair:
    """One radial distribution function.

    ``center`` decides what the distances are measured from: ``atom`` (every
    atom of ``g1``, the standard definition), ``com`` (a single centre at the
    centre of mass of ``g1``) or ``residue`` (equivalently ``proximal`` /
    ``min`` / ``nearest``: the distance to the *closest* atom of ``g1``, which
    is what "solvent around this residue" means for a shape that is not a
    sphere).  ``shell_cutoff`` overrides the automatically detected first
    minimum used for the hydration-number time series.
    """

    name: str
    g1: str
    g2: str
    nbins: int = 75
    range: tuple[float, float] = (0.0, 15.0)
    exclude_same_residue: bool = False
    center: str = "atom"
    shell_cutoff: float | None = None

    @classmethod
    def from_dict(cls, data: Any, index: int) -> "RDFPair":
        where = f"analyses.rdf.pairs[{index}]"
        data = _as_mapping(data, where)
        _check_keys(
            data,
            ["name", "g1", "g2", "nbins", "range", "exclude_same_residue",
             "center", "shell_cutoff"],
            where,
        )
        rng = _as_list(data.get("range", [0.0, 15.0]), f"{where}.range")
        if len(rng) != 2:
            raise ConfigError(f"'{where}.range' must be [rmin, rmax].")
        center = str(data.get("center", "atom")).lower()
        if center not in {"atom", "com", "residue", "proximal", "min", "nearest"}:
            raise ConfigError(
                f"'{where}.center' must be 'atom', 'com' or 'residue' "
                f"('proximal'/'min'/'nearest' are accepted as synonyms of "
                f"'residue'; got '{center}')."
            )
        cutoff = data.get("shell_cutoff")
        return cls(
            name=str(_require(data, "name", where)),
            g1=str(_require(data, "g1", where)),
            g2=str(_require(data, "g2", where)),
            nbins=int(data.get("nbins", 75)),
            range=(float(rng[0]), float(rng[1])),
            exclude_same_residue=bool(data.get("exclude_same_residue", False)),
            center=center,
            shell_cutoff=float(cutoff) if cutoff is not None else None,
        )


@dataclass
class RDFConfig:
    enabled: bool = False
    pairs: list[RDFPair] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Any) -> "RDFConfig":
        where = "analyses.rdf"
        data = _as_mapping(data, where)
        _check_keys(data, ["enabled", "pairs"], where)
        pairs = [RDFPair.from_dict(p, i)
                 for i, p in enumerate(_as_list(data.get("pairs"), f"{where}.pairs"))]
        return cls(enabled=bool(data.get("enabled", True)) and bool(pairs), pairs=pairs)


@dataclass
class BridgePair:
    """Two groups that a solvent molecule may bridge.

    ``cutoff`` applies to both sides: a molecule counts as a bridge when it is
    within that distance of ``group_a`` *and* of ``group_b`` in the same frame.
    """

    name: str
    group_a: str
    group_b: str
    cutoff: float = 3.5
    solvent: str = "resname WAT HOH SOL TIP3 T3P and name O OW OH2"

    @classmethod
    def from_dict(cls, data: Any, index: int) -> "BridgePair":
        where = f"analyses.bridges.pairs[{index}]"
        data = _as_mapping(data, where)
        _check_keys(data, ["name", "group_a", "group_b", "cutoff", "solvent"], where)
        return cls(
            name=str(_require(data, "name", where)),
            group_a=str(_require(data, "group_a", where)),
            group_b=str(_require(data, "group_b", where)),
            cutoff=float(data.get("cutoff", 3.5)),
            solvent=str(data.get(
                "solvent", "resname WAT HOH SOL TIP3 T3P and name O OW OH2")),
        )


@dataclass
class BridgeConfig:
    enabled: bool = False
    pairs: list[BridgePair] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Any) -> "BridgeConfig":
        where = "analyses.bridges"
        data = _as_mapping(data, where)
        _check_keys(data, ["enabled", "pairs"], where)
        pairs = [BridgePair.from_dict(p, i)
                 for i, p in enumerate(_as_list(data.get("pairs"), f"{where}.pairs"))]
        return cls(enabled=bool(data.get("enabled", True)) and bool(pairs), pairs=pairs)


@dataclass
class ClusteringConfig:
    enabled: bool = False
    selection: str = "protein and name CA"
    method: str = "hierarchical"
    n_clusters: int = 3
    linkage: str = "average"
    max_frames: int = 2000
    align: bool = True
    write_structures: bool = True
    write_selection: str | None = None

    @classmethod
    def from_dict(cls, data: Any) -> "ClusteringConfig":
        where = "analyses.clustering"
        data = _as_mapping(data, where)
        _check_keys(
            data,
            ["enabled", "selection", "method", "n_clusters", "linkage",
             "max_frames", "align", "write_structures", "write_selection"],
            where,
        )
        method = str(data.get("method", "hierarchical")).lower()
        if method not in {"hierarchical", "kmeans"}:
            raise ConfigError(f"'{where}.method' must be 'hierarchical' or 'kmeans'.")
        linkage = str(data.get("linkage", "average")).lower()
        if linkage not in {"average", "complete", "single", "ward"}:
            raise ConfigError(
                f"'{where}.linkage' must be 'average', 'complete', 'single' or 'ward'."
            )
        return cls(
            enabled=bool(data.get("enabled", False)),
            selection=str(data.get("selection", "protein and name CA")),
            method=method,
            n_clusters=int(data.get("n_clusters", 3)),
            linkage=linkage,
            max_frames=int(data.get("max_frames", 2000)),
            align=bool(data.get("align", True)),
            write_structures=bool(data.get("write_structures", True)),
            write_selection=data.get("write_selection"),
        )


@dataclass
class ReportConfig:
    enabled: bool = True
    formats: list[str] = field(default_factory=lambda: ["markdown", "html"])
    title: str = "MD interaction analysis"

    @classmethod
    def from_dict(cls, data: Any) -> "ReportConfig":
        where = "report"
        data = _as_mapping(data, where)
        _check_keys(data, ["enabled", "formats", "title"], where)
        formats = [str(f).lower() for f in
                   _as_list(data.get("formats", ["markdown", "html"]), f"{where}.formats")]
        bad = sorted(set(formats) - {"markdown", "md", "html"})
        if bad:
            raise ConfigError(f"'{where}.formats' unsupported: {bad}.")
        return cls(
            enabled=bool(data.get("enabled", True)),
            formats=["markdown" if f == "md" else f for f in formats],
            title=str(data.get("title", "MD interaction analysis")),
        )


# --------------------------------------------------------------------------- #
# top level
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    """Full, validated configuration."""

    system: SystemConfig | None
    data: "DataConfig | None" = None
    output: OutputConfig = field(default_factory=OutputConfig)
    selections: dict[str, str] = field(default_factory=dict)
    distances: DistancesConfig = field(default_factory=DistancesConfig)
    angles: AnglesConfig = field(default_factory=AnglesConfig)
    planarity: PlanarityConfig = field(default_factory=PlanarityConfig)
    rmsd: RMSDConfig = field(default_factory=RMSDConfig)
    rmsf: RMSFConfig = field(default_factory=RMSFConfig)
    hbonds: HBondsConfig = field(default_factory=HBondsConfig)
    free_energy: FreeEnergyConfig = field(default_factory=FreeEnergyConfig)
    rgyr: RgyrConfig = field(default_factory=RgyrConfig)
    rdf: RDFConfig = field(default_factory=RDFConfig)
    bridges: BridgeConfig = field(default_factory=BridgeConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    source: Path | None = None

    # -- construction ------------------------------------------------------ #
    @classmethod
    def from_dict(cls, data: Mapping[str, Any], source: Path | None = None) -> "Config":
        data = _as_mapping(data, "<root>")
        _check_keys(data, ["system", "data", "output", "selections", "analyses",
                           "report"], "<root>")
        if not data.get("system") and not data.get("data"):
            raise ConfigError(
                "The configuration needs a 'system:' section (topology + "
                "trajectory) or a 'data:' section (pre-computed tables)."
            )

        selections_raw = _as_mapping(data.get("selections"), "selections")
        selections = {str(k): str(v) for k, v in selections_raw.items()}

        analyses = _as_mapping(data.get("analyses"), "analyses")
        _check_keys(
            analyses,
            ["distances", "angles", "dihedrals", "planarity", "rmsd", "rmsf",
             "hbonds", "free_energy_maps", "radius_of_gyration", "rdf",
             "bridges", "clustering"],
            "analyses",
        )

        cfg = cls(
            system=(SystemConfig.from_dict(data["system"]) if data.get("system")
                    else None),
            data=DataConfig.from_dict(data.get("data")),
            output=OutputConfig.from_dict(data.get("output")),
            selections=selections,
            distances=DistancesConfig.from_dict(analyses.get("distances")),
            angles=AnglesConfig.from_dict(analyses.get("angles"),
                                          analyses.get("dihedrals")),
            planarity=PlanarityConfig.from_dict(analyses.get("planarity")),
            rmsd=RMSDConfig.from_dict(analyses.get("rmsd")),
            rmsf=RMSFConfig.from_dict(analyses.get("rmsf")),
            hbonds=HBondsConfig.from_dict(analyses.get("hbonds")),
            free_energy=FreeEnergyConfig.from_dict(analyses.get("free_energy_maps")),
            rgyr=RgyrConfig.from_dict(analyses.get("radius_of_gyration")),
            rdf=RDFConfig.from_dict(analyses.get("rdf")),
            bridges=BridgeConfig.from_dict(analyses.get("bridges")),
            clustering=ClusteringConfig.from_dict(analyses.get("clustering")),
            report=ReportConfig.from_dict(data.get("report")),
            source=source,
        )
        cfg._validate_names()
        return cfg

    # -- helpers ----------------------------------------------------------- #
    def _validate_names(self) -> None:
        """Check that observable names are unique (they become column names)."""
        seen: dict[str, str] = {}
        groups = [
            ("distance", [d.name for d in self.distances.pairs]),
            ("angle", [a.name for a in self.angles.angles]),
            ("dihedral", [d.name for d in self.angles.dihedrals]),
            ("planarity", [p.name for p in self.planarity.definitions]),
            ("hbond", [h.name for h in self.hbonds.pairs]),
            ("rmsd", [g.name for g in self.rmsd.groups]),
            ("rgyr", [g.name for g in self.rgyr.groups]),
            ("rdf", [p.name for p in self.rdf.pairs]),
            ("bridge", [p.name for p in self.bridges.pairs]),
            ("free-energy map", [m.name for m in self.free_energy.maps]),
        ]
        for kind, names in groups:
            for name in names:
                if name in seen:
                    raise ConfigError(
                        f"Duplicated observable name '{name}' "
                        f"({seen[name]} and {kind}). Names must be unique."
                    )
                seen[name] = kind

    def resolve_selection(self, token: str) -> str:
        """Translate a selection alias into an MDAnalysis selection string.

        A token that matches a key of the ``selections:`` block is replaced by
        its value; anything else is passed through untouched, so raw selection
        strings can be used inline.
        """
        return self.selections.get(token, token)

    @property
    def data_only(self) -> bool:
        """True when the run works on pre-computed tables, with no trajectory."""
        return self.system is None and self.data is not None

    def enabled_analyses(self) -> list[str]:
        """Names of the analyses that will actually run."""
        if self.data_only:
            names = ["distributions"]
            if self.data and self.data.replicas:
                names.append("replicas")
            if self.free_energy.enabled:
                names.append("free_energy_maps")
            return names
        flags = {
            "distances": self.distances.enabled,
            "angles_dihedrals": self.angles.enabled,
            "planarity": self.planarity.enabled,
            "rmsd": self.rmsd.enabled,
            "rmsf": self.rmsf.enabled,
            "hbonds": self.hbonds.enabled,
            "free_energy_maps": self.free_energy.enabled,
            "radius_of_gyration": self.rgyr.enabled,
            "rdf": self.rdf.enabled,
            "bridges": self.bridges.enabled,
            "clustering": self.clustering.enabled,
        }
        return [name for name, on in flags.items() if on]


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> Config:
    """Read and validate a YAML configuration file.

    Parameters
    ----------
    path
        Path to the YAML file.
    overrides
        Optional shallow overrides for the ``system`` and ``output`` sections
        (used by the CLI flags ``--top``, ``--traj``, ``--out``, ``--stride``).

    Notes
    -----
    Relative paths inside the YAML are interpreted relative to the location of
    the YAML file itself, so a config can be moved around with its data.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if raw is None:
        raise ConfigError(f"Configuration file is empty: {path}")
    if not isinstance(raw, Mapping):
        raise ConfigError(f"Configuration file must contain a mapping: {path}")

    raw = dict(raw)
    if overrides:
        raw = _apply_overrides(raw, overrides)

    cfg = Config.from_dict(raw, source=path)
    cfg = _resolve_paths(cfg, path.parent)
    return cfg


def _apply_overrides(raw: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(_as_mapping(raw.get("output"), "output"))
    if overrides.get("directory"):
        output["directory"] = str(overrides["directory"])
    if overrides.get("formats"):
        output["formats"] = list(overrides["formats"])
    if output:
        raw["output"] = output

    # frame range/stride goes to whichever input section the config uses
    section_name = "system"
    if not raw.get("system") and raw.get("data") and not (
        overrides.get("topology") or overrides.get("trajectory")
    ):
        section_name = "data"
    section = dict(_as_mapping(raw.get(section_name), section_name))
    frames = dict(_as_mapping(section.get("frames"), f"{section_name}.frames"))

    if overrides.get("topology"):
        section["topology"] = str(overrides["topology"])
    if overrides.get("trajectory"):
        section["trajectory"] = [str(p) for p in overrides["trajectory"]]
    if overrides.get("stride"):
        frames["stride"] = int(overrides["stride"])
    if overrides.get("start") is not None:
        frames["start"] = int(overrides["start"])
    if overrides.get("stop") is not None:
        frames["stop"] = int(overrides["stop"])
    if frames:
        section["frames"] = frames
    if section:
        raw[section_name] = section
    return raw


def _resolve_paths(cfg: Config, base: Path) -> Config:
    """Make every path in the config absolute, relative to ``base``."""

    def _resolve(p: Path) -> Path:
        return p if p.is_absolute() else (base / p).resolve()

    if cfg.system is not None:
        cfg.system.topology = _resolve(cfg.system.topology)
        cfg.system.trajectory = [_resolve(p) for p in cfg.system.trajectory]
        cfg.system.replicas = {name: [_resolve(p) for p in paths]
                               for name, paths in cfg.system.replicas.items()}
        if cfg.system.align.reference is not None:
            cfg.system.align.reference = _resolve(cfg.system.align.reference)
    if cfg.data is not None:
        cfg.data.files = [_resolve(p) for p in cfg.data.files]
        cfg.data.replicas = {name: [_resolve(p) for p in paths]
                             for name, paths in cfg.data.replicas.items()}
    if cfg.rmsd.reference is not None:
        cfg.rmsd.reference = _resolve(cfg.rmsd.reference)
    cfg.output.directory = _resolve(cfg.output.directory)
    return cfg
