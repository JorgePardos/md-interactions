"""Pre-flight validation of a configuration against the real topology.

Every check here answers the same question: *would this run fail after doing
the expensive work?*  A misspelled residue number or a 2D map pointing at an
observable that is never computed costs nothing to detect up front and a whole
trajectory pass to discover at the end.

The same code backs ``md-analyzer check`` and the automatic check that ``run``
performs before touching the trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config
from .exceptions import MDInteractionsError

__all__ = [
    "Issue",
    "SelectionReport",
    "selection_tokens",
    "expected_observables",
    "validate",
]


@dataclass
class Issue:
    """Something that would make the run fail or produce nothing."""

    context: str
    detail: str
    selection: str = ""


@dataclass
class SelectionReport:
    """Result of a pre-flight check."""

    resolved: list[tuple[str, str, int]] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues


# --------------------------------------------------------------------------- #
# what the configuration refers to
# --------------------------------------------------------------------------- #
def selection_tokens(config: Config) -> list[tuple[str, str]]:
    """Every (context, selection token) pair the configuration references."""
    tokens: list[tuple[str, str]] = []
    for name in config.selections:
        tokens.append((f"selections.{name}", name))
    if config.system is not None and config.system.align.enabled:
        tokens.append(("system.align", config.system.align.selection))
    for defn in config.distances.pairs:
        tokens += [(f"distance:{defn.name}", token) for token in defn.atoms]
    for defn in config.angles.angles:
        tokens += [(f"angle:{defn.name}", token) for token in defn.atoms]
    for defn in config.angles.dihedrals:
        tokens += [(f"dihedral:{defn.name}", token) for token in defn.atoms]
    for group in config.rmsd.groups:
        tokens.append((f"rmsd:{group.name}", group.selection))
        if group.superposition:
            tokens.append((f"rmsd:{group.name}.fit", group.superposition))
    if config.rmsf.enabled:
        tokens.append(("rmsf", config.rmsf.selection))
        if config.rmsf.align_selection:
            tokens.append(("rmsf.fit", config.rmsf.align_selection))
    for hbond in config.hbonds.pairs:
        tokens.append((f"hbond:{hbond.name}.donor", hbond.donor))
        tokens.append((f"hbond:{hbond.name}.acceptor", hbond.acceptor))
        if hbond.hydrogen:
            tokens.append((f"hbond:{hbond.name}.hydrogen", hbond.hydrogen))
    for index, (first, second) in enumerate(config.hbonds.auto_between):
        tokens.append((f"hbonds.auto_between[{index}].0", first))
        tokens.append((f"hbonds.auto_between[{index}].1", second))
    for group in config.rgyr.groups:
        tokens.append((f"rgyr:{group.name}", group.selection))
    for pair in config.rdf.pairs:
        tokens.append((f"rdf:{pair.name}.g1", pair.g1))
        tokens.append((f"rdf:{pair.name}.g2", pair.g2))
    for pair in config.bridges.pairs:
        tokens.append((f"bridge:{pair.name}.a", pair.group_a))
        tokens.append((f"bridge:{pair.name}.b", pair.group_b))
        tokens.append((f"bridge:{pair.name}.solvent", pair.solvent))
    if config.clustering.enabled:
        tokens.append(("clustering", config.clustering.selection))
        if config.clustering.write_selection:
            tokens.append(("clustering.write_selection",
                           config.clustering.write_selection))

    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str]] = []
    for entry in tokens:
        if entry not in seen:
            seen.add(entry)
            unique.append(entry)
    return unique


def expected_observables(config: Config) -> set[str]:
    """Names of the per-frame observables the run will produce.

    Used to catch a 2D map that points at something no analysis computes —
    a mistake that otherwise only surfaces after every trajectory pass.
    """
    names: set[str] = set()
    if config.data_only and config.data is not None:
        labels = config.data.labels
        names.update(labels.get(column, column) for column in config.data.columns)
        if not config.data.columns:
            names.add("*")                     # columns are known only at load time
        return names

    names.update(defn.name for defn in config.distances.pairs)
    names.update(defn.name for defn in config.angles.angles)
    names.update(defn.name for defn in config.angles.dihedrals)
    for hbond in config.hbonds.pairs:
        names.add(hbond.name)
        names.add(f"{hbond.name}_angle")
    names.update(group.name for group in config.rmsd.groups)
    names.update(group.name for group in config.rgyr.groups)
    for pair in config.rdf.pairs:
        names.add(f"n_{pair.name}")
    for pair in config.bridges.pairs:
        names.add(f"nbridge_{pair.name}")
    return names


# --------------------------------------------------------------------------- #
# the check itself
# --------------------------------------------------------------------------- #
def validate(config: Config, system=None, dataset=None) -> SelectionReport:
    """Resolve every selection and cross-check the 2D maps.

    ``system`` (or ``dataset`` in ``data:`` mode) is reused when given, so the
    topology is not read twice when ``run`` performs this check.
    """
    report = SelectionReport()

    if config.data_only:
        _validate_dataset(config, dataset, report)
        return report

    # A failing selection is usually referenced from several places (its alias
    # plus every observable using it). Report each broken selection once, with
    # the list of places that use it, instead of repeating the same message.
    failures: dict[str, list[str]] = {}
    details: dict[str, str] = {}

    for context, token in selection_tokens(config):
        selection = config.resolve_selection(token)
        try:
            group = system.select(token, name=context)
        except MDInteractionsError as exc:
            failures.setdefault(selection, []).append(context)
            details.setdefault(selection, _first_line(exc))
            continue
        report.resolved.append((context, token, group.n_atoms))

    for selection, contexts in failures.items():
        used_by = [c for c in contexts if not c.startswith("selections.")]
        context = used_by[0] if used_by else contexts[0]
        extra = ""
        if len(used_by) > 1:
            extra = f" (also used by {', '.join(used_by[1:])})"
        report.issues.append(Issue(context=context + extra, selection=selection,
                                   detail=details[selection]))

    _validate_maps(config, expected_observables(config), report)
    return report


def _first_line(exc: Exception) -> str:
    """Headline of an error, without the selection already shown next to it."""
    import re

    headline = str(exc).splitlines()[0]
    return re.sub(r"^Selection '[^']*' -> \"[^\"]*\"\s*", "", headline).capitalize()


def _validate_dataset(config: Config, dataset, report: SelectionReport) -> None:
    """In ``data:`` mode the columns of the table play the role of selections."""
    if dataset is None:
        return
    for column in dataset.columns:
        report.resolved.append((f"column:{column}", column, dataset.n_frames))
    _validate_maps(config, set(dataset.columns), report)


def _validate_maps(config: Config, available: set[str],
                   report: SelectionReport) -> None:
    if "*" in available:
        return
    for spec in config.free_energy.maps:
        for axis, name in (("x", spec.x), ("y", spec.y)):
            if name not in available:
                closest = _suggest(name, available)
                report.issues.append(Issue(
                    context=f"free_energy_maps:{spec.name}.{axis}",
                    selection=name,
                    detail=(f"'{name}' is not produced by any analysis"
                            + (f"; did you mean '{closest}'?" if closest else "")),
                ))


def _suggest(name: str, options: set[str]) -> str | None:
    """Closest available name, to turn a typo into a usable hint."""
    import difflib

    matches = difflib.get_close_matches(name, sorted(options), n=1, cutoff=0.6)
    return matches[0] if matches else None
