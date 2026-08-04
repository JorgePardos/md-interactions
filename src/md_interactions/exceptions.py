"""Custom exceptions for :mod:`md_interactions`.

All errors raised on purpose by this package derive from
:class:`MDInteractionsError`, so user code (and the CLI) can catch them and
show a clean message instead of a raw MDAnalysis traceback.
"""

from __future__ import annotations

__all__ = [
    "MDInteractionsError",
    "ConfigError",
    "SelectionError",
    "EmptySelectionError",
    "AtomCountError",
    "MissingTopologyInfoError",
    "AnalysisError",
]


class MDInteractionsError(Exception):
    """Base class for every error raised by this package."""


class ConfigError(MDInteractionsError):
    """The YAML configuration is malformed, incomplete or inconsistent."""


class SelectionError(MDInteractionsError):
    """An MDAnalysis selection string could not be parsed."""


class EmptySelectionError(SelectionError):
    """A selection is syntactically valid but matched zero atoms."""

    def __init__(self, name: str, selection: str) -> None:
        super().__init__(
            f"Selection '{name}' -> \"{selection}\" matched 0 atoms.\n"
            "  Check residue numbering (MDAnalysis 'resid' follows the topology, "
            "which for AMBER prmtop files is 1-based and continuous across chains), "
            "atom names (e.g. 'OG' vs 'OG1') and residue names (e.g. 'HIE'/'HID' "
            "instead of 'HIS' in AMBER topologies)."
        )
        self.name = name
        self.selection = selection


class AtomCountError(SelectionError):
    """A selection matched a number of atoms incompatible with the analysis."""

    def __init__(self, name: str, selection: str, n_found: int, expected: str) -> None:
        super().__init__(
            f"Selection '{name}' -> \"{selection}\" matched {n_found} atoms, "
            f"but {expected} was expected.\n"
            "  Either refine the selection, or set mode: com / mode: min for this "
            "observable so that groups of atoms are allowed."
        )
        self.name = name
        self.selection = selection
        self.n_found = n_found


class MissingTopologyInfoError(MDInteractionsError):
    """The topology lacks an attribute required by a given analysis.

    Typical case: hydrogen-bond analysis on a topology without explicit
    hydrogens, or without charges/types needed for automatic donor detection.
    """


class AnalysisError(MDInteractionsError):
    """An analysis module failed for a reason other than a bad selection."""
