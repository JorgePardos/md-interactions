"""md_interactions - reusable interaction/geometry analysis of MD trajectories.

Designed for enzyme-substrate systems: catalytic distances, attack angles,
hydrogen bonds, RMSD/RMSF, apparent free-energy maps, radius of gyration, RDF
and conformational clustering, all driven by a single YAML file and all
producing publication-ready figures plus CSV tables.

Typical use from a notebook::

    import md_interactions as mdi

    config = mdi.load_config("config.yaml")
    out = mdi.run_analyses(config)
    out.summary                     # pandas DataFrame with mean +/- std
    out.result("distances").tables["distances"]

Or one analysis at a time::

    system = mdi.load_system(config)
    df = mdi.analyses.distances.compute_distances(system)

From the shell::

    md-analyzer init -o config.yaml
    md-analyzer check -c config.yaml
    md-analyzer run   -c config.yaml
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import analyses  # noqa: F401  (namespace convenience)
from .config import Config, load_config
from .exceptions import (
    AtomCountError,
    ConfigError,
    EmptySelectionError,
    MDInteractionsError,
    MissingTopologyInfoError,
    SelectionError,
)
from .inputfile import load_input, parse_input, resolve_atom_spec
from .io_utils import OutputPaths, prepare_output
from .plotting import PALETTE, apply_style
from .results import AnalysisResult
from .runner import RunOutput, run_analyses, run_from_config
from .system import MDSystem, load_system
from .tabular import Dataset, load_dataset, read_table
from .templates import EXAMPLE_CONFIG

__all__ = [
    "__version__",
    "AnalysisResult",
    "AtomCountError",
    "Config",
    "ConfigError",
    "Dataset",
    "EmptySelectionError",
    "EXAMPLE_CONFIG",
    "MDInteractionsError",
    "MDSystem",
    "MissingTopologyInfoError",
    "OutputPaths",
    "PALETTE",
    "RunOutput",
    "SelectionError",
    "analyses",
    "apply_style",
    "load_config",
    "load_dataset",
    "load_input",
    "load_system",
    "parse_input",
    "read_table",
    "resolve_atom_spec",
    "prepare_output",
    "run_analyses",
    "run_from_config",
]
