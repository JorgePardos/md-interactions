"""A compact input file, as an alternative to writing YAML by hand.

The YAML configuration is expressive but verbose, and indentation errors are
easy to make.  This module reads a flat, cpptraj-flavoured input file instead::

    topology    system.prmtop
    trajectory  prod.nc
    time        0.1 ps

    [distances]
    d_nuc    ASP20:OD1   TRH453:C1   threshold=3.5
    d_wat    TRH453:O2P  water       mode=min

    [angles]
    a_attack ASP20:OD1   TRH453:C1   TRH453:O1

Atoms can be written in whichever notation is at hand:

============================ =================================================
``ASP20:OD1``                residue name + number, atom name
``20:OD1``                   residue number and atom name
``TRH:O2P``                  by residue name (any residue so called)
``:20@OD1``                  cpptraj mask
``@1123``                    atom number, 1-based as in cpptraj/VMD
``ASP20:OD1,OD2``            several atoms of one residue
``water`` / ``protein`` ...  keywords
``{resid 20 and name OD1}``  a literal MDAnalysis selection
============================ =================================================

The result is an ordinary :class:`~md_interactions.config.Config`, so both
input styles share one code path and one set of validations.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .config import Config
from .exceptions import ConfigError

__all__ = ["parse_input", "input_to_dict", "load_input", "resolve_atom_spec"]

#: Sections understood in the input file.
_SECTIONS = {
    "selections", "distances", "angles", "dihedrals", "hbonds", "rmsd", "rmsf",
    "rdf", "bridges", "hydration", "rgyr", "clustering", "fes", "options",
}

#: Shorthand that expands to a full MDAnalysis selection.
_KEYWORDS = {
    "water": "resname WAT HOH SOL TIP3 T3P and name O OW OH2",
    "waters": "resname WAT HOH SOL TIP3 T3P and name O OW OH2",
    "water_all": "resname WAT HOH SOL TIP3 T3P",
    "protein": "protein",
    "backbone": "backbone",
    "heavy": "not name H*",
    "ligand": "not protein and not resname WAT HOH SOL TIP3 T3P NA CL K MG ZN CA",
}

_RESID_ATOM = re.compile(r"^(?P<resname>[A-Za-z][A-Za-z0-9]*?)?(?P<resid>\d+):(?P<atoms>[\w',*]+(?:,[\w',*]+)*)$")
_RESNAME_ATOM = re.compile(r"^(?P<resname>[A-Za-z][A-Za-z0-9]{0,4}):(?P<atoms>[\w',*]+(?:,[\w',*]+)*)$")
_CPPTRAJ = re.compile(r"^:(?P<res>[\w,\-]+)?(?:@(?P<atoms>[\w',*]+(?:,[\w',*]+)*))?$")
_ATOM_NUMBER = re.compile(r"^@(?P<numbers>\d+(?:,\d+)*)$")
_BARE_ATOM = re.compile(r"^@(?P<atoms>[A-Za-z][\w',*]*(?:,[\w',*]+)*)$")


# --------------------------------------------------------------------------- #
# atom specifications
# --------------------------------------------------------------------------- #
def resolve_atom_spec(spec: str, aliases: dict[str, str] | None = None) -> tuple[str, str]:
    """Translate one atom specification into ``(alias, MDAnalysis selection)``.

    Purely textual: no topology is needed here, so the parser stays fast and
    testable.  ``md-analyzer check`` is what confirms the selection matches
    something in a given topology.
    """
    aliases = aliases or {}
    spec = spec.strip()
    if not spec:
        raise ConfigError("Empty atom specification.")

    if spec in aliases:
        return _alias(spec), aliases[spec]

    # {literal MDAnalysis selection} or "quoted selection"
    if spec.startswith("{") and spec.endswith("}"):
        return _alias(spec[1:-1]), spec[1:-1].strip()
    if len(spec) > 1 and spec[0] == spec[-1] and spec[0] in "\"'":
        return _alias(spec[1:-1]), spec[1:-1].strip()

    lowered = spec.lower()
    if lowered in _KEYWORDS:
        return lowered, _KEYWORDS[lowered]

    match = _ATOM_NUMBER.match(spec)
    if match:                                   # @1123 -> 1-based atom number
        numbers = [int(n) for n in match.group("numbers").split(",")]
        if any(n < 1 for n in numbers):
            raise ConfigError(f"Atom numbers are 1-based, got '{spec}'.")
        indices = " ".join(str(n - 1) for n in numbers)
        return f"at{'_'.join(str(n) for n in numbers)}", f"index {indices}"

    match = _CPPTRAJ.match(spec)                # :20@OD1, :20, :TRH@O2P
    if match and (match.group("res") or match.group("atoms")):
        return _from_parts(match.group("res"), match.group("atoms"))

    match = _BARE_ATOM.match(spec)              # @CA (atom name, any residue)
    if match:
        atoms = match.group("atoms").replace(",", " ")
        return f"at_{match.group('atoms').replace(',', '_')}", f"name {atoms}"

    match = _RESID_ATOM.match(spec)             # ASP20:OD1 or 20:OD1
    if match:
        return _from_parts(
            (match.group("resname") or "") + match.group("resid"),
            match.group("atoms"),
        )

    match = _RESNAME_ATOM.match(spec)           # TRH:O2P
    if match:
        return _from_parts(match.group("resname"), match.group("atoms"))

    # anything else is taken as a literal MDAnalysis selection
    return _alias(spec), spec


def _looks_like_atom_spec(token: str, aliases: dict[str, str] | None = None) -> bool:
    """Whether a bare token can be read as an atom, rather than as a name."""
    token = token.strip()
    if aliases and token in aliases:
        return True
    if token.lower() in _KEYWORDS:
        return True
    if token[:1] in {"@", ":", "{", "\"", "'"}:
        return True
    return ":" in token or " " in token


def _from_parts(residue: str | None, atoms: str | None) -> tuple[str, str]:
    """Build a selection from the residue part and the atom part of a spec."""
    clauses: list[str] = []
    alias_bits: list[str] = []

    if residue:
        residue = residue.strip()
        name = "".join(c for c in residue if c.isalpha())
        number = "".join(c for c in residue if c.isdigit() or c in ",-")
        if name and number:
            clauses.append(f"resname {name.upper()}")
            clauses.append(_resid_clause(number))
        elif number:
            clauses.append(_resid_clause(number))
        elif name:
            clauses.append(f"resname {name.upper()}")
        alias_bits.append(residue.upper().replace(",", "_").replace("-", "to"))

    if atoms and atoms.strip() != "*":          # '*' means "the whole residue"
        names = atoms.replace(",", " ").split()
        clauses.append("name " + " ".join(names))
        alias_bits.append("_".join(names))

    if not clauses:
        raise ConfigError("An atom specification needs a residue or an atom name.")
    return _alias("_".join(alias_bits)), " and ".join(clauses)


def _resid_clause(numbers: str) -> str:
    """``20`` -> ``resid 20``; ``20,25`` -> ``resid 20 25``; ``20-30`` -> range."""
    parts = []
    for chunk in numbers.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            start, _sep, stop = chunk.partition("-")
            parts.append(f"{start.strip()}:{stop.strip()}")
        elif chunk:
            parts.append(chunk)
    return "resid " + " ".join(parts)


def _alias(text: str) -> str:
    keep = [c if (c.isalnum() or c == "_") else "_" for c in str(text)]
    alias = "".join(keep).strip("_")
    while "__" in alias:
        alias = alias.replace("__", "_")
    return alias or "sel"


# --------------------------------------------------------------------------- #
# tokenising
# --------------------------------------------------------------------------- #
def _split_tokens(line: str) -> tuple[list[str], dict[str, str]]:
    """Split a line into positional tokens and ``key=value`` options.

    Braces and quotes keep their contents together, so
    ``d1 {resid 20 and name OD1} @55 mode=min`` tokenises correctly.
    """
    tokens: list[str] = []
    options: dict[str, str] = {}
    current = ""
    depth = 0
    quote = ""

    def flush() -> None:
        nonlocal current
        if current:
            tokens.append(current)
            current = ""

    for char in line:
        if quote:
            current += char
            if char == quote:
                quote = ""
            continue
        if char in "\"'":
            quote = char
            current += char
            continue
        if char == "{":
            depth += 1
            current += char
            continue
        if char == "}":
            depth = max(0, depth - 1)
            current += char
            continue
        if char.isspace() and depth == 0:
            flush()
            continue
        current += char
    flush()
    if depth or quote:
        raise ConfigError(f"Unbalanced braces or quotes in: {line!r}")

    positional = []
    for token in tokens:
        if "=" in token and not token.startswith(("{", "\"", "'")):
            key, _sep, value = token.partition("=")
            if key and not key.startswith("@") and ":" not in key:
                options[key.strip().lower()] = value.strip().strip("\"'")
                continue
        positional.append(token)
    return positional, options


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def parse_input(path: str | Path) -> dict[str, Any]:
    """Read an input file into ``{"global": {...}, "sections": {name: [lines]}}``."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Input file not found: {path}")

    globals_: dict[str, list[str]] = {}
    sections: dict[str, list[tuple[list[str], dict[str, str], int]]] = {}
    current: str | None = None

    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].split(";", 1)[0].strip()
        if not line:
            continue

        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip().lower()
            if current not in _SECTIONS:
                raise ConfigError(
                    f"Unknown section '[{current}]' on line {number}. "
                    f"Known sections: {', '.join('[' + s + ']' for s in sorted(_SECTIONS))}"
                )
            sections.setdefault(current, [])
            continue

        tokens, options = _split_tokens(line)
        if current is None:
            if not tokens:
                raise ConfigError(f"Cannot parse line {number}: {raw!r}")
            key = tokens[0].lower()
            globals_.setdefault(key, []).extend(tokens[1:])
            for option_key, value in options.items():
                globals_.setdefault(option_key, []).append(value)
        else:
            sections[current].append((tokens, options, number))

    return {"global": globals_, "sections": sections, "source": path}


def input_to_dict(parsed: dict[str, Any]) -> dict[str, Any]:
    """Turn a parsed input file into the configuration dictionary."""
    globals_ = parsed["global"]
    sections = parsed["sections"]
    selections: dict[str, str] = {}

    # ---- user aliases first, so later sections can use them ---- #
    aliases: dict[str, str] = {}
    for tokens, _options, number in sections.get("selections", []):
        if len(tokens) < 2:
            raise ConfigError(f"Line {number}: a selection needs a name and a value.")
        name = tokens[0]
        _alias_name, selection = resolve_atom_spec(" ".join(tokens[1:]), aliases)
        aliases[name] = selection
        selections[name] = selection

    def resolve(spec: str) -> str:
        """Register the alias for ``spec`` and return the alias name."""
        if spec in aliases:
            return spec
        alias, selection = resolve_atom_spec(spec, aliases)
        existing = selections.get(alias)
        if existing and existing != selection:
            alias = f"{alias}_{len(selections)}"
        selections[alias] = selection
        return alias

    config: dict[str, Any] = {"system": _system(globals_), "analyses": {}}
    output = _output(globals_)
    if output:
        config["output"] = output

    analyses = config["analyses"]
    _geometry(sections, "distances", 2, analyses, resolve)
    _geometry(sections, "angles", 3, analyses, resolve)
    _geometry(sections, "dihedrals", 4, analyses, resolve)
    _hbonds(sections, analyses, resolve)
    _rmsd(sections, globals_, analyses, resolve)
    _rmsf(sections, globals_, analyses)
    _rdf(sections, analyses, resolve)
    _bridges(sections, analyses, resolve)
    _rgyr(sections, analyses, resolve)
    _clustering(sections, analyses, resolve)
    _fes(sections, analyses)

    if selections:
        config["selections"] = selections
    config["report"] = {"enabled": _flag(globals_, "report", True),
                        "formats": ["markdown", "html"]}
    return config


def load_input(path: str | Path) -> Config:
    """Read an input file and return a validated configuration."""
    parsed = parse_input(path)
    config = Config.from_dict(input_to_dict(parsed), source=Path(path))
    from .config import _resolve_paths

    return _resolve_paths(config, Path(path).parent)


# --------------------------------------------------------------------------- #
# section handlers
# --------------------------------------------------------------------------- #
def _system(globals_: dict[str, list[str]]) -> dict[str, Any]:
    if "topology" not in globals_:
        raise ConfigError("The input file must declare a 'topology'.")
    system: dict[str, Any] = {"topology": globals_["topology"][0]}

    replicas = globals_.get("replica", [])
    if replicas:
        if len(replicas) % 2:
            raise ConfigError(
                "Each 'replica' line needs a name and at least one file: "
                "replica rep1 rep1/prod.nc"
            )
        mapping: dict[str, list[str]] = {}
        for index in range(0, len(replicas), 2):
            mapping.setdefault(replicas[index], []).append(replicas[index + 1])
        system["replicas"] = mapping
    elif "trajectory" in globals_:
        system["trajectory"] = list(globals_["trajectory"])
    else:
        raise ConfigError("The input file must declare a 'trajectory' or 'replica' lines.")

    if "time" in globals_:
        values = globals_["time"]
        dt = None if values[0].lower() in {"auto", "none", "null"} else float(values[0])
        system["time"] = {"dt": dt, "unit": values[1] if len(values) > 1 else "ps"}
    frames: dict[str, Any] = {}
    if "stride" in globals_:
        frames["stride"] = int(globals_["stride"][0])
    if "frames" in globals_:
        values = globals_["frames"]
        frames["start"] = int(values[0])
        if len(values) > 1 and values[1].lower() not in {"end", "none", "null"}:
            frames["stop"] = int(values[1])
    if frames:
        system["frames"] = frames
    if _flag(globals_, "align", False):
        system["align"] = {"enabled": True,
                           "selection": " ".join(globals_.get("align_selection",
                                                              ["protein and name CA"]))}
    return system


def _output(globals_: dict[str, list[str]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    if "output" in globals_:
        output["directory"] = globals_["output"][0]
    if "formats" in globals_:
        output["formats"] = list(globals_["formats"])
    if "dpi" in globals_:
        output["dpi"] = int(globals_["dpi"][0])
    return output


def _geometry(sections, name: str, n_atoms: int, analyses: dict, resolve) -> None:
    entries = sections.get(name, [])
    if not entries:
        return
    definitions = []
    for tokens, options, number in entries:
        if len(tokens) < n_atoms:
            raise ConfigError(
                f"Line {number}: '{name[:-1]}' needs {n_atoms} atom "
                f"specifications, found {len(tokens)}."
            )
        if len(tokens) == n_atoms:
            # Ambiguous: either the name was omitted, or one atom is missing.
            # Guessing wrong would silently measure a bogus selection, so the
            # first token has to look like an atom specification.
            if not _looks_like_atom_spec(tokens[0]):
                raise ConfigError(
                    f"Line {number}: '{tokens[0]}' does not look like an atom "
                    f"specification. If it is the name of the observable, the "
                    f"line needs {n_atoms} atom specifications after it; "
                    f"only {n_atoms - 1} were given."
                )
            label, atoms = None, tokens
        else:
            label, atoms = tokens[0], tokens[1:n_atoms + 1]
        aliases = [resolve(a) for a in atoms]
        entry: dict[str, Any] = {
            "name": label or f"{name[0]}_{'_'.join(aliases)}"[:40],
            "atoms": aliases,
        }
        if "threshold" in options:
            entry["threshold"] = float(options["threshold"])
        if "mode" in options:
            entry["mode"] = options["mode"]
        if "label" in options:
            entry["label"] = options["label"]
        definitions.append(entry)

    key = "distances" if name == "distances" else name
    field = "pairs" if name == "distances" else "definitions"
    analyses[key] = {"enabled": True, field: definitions}


def _hbonds(sections, analyses: dict, resolve) -> None:
    entries = sections.get("hbonds", [])
    if not entries:
        return
    pairs, between = [], []
    options_global: dict[str, Any] = {}
    for tokens, options, number in entries:
        head = tokens[0].lower() if tokens else ""
        if head == "auto":
            targets = [t for t in tokens[1:]]
            if len(targets) == 2:
                between.append([resolve_atom_spec(targets[0])[1],
                                resolve_atom_spec(targets[1])[1]])
            continue
        if head in {"cutoff", "distance"}:
            options_global["d_a_cutoff"] = float(tokens[1])
            continue
        if head == "angle":
            options_global["d_h_a_angle_cutoff"] = float(tokens[1])
            continue

        donor = options.get("donor")
        acceptor = options.get("acceptor")
        if donor and acceptor:
            name = tokens[0] if tokens else f"hb_{len(pairs) + 1}"
        elif len(tokens) >= 3:
            name, donor, acceptor = tokens[0], tokens[1], tokens[2]
        elif len(tokens) == 2:
            name, donor, acceptor = f"hb_{len(pairs) + 1}", tokens[0], tokens[1]
        else:
            raise ConfigError(
                f"Line {number}: an H-bond needs a donor and an acceptor."
            )
        entry = {"name": name, "donor": resolve(donor), "acceptor": resolve(acceptor)}
        if options.get("hydrogen"):
            entry["hydrogen"] = resolve(options["hydrogen"])
        pairs.append(entry)

    section: dict[str, Any] = {"enabled": True, **options_global}
    if pairs:
        section["pairs"] = pairs
    if between:
        section["auto"] = True
        section["auto_between"] = between
    analyses["hbonds"] = section


def _rmsd(sections, globals_, analyses: dict, resolve) -> None:
    """RMSD runs by default; the section customises it, 'rmsd off' disables it."""
    if not _flag(globals_, "rmsd", True):
        return
    entries = sections.get("rmsd", [])
    groups = []
    for tokens, options, number in entries:
        if len(tokens) == 1 and tokens[0].lower() in {"off", "no", "false"}:
            return
        if len(tokens) < 2:
            raise ConfigError(f"Line {number}: RMSD needs a name and a selection.")
        entry = {"name": tokens[0], "selection": resolve(" ".join(tokens[1:]))}
        if "fit" in options:
            entry["superposition"] = resolve(options["fit"])
        groups.append(entry)
    if not groups:
        groups = [{"name": "backbone", "selection": "backbone"}]
    analyses["rmsd"] = {"enabled": True, "groups": groups}


def _rmsf(sections, globals_, analyses: dict) -> None:
    if not _flag(globals_, "rmsf", True):
        return
    entries = sections.get("rmsf", [])
    section: dict[str, Any] = {"enabled": True, "selection": "protein and name CA"}
    for tokens, _options, number in entries:
        head = tokens[0].lower()
        if head in {"off", "no", "false"}:
            return
        if head == "selection":
            section["selection"] = " ".join(tokens[1:]).strip("{}")
        elif head == "highlight":
            section["highlight"] = [int(t) for t in tokens[1:] if t.isdigit()]
        else:
            section["selection"] = " ".join(tokens).strip("{}")
    analyses["rmsf"] = section


def _rdf(sections, analyses: dict, resolve) -> None:
    entries = sections.get("rdf", [])
    if not entries:
        return
    pairs = []
    for tokens, options, number in entries:
        if len(tokens) < 2:
            raise ConfigError(f"Line {number}: RDF needs two selections.")
        if len(tokens) == 2:
            name, first, second = f"rdf_{len(pairs) + 1}", tokens[0], tokens[1]
        else:
            name, first, second = tokens[0], tokens[1], tokens[2]
        entry: dict[str, Any] = {"name": name, "g1": resolve(first),
                                 "g2": resolve(second)}
        if "rmax" in options:
            entry["range"] = [0.0, float(options["rmax"])]
        if "bins" in options:
            entry["nbins"] = int(options["bins"])
        if options.get("exclude_same_residue", "").lower() in {"1", "yes", "true"}:
            entry["exclude_same_residue"] = True
        if "by" in options:
            entry["center"] = options["by"]
        pairs.append(entry)
    analyses["rdf"] = {"enabled": True, "pairs": pairs}


def _bridges(sections, analyses: dict, resolve) -> None:
    """``[bridges]``: ``name  groupA  groupB  [cutoff=3.5]``."""
    entries = sections.get("bridges", [])
    if not entries:
        return
    pairs = []
    for tokens, options, number in entries:
        if len(tokens) < 2:
            raise ConfigError(
                f"Line {number}: a bridge needs two groups (and optionally a name)."
            )
        if len(tokens) == 2:
            name, first, second = f"bridge_{len(pairs) + 1}", tokens[0], tokens[1]
        else:
            name, first, second = tokens[0], tokens[1], tokens[2]
        entry: dict[str, Any] = {"name": name, "group_a": resolve(first),
                                 "group_b": resolve(second)}
        if "cutoff" in options:
            entry["cutoff"] = float(options["cutoff"])
        if "solvent" in options:
            entry["solvent"] = resolve_atom_spec(options["solvent"])[1]
        pairs.append(entry)
    analyses["bridges"] = {"enabled": True, "pairs": pairs}


def _rgyr(sections, analyses: dict, resolve) -> None:
    entries = sections.get("rgyr", [])
    if not entries:
        return
    groups = []
    for tokens, _options, number in entries:
        if len(tokens) < 2:
            groups.append({"name": f"rg_{len(groups) + 1}",
                           "selection": resolve(tokens[0])})
        else:
            groups.append({"name": tokens[0],
                           "selection": resolve(" ".join(tokens[1:]))})
    analyses["radius_of_gyration"] = {"enabled": True, "groups": groups}


def _clustering(sections, analyses: dict, resolve) -> None:
    entries = sections.get("clustering", [])
    if not entries:
        return
    section: dict[str, Any] = {"enabled": True}
    for tokens, options, _number in entries:
        head = tokens[0].lower()
        if head in {"selection", "sel"}:
            section["selection"] = resolve(" ".join(tokens[1:]))
        elif head in {"n", "clusters", "n_clusters"}:
            section["n_clusters"] = int(tokens[1])
        elif head == "method":
            section["method"] = tokens[1]
        elif head == "write_selection":
            section["write_selection"] = resolve(" ".join(tokens[1:]))
        else:
            section["selection"] = resolve(" ".join(tokens))
        for key, value in options.items():
            if key in {"n", "clusters"}:
                section["n_clusters"] = int(value)
            elif key in {"method", "linkage"}:
                section[key] = value
    analyses["clustering"] = section


def _fes(sections, analyses: dict) -> None:
    entries = sections.get("fes", [])
    if not entries:
        return
    maps = []
    for tokens, options, number in entries:
        x = options.get("x")
        y = options.get("y")
        if not (x and y):
            if len(tokens) >= 3:
                name, x, y = tokens[0], tokens[1], tokens[2]
            elif len(tokens) == 2:
                name, x, y = f"fes_{len(maps) + 1}", tokens[0], tokens[1]
            else:
                raise ConfigError(f"Line {number}: a map needs two observables.")
        else:
            name = tokens[0] if tokens else f"fes_{len(maps) + 1}"
        entry: dict[str, Any] = {"name": name, "x": x, "y": y}
        if "bins" in options:
            entry["bins"] = int(options["bins"])
        if "temperature" in options:
            entry["temperature"] = float(options["temperature"])
        if options.get("free_energy", "").lower() in {"0", "no", "false"}:
            entry["free_energy"] = False
        maps.append(entry)
    analyses["free_energy_maps"] = {"enabled": True, "maps": maps}


def _flag(globals_: dict[str, list[str]], key: str, default: bool) -> bool:
    if key not in globals_:
        return default
    values = globals_[key]
    if not values:
        return True
    return values[0].strip().lower() not in {"off", "no", "false", "0"}
