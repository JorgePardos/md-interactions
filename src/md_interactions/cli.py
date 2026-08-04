"""Command line interface (``md-analyzer`` / ``md_interactions``).

Subcommands
-----------
``run``
    Execute every analysis enabled in the YAML configuration.
``check``
    Load the system and resolve every selection without running anything.
    The fastest way to debug a configuration on a big trajectory.
``init``
    Write a commented example configuration to disk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import Config, load_config
from .exceptions import MDInteractionsError
from .system import load_system
from .templates import DATA_CONFIG, EXAMPLE_CONFIG

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="md-analyzer",
        description="Interaction/geometry analysis of MD trajectories "
                    "(AMBER prmtop + nc/dcd) driven by a YAML configuration.",
    )
    parser.add_argument("--version", action="version", version=f"md_interactions {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the analyses defined in the configuration")
    _add_common(run)
    run.add_argument("--strict", action="store_true",
                     help="abort on the first failing analysis")
    run.add_argument("-q", "--quiet", action="store_true", help="reduce output")
    run.set_defaults(func=_cmd_run)

    check = sub.add_parser("check", help="validate the configuration and all selections")
    _add_common(check)
    check.set_defaults(func=_cmd_check)

    gui = sub.add_parser(
        "gui",
        help="open the graphical interface: 3D structure, click the atoms to "
             "measure, run and see the figures in one window",
    )
    gui.add_argument("--top", required=True, type=Path, help="topology")
    gui.add_argument("--traj", type=Path, nargs="+", help="trajectory file(s)")
    gui.add_argument("--port", type=int, default=8765, help="port (default: 8765)")
    gui.add_argument("--stride", type=int, default=1, help="default frame stride")
    gui.add_argument("--out", type=Path, default=Path("results_gui"),
                     help="results directory")
    gui.add_argument("--no-browser", action="store_true",
                     help="do not open a browser (use with SSH port forwarding)")
    gui.set_defaults(func=_cmd_gui)

    explore = sub.add_parser(
        "explore",
        help="discover what is worth measuring: contacts, chemical changes, "
             "differences between runs",
    )
    explore.add_argument("mode", choices=["contacts", "changes", "compare", "reaction"],
                         help="contacts: polar contacts around a region | "
                              "changes: bonds and protons that move | "
                              "compare: two trajectories side by side | "
                              "reaction: reaction geometry only")
    explore.add_argument("--top", required=True, type=Path, help="topology")
    explore.add_argument("--traj", required=True, type=Path, nargs="+",
                         help="trajectory file(s)")
    explore.add_argument("--traj-b", type=Path, nargs="+",
                         help="second trajectory (compare mode)")
    explore.add_argument("--around", default=None,
                         help='region to look around (default: the ligand, i.e. '
                              'everything that is not protein or solvent)')
    explore.add_argument("--cutoff", type=float, default=4.0,
                         help="heavy-atom distance for candidate contacts (Å)")
    explore.add_argument("--stride", type=int, default=1, help="frame stride")
    explore.add_argument("--waters", choices=["aggregate", "individual", "ignore"],
                         default="aggregate", help="how to treat solvent partners")
    explore.add_argument("--min-occupancy", type=float, default=5.0,
                         help="drop contacts below this occupancy (%%)")
    explore.add_argument("--export", type=Path,
                         help="write the detected interactions as a config.yaml")
    explore.add_argument("--top-n", type=int, default=25,
                         help="how many rows to show")
    explore.set_defaults(func=_cmd_explore)

    wizard = sub.add_parser(
        "wizard",
        help="build the configuration interactively, validating selections "
             "against the topology as you go",
    )
    wizard.add_argument("--top", type=Path, help="topology (asked for if omitted)")
    wizard.add_argument("--traj", type=Path, nargs="+", help="trajectory file(s)")
    wizard.add_argument("-o", "--output", default=Path("config.yaml"), type=Path,
                        help="destination file (default: config.yaml)")
    wizard.add_argument("-f", "--force", action="store_true",
                        help="overwrite the destination without asking")
    wizard.set_defaults(func=_cmd_wizard)

    init = sub.add_parser("init", help="write an example config.yaml")
    init.add_argument("-o", "--output", default="config.yaml", type=Path,
                      help="destination file (default: config.yaml)")
    init.add_argument("-f", "--force", action="store_true", help="overwrite if it exists")
    init.add_argument("--data", action="store_true",
                      help="template for pre-computed tables (cpptraj .dat) "
                           "instead of a topology + trajectory")
    init.set_defaults(func=_cmd_init)

    return parser


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-c", "--config", required=True, type=Path,
                        help="YAML configuration file")
    parser.add_argument("--top", type=Path, help="override system.topology")
    parser.add_argument("--traj", type=Path, nargs="+", help="override system.trajectory")
    parser.add_argument("--out", type=Path, help="override output.directory")
    parser.add_argument("--stride", type=int, help="override system.frames.stride")
    parser.add_argument("--start", type=int, help="override system.frames.start")
    parser.add_argument("--stop", type=int, help="override system.frames.stop")
    parser.add_argument("--formats", nargs="+", help="override output.formats (png pdf svg)")


def _load(args: argparse.Namespace) -> Config:
    """Load the configuration, accepting both the ``.in`` and YAML formats."""
    overrides = {
        "topology": args.top,
        "trajectory": args.traj,
        "directory": args.out,
        "stride": args.stride,
        "start": args.start,
        "stop": args.stop,
        "formats": args.formats,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}

    if Path(args.config).suffix.lower() in {".in", ".inp", ".txt"}:
        from .inputfile import load_input

        if overrides:
            print("[md_interactions] note: command-line overrides are only "
                  "applied to YAML configurations; edit the input file instead.")
        return load_input(args.config)
    return load_config(args.config, overrides=overrides or None)


def _cmd_run(args: argparse.Namespace) -> int:
    from .runner import run_analyses

    config = _load(args)
    verbose = not args.quiet
    if verbose:
        print(f"[md_interactions] config: {config.source}")
        print(f"[md_interactions] analyses: {', '.join(config.enabled_analyses()) or 'none'}")
    output = run_analyses(config, verbose=verbose, strict=args.strict)
    return 0 if output.ok else 1


def _cmd_check(args: argparse.Namespace) -> int:
    config = _load(args)
    print(f"Configuration OK: {config.source}")
    if config.data_only:
        return _check_data(config)
    print(f"  topology  : {config.system.topology}")
    for traj in config.system.trajectory:
        print(f"  trajectory: {traj}")
    print(f"  analyses  : {', '.join(config.enabled_analyses()) or 'none enabled'}")

    system = load_system(config, verbose=True)
    print(f"  frames    : {system.n_frames} analysed "
          f"({system.times[0]:g} - {system.times[-1]:g} {system.time_unit})")

    failures = 0
    print("\nSelections:")
    for context, token in _collect_tokens(config):
        selection = config.resolve_selection(token)
        try:
            group = system.select(token, name=context)
            print(f"  [ok]   {context:<28} {token:<22} -> {group.n_atoms:6d} atoms")
        except MDInteractionsError as exc:
            failures += 1
            print(f"  [FAIL] {context:<28} {token:<22} -> \"{selection}\"")
            print(f"         {str(exc).splitlines()[0]}")
    if failures:
        print(f"\n{failures} selection(s) failed.")
        return 1
    print("\nAll selections resolve. Ready to run.")
    return 0


def _cmd_gui(args: argparse.Namespace) -> int:
    from .gui import serve

    serve(topology=args.top, trajectory=args.traj or [], port=args.port,
          stride=args.stride, open_browser=not args.no_browser,
          results_dir=args.out)
    return 0


def _explore_system(topology: Path, trajectory: list[Path], stride: int):
    """Minimal system for exploration: no analyses, no output tree."""
    from .config import Config
    from .system import load_system

    config = Config.from_dict({
        "system": {
            "topology": str(topology),
            "trajectory": [str(p) for p in trajectory],
            "frames": {"stride": int(stride)},
        },
        "output": {"directory": "explore_tmp"},
    })
    return load_system(config, verbose=True)


def _cmd_explore(args: argparse.Namespace) -> int:
    import pandas as pd

    from . import explore as ex

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 40)
    region = args.around or ex.DEFAULT_REGION
    system = _explore_system(args.top, args.traj, args.stride)

    if args.mode == "contacts":
        table = ex.find_contacts(system, region=region, cutoff=args.cutoff,
                                 waters=args.waters,
                                 min_occupancy=args.min_occupancy)
        if table.empty:
            print("No contacts found above the occupancy threshold.")
            return 0
        shown = table.drop(columns=["index_a", "index_b"]).head(args.top_n)
        print(f"\nContacts around \"{region}\":\n")
        print(shown.to_string(index=False))
        if args.export:
            _export_contacts(table.head(args.top_n), args, region)
        return 0

    if args.mode == "changes":
        bonds = ex.detect_bond_changes(system)
        protons = ex.detect_proton_transfers(system)
        if bonds.empty and protons.empty:
            print("\nNo covalent bond changes or proton transfers detected: "
                  "the topology is consistent with this trajectory.")
            return 0
        if not bonds.empty:
            print("\nCovalent bond changes:\n")
            print(bonds.to_string(index=False))
        if not protons.empty:
            print("\nProton transfers during the run:\n")
            print(protons.to_string(index=False))
        print("\nA reactive trajectory means the topology describes the starting "
              "structure only: selections by atom name may not measure what "
              "their name suggests.")
        return 0

    if args.mode == "reaction":
        report = ex.reaction_summary(system, region=region)
        if not report.bonds.empty:
            print("\nBonds made and broken:\n")
            print(report.bonds.to_string(index=False))
        if not report.protons.empty:
            print("\nProton transfers:\n")
            print(report.protons.to_string(index=False))
        if not report.approaches.empty:
            print("\nClose approaches in the region:\n")
            print(report.approaches.drop(columns=["index_a", "index_b"])
                  .head(args.top_n).to_string(index=False))
        if not report.is_reactive:
            print("\nNo chemical change detected: this looks like a "
                  "non-reactive (classical) trajectory.")
        return 0

    # compare
    if not args.traj_b:
        print("ERROR: 'compare' needs a second trajectory (--traj-b).",
              file=sys.stderr)
        return 2
    first = ex.find_contacts(system, region=region, cutoff=args.cutoff,
                             waters=args.waters, min_occupancy=0.0)
    system_b = _explore_system(args.top, args.traj_b, args.stride)
    second = ex.find_contacts(system_b, region=region, cutoff=args.cutoff,
                              waters=args.waters, min_occupancy=0.0)
    merged = ex.compare_contacts(first, second, names=("A", "B"))
    if merged.empty:
        print("\nNo contact changes occupancy by more than 10 % between the runs.")
        return 0
    print("\nContacts that change between the two trajectories "
          "(A = --traj, B = --traj-b):\n")
    print(merged.head(args.top_n).to_string(index=False))
    return 0


def _export_contacts(table, args: argparse.Namespace, region: str) -> None:
    """Turn the detected contacts into a ready-to-run configuration."""
    import yaml

    selections: dict[str, str] = {}
    pairs: list[dict] = []
    for _i, row in table.iterrows():
        if str(row["atom_b"]).endswith("(any)"):
            continue                      # aggregated solvent has no fixed partner
        alias_a = _alias_from_label(row["atom_a"])
        alias_b = _alias_from_label(row["atom_b"])
        selections[alias_a] = _selection_from_label(row["atom_a"])
        selections[alias_b] = _selection_from_label(row["atom_b"])
        pairs.append({
            "name": f"d_{alias_a}_{alias_b}"[:40],
            "atoms": [alias_a, alias_b],
            "label": f"{row['atom_a']} - {row['atom_b']}",
            "threshold": 3.5,
        })

    config = {
        "system": {
            "topology": str(args.top),
            "trajectory": [str(p) for p in args.traj],
            "time": {"dt": None, "unit": "ps"},
        },
        "output": {"directory": "results", "formats": ["png", "pdf"], "dpi": 300},
        "selections": selections,
        "analyses": {"distances": {"enabled": True, "pairs": pairs}},
        "report": {"enabled": True, "formats": ["markdown", "html"]},
    }
    header = (f"# Generated by `md-analyzer explore contacts --around \"{region}\"`\n"
              f"# {len(pairs)} contact(s) detected automatically; edit freely.\n\n")
    args.export.parent.mkdir(parents=True, exist_ok=True)
    args.export.write_text(
        header + yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"\nWritten {args.export} with {len(pairs)} distance(s). Next:\n"
          f"  md-analyzer check -c {args.export}")


def _alias_from_label(label: str) -> str:
    keep = [c if (c.isalnum() or c == "_") else "_" for c in str(label)]
    return "".join(keep).strip("_")


def _selection_from_label(label: str) -> str:
    """``ASP20:OD1`` -> ``resid 20 and name OD1``."""
    residue, _sep, atom = str(label).partition(":")
    number = "".join(c for c in residue if c.isdigit())
    return f"resid {number} and name {atom}" if number else f"name {atom}"


def _cmd_wizard(args: argparse.Namespace) -> int:
    from .wizard import run_wizard

    try:
        run_wizard(topology=args.top, trajectory=args.traj, output=args.output,
                   force=args.force)
    except (KeyboardInterrupt, EOFError):
        print("\nWizard cancelled; nothing was written.", file=sys.stderr)
        return 130
    return 0


def _check_data(config: Config) -> int:
    """``check`` for the ``data:`` mode: files, columns and replica balance."""
    from .tabular import load_dataset

    for path in config.data.files:
        print(f"  data file : {path}")
    for name, paths in config.data.replicas.items():
        print(f"  replica {name}: {', '.join(str(p) for p in paths)}")

    dataset = load_dataset(config, verbose=True)
    print(f"  rows      : {dataset.n_frames}")
    print(f"  time axis : {dataset.time_label}")
    print("\nColumns:")
    for column in dataset.columns:
        values = dataset.values(column)
        print(f"  [ok]   {column:<28} n = {values.size:6d}   "
              f"{values.mean():7.3f} ± {values.std():.3f} {dataset.unit}   "
              f"[{values.min():.2f}, {values.max():.2f}]")

    unknown = [c for c in config.free_energy.maps
               if c.x not in dataset.columns or c.y not in dataset.columns]
    if unknown:
        for spec in unknown:
            print(f"  [FAIL] free-energy map '{spec.name}' references "
                  f"'{spec.x}' / '{spec.y}', not in the data")
        return 1
    print("\nAll columns available. Ready to run.")
    return 0


def _cmd_init(args: argparse.Namespace) -> int:
    path: Path = args.output
    if path.exists() and not args.force:
        print(f"{path} already exists (use --force to overwrite).", file=sys.stderr)
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DATA_CONFIG if args.data else EXAMPLE_CONFIG, encoding="utf-8")
    print(f"Example configuration written to {path}")
    return 0


def _collect_tokens(config: Config) -> list[tuple[str, str]]:
    """Every (context, selection token) pair referenced by the configuration."""
    tokens: list[tuple[str, str]] = []
    for name, selection in config.selections.items():
        tokens.append((f"selections.{name}", name))
    if config.system.align.enabled:
        tokens.append(("system.align", config.system.align.selection))
    for defn in config.distances.pairs:
        tokens += [(f"distance:{defn.name}", tok) for tok in defn.atoms]
    for defn in config.angles.angles:
        tokens += [(f"angle:{defn.name}", tok) for tok in defn.atoms]
    for defn in config.angles.dihedrals:
        tokens += [(f"dihedral:{defn.name}", tok) for tok in defn.atoms]
    for group in config.rmsd.groups:
        tokens.append((f"rmsd:{group.name}", group.selection))
        if group.superposition:
            tokens.append((f"rmsd:{group.name}.fit", group.superposition))
    if config.rmsf.enabled:
        tokens.append(("rmsf", config.rmsf.selection))
        if config.rmsf.align_selection:
            tokens.append(("rmsf.fit", config.rmsf.align_selection))
    for hb in config.hbonds.pairs:
        tokens.append((f"hbond:{hb.name}.donor", hb.donor))
        tokens.append((f"hbond:{hb.name}.acceptor", hb.acceptor))
        if hb.hydrogen:
            tokens.append((f"hbond:{hb.name}.hydrogen", hb.hydrogen))
    for i, (sel_a, sel_b) in enumerate(config.hbonds.auto_between):
        tokens.append((f"hbonds.auto_between[{i}].0", sel_a))
        tokens.append((f"hbonds.auto_between[{i}].1", sel_b))
    for group in config.rgyr.groups:
        tokens.append((f"rgyr:{group.name}", group.selection))
    for pair in config.rdf.pairs:
        tokens.append((f"rdf:{pair.name}.g1", pair.g1))
        tokens.append((f"rdf:{pair.name}.g2", pair.g2))
    if config.clustering.enabled:
        tokens.append(("clustering", config.clustering.selection))

    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str]] = []
    for entry in tokens:
        if entry not in seen:
            seen.add(entry)
            unique.append(entry)
    return unique


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except MDInteractionsError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
