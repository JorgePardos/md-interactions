"""Orchestration: configuration in, ``results/`` directory out."""

from __future__ import annotations

import time as _time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from .config import Config, load_config
from .exceptions import MDInteractionsError
from .io_utils import OutputPaths, prepare_output, write_csv
from .plotting import apply_style
from .results import AnalysisResult
from .system import MDSystem, load_system
from .tabular import Dataset, load_dataset

__all__ = ["RunOutput", "run_analyses", "run_from_config"]


@dataclass
class RunOutput:
    """Everything a run produced, for programmatic use in notebooks."""

    config: Config
    system: MDSystem | None
    paths: OutputPaths
    dataset: "Dataset | None" = None
    results: list[AnalysisResult] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)
    observables: pd.DataFrame | None = None
    summary: pd.DataFrame | None = None
    report_files: list[Path] = field(default_factory=list)

    def result(self, name: str) -> AnalysisResult | None:
        """Return the result of a given analysis, if it ran."""
        for res in self.results:
            if res.name == name:
                return res
        return None

    @property
    def ok(self) -> bool:
        return not self.failures


def _dispatch(system: MDSystem) -> list[tuple[str, Callable[..., AnalysisResult]]]:
    """Build the ordered list of (name, runner) for the enabled analyses.

    Imports are local so that a run only pays for what it uses.
    """
    cfg = system.config
    jobs: list[tuple[str, Callable[..., AnalysisResult]]] = []

    if cfg.distances.enabled:
        from .analyses import distances
        jobs.append(("distances", distances.run))
    if cfg.angles.enabled:
        from .analyses import angles_dihedrals
        jobs.append(("angles_dihedrals", angles_dihedrals.run))
    if cfg.rmsd.enabled or cfg.rmsf.enabled:
        from .analyses import rmsd_rmsf
        jobs.append(("rmsd_rmsf", rmsd_rmsf.run))
    if cfg.hbonds.enabled:
        from .analyses import hbonds
        jobs.append(("hbonds", hbonds.run))
    if cfg.rgyr.enabled:
        from .analyses import radius_of_gyration
        jobs.append(("radius_of_gyration", radius_of_gyration.run))
    if cfg.rdf.enabled:
        from .analyses import rdf
        jobs.append(("rdf", rdf.run))
    if cfg.clustering.enabled:
        from .analyses import clustering
        jobs.append(("clustering", clustering.run))
    return jobs


def run_analyses(
    config: Config,
    verbose: bool = True,
    strict: bool = False,
    system: MDSystem | None = None,
    dataset: "Dataset | None" = None,
) -> RunOutput:
    """Run every enabled analysis and write the ``results/`` tree.

    Parameters
    ----------
    config
        Validated configuration (see :func:`~md_interactions.config.load_config`).
    verbose
        Print progress to stdout.
    strict
        Abort on the first failing analysis instead of continuing with the
        remaining ones.
    system
        Pre-loaded system; loaded from ``config`` when omitted.

    Notes
    -----
    Analyses are independent: a failure in one module (typically an empty
    selection) is reported and the rest still run, so a long trajectory does
    not have to be re-read because of a typo in one selection.
    """
    apply_style(config.output.style)
    paths = prepare_output(config.output.directory)

    if config.data_only:
        return _run_dataset(config, paths, verbose=verbose, strict=strict,
                            dataset=dataset)

    system = system or load_system(config, verbose=verbose)

    output = RunOutput(config=config, system=system, paths=paths)
    started = _time.perf_counter()

    for name, runner in _dispatch(system):
        try:
            t0 = _time.perf_counter()
            result = runner(system, paths, verbose=verbose)
            output.results.append(result)
            if verbose:
                print(f"[{name}] done in {_time.perf_counter() - t0:.1f} s")
        except MDInteractionsError as exc:
            if strict:
                raise
            output.failures.append((name, str(exc)))
            print(f"[{name}] FAILED: {exc}")
        except Exception as exc:  # pragma: no cover - unexpected library errors
            if strict:
                raise
            output.failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"[{name}] FAILED ({type(exc).__name__}): {exc}")
            if verbose:
                traceback.print_exc()

    output.observables = _merge_observables(output.results)

    # replicas keep their identity: check whether they sample the same thing
    if system.replica_labels is not None and output.observables is not None:
        _run_replica_check(output, verbose=verbose, strict=strict)

    # 2D maps need the observables produced above, so they run last.
    if config.free_energy.enabled:
        try:
            from .analyses import free_energy_map
            result = free_energy_map.run(
                system, paths, observables=output.observables,
                units=_observable_units(output.results), verbose=verbose,
            )
            output.results.append(result)
        except MDInteractionsError as exc:
            if strict:
                raise
            output.failures.append(("free_energy_maps", str(exc)))
            print(f"[free_energy_maps] FAILED: {exc}")

    output.summary = _merge_summaries(output.results)
    if output.summary is not None and not output.summary.empty:
        write_csv(output.summary, paths.summary, "summary")
    if output.observables is not None and output.observables.shape[1] > 2:
        write_csv(output.observables, paths.data, "observables")

    if config.report.enabled:
        from .report import write_report
        output.report_files = write_report(output)

    if verbose:
        elapsed = _time.perf_counter() - started
        print(f"[md_interactions] finished in {elapsed:.1f} s -> {paths.root}")
        if output.failures:
            print(f"[md_interactions] {len(output.failures)} analysis/analyses failed:")
            for name, msg in output.failures:
                print(f"  - {name}: {msg.splitlines()[0]}")
    return output


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    """Observable columns of a time-series table (frame/time/replica excluded)."""
    import pandas.api.types as ptypes

    return [c for c in frame.columns
            if c not in ("frame", "time", "replica") and ptypes.is_numeric_dtype(frame[c])]


def _run_replica_check(output: RunOutput, verbose: bool, strict: bool) -> None:
    """Compare the replicas and append the result (never fatal by itself)."""
    from .analyses import replicas

    frame = output.observables
    columns = _numeric_columns(frame)
    if not columns or "replica" not in frame.columns:
        return
    unit = output.config.data.unit if output.config.data else "Å"
    try:
        result = replicas.run(
            frame, columns, output.paths,
            formats=output.config.output.formats, dpi=output.config.output.dpi,
            unit=unit, verbose=verbose,
        )
        output.results.append(result)
    except MDInteractionsError as exc:
        if strict:
            raise
        output.failures.append(("replicas", str(exc)))
        print(f"[replicas] FAILED: {exc}")


def _run_dataset(
    config: Config, paths: OutputPaths, verbose: bool, strict: bool,
    dataset: "Dataset | None" = None,
) -> RunOutput:
    """``data:`` mode — analyse pre-computed tables, no trajectory involved."""
    from .analyses import distributions

    started = _time.perf_counter()
    dataset = dataset or load_dataset(config, verbose=verbose)
    output = RunOutput(config=config, system=None, paths=paths, dataset=dataset)

    result = distributions.run(dataset, paths, verbose=verbose)
    output.results.append(result)
    output.observables = dataset.data

    if dataset.replica_labels is not None:
        _run_replica_check(output, verbose=verbose, strict=strict)

    if config.free_energy.enabled:
        try:
            from .analyses import free_energy_map
            output.results.append(free_energy_map.run(
                None, paths, config=config.free_energy,
                observables=dataset.data,
                units={c: dataset.unit for c in dataset.columns},
                output_config=config.output, verbose=verbose,
            ))
        except MDInteractionsError as exc:
            if strict:
                raise
            output.failures.append(("free_energy_maps", str(exc)))
            print(f"[free_energy_maps] FAILED: {exc}")

    output.summary = _merge_summaries(output.results)
    if output.summary is not None and not output.summary.empty:
        write_csv(output.summary, paths.summary, "summary")

    if config.report.enabled:
        from .report import write_report
        output.report_files = write_report(output)

    if verbose:
        print(f"[md_interactions] finished in "
              f"{_time.perf_counter() - started:.1f} s -> {paths.root}")
    return output


def run_from_config(
    path: str | Path, verbose: bool = True, strict: bool = False, **overrides
) -> RunOutput:
    """Convenience wrapper: load a YAML file and run everything in it."""
    config = load_config(path, overrides=overrides or None)
    return run_analyses(config, verbose=verbose, strict=strict)


def _merge_observables(results: list[AnalysisResult]) -> pd.DataFrame | None:
    """Join every per-frame time series into a single wide DataFrame."""
    frames = [r.series for r in results if r.series is not None and not r.series.empty]
    if not frames:
        return None
    merged = frames[0]
    for extra in frames[1:]:
        cols = [c for c in extra.columns if c not in ("frame", "time", "replica")]
        if not cols:
            continue
        merged = merged.merge(extra[["frame", *cols]], on="frame", how="outer")
    return merged.sort_values("frame").reset_index(drop=True)


def _observable_units(results: list[AnalysisResult]) -> dict[str, str]:
    """Map observable name -> unit, used to label the axes of the 2D maps."""
    units: dict[str, str] = {}
    for result in results:
        if result.summary is None or result.summary.empty:
            continue
        if {"observable", "unit"} <= set(result.summary.columns):
            for name, unit in zip(result.summary["observable"], result.summary["unit"]):
                units.setdefault(str(name), str(unit))
    return units


def _merge_summaries(results: list[AnalysisResult]) -> pd.DataFrame | None:
    frames = [r.summary for r in results if r.summary is not None and not r.summary.empty]
    if not frames:
        return None
    summary = pd.concat(frames, ignore_index=True)
    leading = ["analysis", "observable", "unit", "n_frames", "mean", "std", "min", "max"]
    ordered = [c for c in leading if c in summary.columns]
    ordered += [c for c in summary.columns if c not in ordered]
    return summary[ordered]
