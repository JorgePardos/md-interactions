# md_interactions

Reproducible analysis of interactions and **reaction geometry** in molecular
dynamics trajectories (AMBER `prmtop` + `.nc`/`.dcd`), aimed at enzyme–substrate
systems: catalytic distances, attack angles, hydrogen bonds, RMSD/RMSF, apparent
free-energy maps, radius of gyration, solvation shells and conformational
clustering.

Point it at a topology, a trajectory and a short input file describing what you
care about, and you get a `results/` folder with publication-ready figures
(PNG + PDF/SVG), CSV tables with every number behind them, and a single report.

```
results/
├── plots/        dist_d_nuc_timeseries.png/.pdf, rdf_w_od1.png, rmsf_per_residue.png ...
├── data/         distances.csv, angles.csv, rmsd.csv, rdf_w_od1.csv, observables.csv ...
├── summary/      summary.csv, summary.md, replica_overlap.csv
├── structures/   cluster1_frame1834.pdb   (representative of each cluster)
├── report.md
└── report.html
```

---

## Installation

```bash
pip install -e .
```

Python ≥ 3.10, and MDAnalysis, numpy, pandas, matplotlib, scipy and PyYAML.
AMBER `.nc` files are read through `scipy.io.netcdf`, so `netCDF4` is **not**
required. Clustering uses `scipy`, so scikit-learn is not required either.

---

## The input file

The quickest way in. A flat text file: no indentation rules, no quoting.

```
topology    system.prmtop
trajectory  prod.nc
time        0.1 ps
stride      1
output      results

[distances]
d_nuc    ASP20:OD1    TRH453:C1    threshold=3.5
d_wat    TRH453:O2P   water        mode=min

[angles]
a_attack ASP20:OD1    TRH453:C1    TRH453:O1

[rdf]
w_od1    ASP20:OD1    water   rmax=10
```

```bash
md-analyzer run -c analysis.in
```

`run` validates the whole configuration against the topology **before reading a
single frame**: every selection is resolved, and each 2D map is checked against
the observables the run will actually produce. A typo costs a second instead of
a full trajectory pass, and nothing is written when the check fails:

```
3 problem(s) found:

  [FAIL] distance:d_typo -> "resname ARG and resid 414 and name HH99"
         Matched 0 atoms.
  [FAIL] free_energy_maps:map1.y -> "d_typoo"
         'd_typoo' is not produced by any analysis; did you mean 'd_typo'?

Nothing was run. Fix the configuration, or use --no-check to run anyway.
```

`md-analyzer check -c analysis.in` runs the same validation on its own and
prints the full list of resolved selections with their atom counts.

### Writing atoms

Use whichever notation you have at hand — they all become MDAnalysis selections:

| You write | It means |
| --- | --- |
| `ASP20:OD1` | residue name + number, atom name (the residue name is **checked**) |
| `20:OD1` | residue number and atom name |
| `TRH:O2P` | by residue name — any residue so called |
| `:20@OD1` | cpptraj mask |
| `@1123` | atom number, 1-based as in cpptraj/VMD |
| `ASP20:OD1,OD2` | several atoms of the same residue |
| `TRH:*` | every atom of the residue |
| `:20-30` | a range of residues |
| `water` | water oxygens (also `protein`, `backbone`, `heavy`, `ligand`) |
| `{resid 20 and name OD1}` | a literal MDAnalysis selection |

Names declared in `[selections]` can be used everywhere afterwards:

```
[selections]
NUC   ASP20:OD1
SITE  {byres (protein and around 6 resname TRH)}
```

### Sections

| Section | Line format |
| --- | --- |
| `[distances]` | `name  atomA  atomB  [threshold=…] [mode=atom\|com\|min]` |
| `[angles]` | `name  atomA  atomB  atomC` |
| `[dihedrals]` | `name  atomA  atomB  atomC  atomD` |
| `[hbonds]` | `name donor=… acceptor=…`, plus `auto  regionA  regionB` |
| `[rmsd]` | `name  selection  [fit=selection]` — **on by default** |
| `[rmsf]` | `selection …`, `highlight 20 414` — **on by default** |
| `[rdf]` | `name  centre  partner  [rmax=…] [bins=…] [by=residue]` |
| `[bridges]` | `name  groupA  groupB  [cutoff=3.5]` |
| `[rgyr]` | `name  selection` |
| `[clustering]` | `selection …`, `n 3` |
| `[fes]` | `name x=obs1 y=obs2 [bins=…]` |

**RMSD and RMSF run by default** (backbone RMSD, per-residue RMSF on the Cα).
Declaring the section customises them; `rmsd off` / `rmsf off` disables them.
Everything else runs only when its section is present.

Global keys: `topology`, `trajectory`, `replica <name> <file>`, `time <dt> <unit>`,
`stride`, `frames <start> <stop>`, `output`, `formats`, `dpi`, `align`, `report`.

YAML is still fully supported (`md-analyzer init -o config.yaml`) and is what the
package uses internally; the input file is translated into exactly the same
`Config` object, so both go through the same validation.

---

## Solvation and RDF

Written around the question that actually comes up: *how is this atom or this
residue hydrated?* Every RDF reports `g(r)`, the running coordination number
`n(r)`, the **first solvation shell** (first peak, closing minimum and how many
molecules fit inside it) and the **hydration number over time**.

```
[rdf]
w_od1     ASP20:OD1   water   rmax=10 bins=100
w_res20   {resid 20}  water   rmax=12 by=residue
```

`by=residue` (also `proximal`) measures each water to the **nearest atom** of the
group, which is what "water around a residue" means for a non-spherical solute.
That profile is reported as a distribution (molecules/Å) rather than as a `g(r)`:
the accessible volume around an irregular solute is not `4πr²dr`, and dividing by
it anyway would produce a curve that cannot be compared with bulk density.
`by=com` uses the centre of mass instead — useful only for compact groups, since
from inside a buried residue `g(r)` never reaches bulk.

On a solvent-exposed aspartate this gives 2.6 waters within 3.25 Å of OD1, and
5.1 waters in the first shell of the whole residue.

### Three regimes, not two

A flat `g(r)` is not a failed measurement. Every pair is classified in the
summary as one of

| `regime` | Meaning |
| --- | --- |
| `structured` | a first peak and a closing minimum: a shell, with a coordination number |
| `excluded` | `g(r)` stays **below 1 everywhere** — the solvent is kept out. A buried site, or one whose donor is already committed to the backbone. This is a result |
| `bulk-like` | no structure, but no exclusion either |

When no minimum can be detected there is still a number: occupancy falls back to
a fixed 3.5 Å (the usual hydrogen-bond limit for water) and the summary says so,
rather than leaving the site with no measurement at all.

A `g_bulk_tail` column reports the mean `g(r)` over the outer quarter of the
range. It should be ~1; when it is not, the run warns that the range stops
before bulk or the centre sits where solvent cannot go, and that **the absolute
scale of `g(r)` should not be quoted** — which is the normal situation for a
buried active site.

### Who is in the shell

A coordination number of 2.0 is the same whether it is the same two molecules
throughout or a different pair every frame — and the chemistry is not. Each RDF
also writes `rdf_<name>_residents.csv` and a bar chart with, per partner
residue, its occupancy, its **longest uninterrupted stay** and its mean closest
approach:

```
resid   frames  occupancy_pct  longest_frames  mean_distance
16589     2660           99.0            2207           2.59
24244     2440           90.8            2330           2.61
```

Two permanently bound waters, not an exchanging shell. Identities are tracked
out to 6 Å; if a detected shell reaches further, the run says the occupancy per
molecule is unavailable instead of reporting an empty table.

## Bridging solvent

Two RDFs can both show a full first shell without a single molecule ever
touching *both* groups at once. When the question is whether a water could
relay a proton, what matters is the intersection:

```
[bridges]
# name        group A      group B   cutoff (A, both sides)
br_relay      ASP20:OD2    TRH:O1    cutoff=3.5
```

Reports how often a bridge exists, how many there are, **which molecule** it is,
the two distances that define it, and how long the same molecule stays — a
bridge held by one water for tens of picoseconds is a very different thing from
one remade by a different water every frame.

---

## Automatic detection

Rather than naming residues up front, ask what is there:

```bash
md-analyzer explore contacts --top x.prmtop --traj y.nc --around "resname LIG"
md-analyzer explore changes  --top x.prmtop --traj y.nc
```

`contacts` ranks every polar contact, salt bridge and hydrogen bond of a region
by occupancy. `changes` detects covalent bonds that break or form and protons
that hop between atoms — that is, it tells you the trajectory is **reactive** and
that the topology only describes the starting structure, so a selection by atom
name may not be measuring what its name suggests. `--export config.yaml` turns
what it found into a runnable configuration.

---

## Replicas

Listing several files under `trajectory` concatenates them into one series.
Independent runs should be declared as replicas instead:

```
replica rep1  rep1/prod.nc
replica rep2  rep2/prod.nc
replica rep3  rep3/prod.nc
```

Every frame then keeps its origin (a `replica` column in all CSVs), each
distribution gets one dashed curve per replica over the pooled histogram, and
`summary/replica_overlap.csv` reports the **overlap coefficient** of every pair
(1 = identical distributions, 0 = disjoint). Pairs below 0.70 are flagged in the
report, and bit-identical replicas are called out as duplicated input rather than
as perfect convergence.

No p-values are reported on purpose: consecutive MD frames are correlated, so a
Kolmogorov–Smirnov test assumes an effective sample size a trajectory does not
have and ends up calling almost everything significant.

---

## Pre-computed tables

For data measured elsewhere (`cpptraj` output, Monte Carlo runs, old results),
with no topology or trajectory needed:

```bash
md-analyzer init --data -o config_data.yaml
```

Reads the `#Frame` format of cpptraj (and CSV, and headerless files), validates
that every replica has the same columns in the same order, and produces the same
distributions, summary, 2D maps and replica comparison.

---

## Other entry points

```bash
md-analyzer wizard --top system.prmtop --traj prod.nc   # interactive builder
md-analyzer gui    --top system.prmtop --traj prod.nc   # 3D structure, click atoms
```

The wizard asks question by question, validating each selection against the real
topology (`?20`, `?ARG`, `?ligands` explore it without leaving the prompt). The
GUI opens a browser with the structure in 3D: click 2 atoms for a distance, 3 for
an angle, 4 for a dihedral. The 3D viewer is vendored, so it works with no
internet access; over SSH use `-L 8765:localhost:8765` and `--no-browser`.

---

## Available analyses

| Module | Produces |
| --- | --- |
| `distances` | Time series + histogram per distance, a multi-panel figure with all of them, mean ± sd, min/max, % below a threshold |
| `angles_dihedrals` | Angles in [0,180]° and dihedrals in (−180,180]°, with **circular** statistics for dihedrals |
| `rmsd_rmsf` | Global and local RMSD (fit on one selection, measure another) + per-residue RMSF |
| `hbonds` | D–A distance, D–H···A angle and occupancy of named bonds, plus automatic detection in a region |
| `rdf` | g(r), n(r), first solvation shell, hydration number over time, solvation regime (structured / excluded / bulk-like) and the occupancy of the shell per molecule |
| `water_bridges` | Solvent molecules within reach of two groups at once: occupancy, identity, geometry and residence |
| `free_energy_map` | 2D histogram or KDE of two observables, optionally as −kT ln P |
| `radius_of_gyration` | Rg of one or more selections |
| `clustering` | Hierarchical or k-means clustering, populations, PCA projection and a **PDB of the representative frame** |
| `replicas` | Distribution overlap between replicas and convergence warnings |
| `report` | Summary table (CSV/Markdown) and a single Markdown + HTML report |

### Details that affect interpretation

- **Local RMSD**: `fit=` says what to superpose on, the selection says what to
  measure. For a ligand or an active site, fit on `backbone`.
- **RMSF**: every frame is superposed on the average structure (two-pass
  iterative fit) before computing fluctuations; the per-residue value is the
  mass-weighted RMS of its atoms. Only the coordinates of the selection are held
  in memory.
- **Hydrogen bonds**: when the acceptor spans several atoms (a carboxylate, say)
  the closest one is used at each frame, and of the donor's hydrogens the one
  giving the most linear arrangement. Occupancy is the percentage of frames
  satisfying both criteria (default d ≤ 3.5 Å and ∠ ≥ 150°).
- **2D maps**: −kT ln P referred to the most populated bin. For unbiased MD this
  is a population map, **not** a converged free-energy surface; unsampled bins
  are left blank. Reweight metadynamics or umbrella data before feeding it here.
  The number of bins is capped automatically (√(n/2) rule) when there are few
  frames, and `smooth: <sigma>` applies Gaussian smoothing.
- **Clustering**: pairwise RMSD is computed after superposing on the average
  structure; above `max_frames` (2000) the trajectory is subsampled, since the
  cost is O(N²). The representative is the **medoid**, written as a PDB of the
  whole system — directly usable as a QM/MM starting point. `write_selection`
  trims it to a region.
- **Histograms**: probability density (Å⁻¹) by default. It is the only
  normalisation comparable between panels with different ranges, and the reason
  the ordinate exceeds 1 Å⁻¹ for narrow distributions (whenever σ < 0.40 Å).
- **PBC**: minimum-image convention is applied whenever the trajectory carries a
  valid box.

---

## Package layout

```
src/md_interactions/
├── config.py        typed configuration + YAML loading and validation
├── inputfile.py     the compact input format and its atom specifications
├── system.py        Universe, stride, time axis, selections, replicas
├── tabular.py       reading pre-computed tables (cpptraj .dat, CSV)
├── explore.py       automatic detection of contacts and chemical changes
├── plotting.py      shared style, palette, multi-format saving
├── io_utils.py      results/ tree, CSV, Markdown tables
├── results.py       AnalysisResult (tables, series, summary, figures)
├── runner.py        orchestration
├── report.py        summary + Markdown/HTML report
├── cli.py           run / check / explore / wizard / gui / init
├── wizard.py        interactive builder
├── gui/             local web interface with a 3D viewer
├── testing.py       synthetic toy system
└── analyses/        distances, angles_dihedrals, rmsd_rmsf, hbonds,
                     free_energy_map, radius_of_gyration, rdf, clustering,
                     water_bridges, distributions, replicas
```

Every module in `analyses/` follows the same contract:

```python
run(system, paths, config=None, verbose=True) -> AnalysisResult   # writes CSV + figures
compute_*(system, config=None) -> pandas.DataFrame                # numbers only
```

---

## Library use

```python
import md_interactions as mdi

config = mdi.load_input("analysis.in")        # or mdi.load_config("config.yaml")
out = mdi.run_analyses(config)

out.summary                                   # mean ± sd, min, max
out.observables                               # every time series in one frame
out.result("distances").tables["distances"]
```

---

## Tests

```bash
pytest
```

142 tests on a synthetic 42-atom system generated on the fly
(`md_interactions.testing`), so no trajectories are needed in the repository.
They cover configuration and input-file parsing, frame handling, numerical
checks of distances/angles/RMSD/Rg against direct numpy calculations, hydrogen
bond occupancies, free-energy scaling with temperature, RDF shells, clustering,
replica comparison and the full CLI → report path.

---

## Units

| Quantity | Unit |
| --- | --- |
| Distances, RMSD, RMSF, Rg, r of g(r) | Å |
| Angles and dihedrals | ° |
| Time | ns by default (`ps` or `frame` configurable) |
| Free energy | kcal/mol (or kJ/mol, kT) |
