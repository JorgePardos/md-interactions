"""Configuration template shipped with the package (``md-analyzer init``)."""

from __future__ import annotations

__all__ = ["EXAMPLE_CONFIG", "DATA_CONFIG"]

EXAMPLE_CONFIG = """\
# =============================================================================
# md_interactions - example configuration
#
#   md-analyzer run -c config.yaml
#
# Selection syntax is MDAnalysis':
#   https://userguide.mdanalysis.org/stable/selections.html
# Reminder for AMBER prmtop topologies: histidines are HIE/HID/HIP, cysteines
# in disulfides are CYX, and 'resid' follows the topology numbering.
# =============================================================================

system:
  topology: system.prmtop
  trajectory:                 # one file, or several concatenated as one run
    - prod_01.nc
    - prod_02.nc
  # Independent runs? Declare them as replicas instead of a flat list: the
  # frames keep their origin, every distribution gets one curve per replica and
  # the report warns when two replicas do not sample the same thing.
  #   replicas:
  #     rep1: [rep1/prod.nc]
  #     rep2: [rep2/prod.nc]
  #     rep3: [rep3/prod.nc]
  time:
    dt: 0.01                  # time between SAVED frames; null -> read from traj
    unit: ns                  # ps | ns | frame
  frames:
    start: 0
    stop: null                # null = until the end
    stride: 1
  align:
    enabled: false            # distances/angles do not need it; RMSF aligns internally
    selection: "protein and name CA"
    reference: null           # optional external structure (pdb/rst7)

output:
  directory: results
  formats: [png, pdf]         # pdf/svg = vector, for publication
  dpi: 300

# Named selections: reuse them everywhere below instead of repeating strings.
selections:
  SER160_OG:  "resid 160 and name OG"
  HIS237_NE2: "resid 237 and name NE2"
  ASP206_OD2: "resid 206 and name OD2"
  LIG_C:      "resname LIG and name C1"
  LIG_O:      "resname LIG and name O1"
  ACTIVE_SITE: "byres (protein and around 6 resname LIG)"
  WATERS:     "resname WAT HOH and name O"

analyses:

  distances:
    enabled: true
    bins: 60
    kde: true
    running_average: 0        # >1 to overlay a smoothed line
    pbc: true
    facet: true               # single multi-panel figure with all distributions
    normalization: density    # density (Å⁻¹) | percent | counts
                              # only 'density' is comparable between panels
                              # whose bin widths differ
    pairs:
      - name: d_nuc           # nucleophilic attack distance
        atoms: [SER160_OG, LIG_C]
        label: "Ser160 OG - LIG C1"
        threshold: 3.5        # optional: reports % of frames below it
      - name: d_acid
        atoms: [HIS237_NE2, SER160_OG]
      - name: d_water_min     # closest catalytic water (mode: min)
        atoms: [LIG_C, WATERS]
        mode: min             # atom | com | min

  angles:
    enabled: true
    definitions:
      - name: a_attack        # Burgi-Dunitz type attack angle
        atoms: [SER160_OG, LIG_C, LIG_O]
        label: "OG-C1-O1 attack angle"

  dihedrals:
    enabled: false
    definitions:
      - name: chi1_ser160
        atoms: ["resid 160 and name N", "resid 160 and name CA",
                "resid 160 and name CB", "resid 160 and name OG"]

  rmsd:
    enabled: true
    ref_frame: 0              # or 'reference:' with an external structure
    groups:
      - name: backbone
        selection: "backbone"
      - name: active_site
        selection: "ACTIVE_SITE and not name H*"
        superposition: "backbone"   # fit on the protein, measure locally
      - name: ligand
        selection: "resname LIG and not name H*"
        superposition: "backbone"

  rmsf:
    enabled: true
    selection: "protein and name CA"
    align: true               # align to the average structure first
    highlight: [160, 206, 237]  # residues marked in the plot

  hbonds:
    enabled: true
    d_a_cutoff: 3.5           # donor-acceptor distance (A)
    d_h_a_angle_cutoff: 150   # D-H...A angle (degrees)
    pairs:                    # explicitly tracked H-bonds
      - name: hb_ser_his
        donor: SER160_OG
        acceptor: HIS237_NE2
      - name: hb_his_asp
        donor: HIS237_NE2
        acceptor: ASP206_OD2
    auto: false               # automatic detection in a region
    auto_between:
      - ["resname LIG", "protein"]
    min_occupancy: 5.0        # % below which detected H-bonds are dropped
    max_reported: 25

  free_energy_maps:
    enabled: true
    temperature: 300.0
    maps:
      - name: fes_dnuc_dacid
        x: d_nuc              # any observable computed above
        y: d_acid
        method: histogram     # histogram | kde
        bins: 60              # capped automatically if there are few frames
        smooth: 0.0           # Gaussian smoothing of the histogram, in bins
        free_energy: true     # -kT ln P (relative to the most populated bin)
        energy_unit: kcal/mol # kcal/mol | kJ/mol | kT
        max_energy: 6.0       # clip the colour bar

  radius_of_gyration:
    enabled: false
    groups:
      - name: rg_protein
        selection: "protein"
      - name: rg_active_site
        selection: "ACTIVE_SITE"

  rdf:
    enabled: false
    pairs:
      - name: rdf_lig_water
        g1: LIG_O
        g2: WATERS
        nbins: 100
        range: [0.0, 12.0]

  clustering:
    enabled: false
    selection: "ACTIVE_SITE and not name H*"
    method: hierarchical      # hierarchical | kmeans
    n_clusters: 3
    linkage: average
    max_frames: 2000          # subsample above this (pairwise RMSD is O(N^2))
    write_structures: true    # PDB of the representative frame of each cluster
    write_selection: null     # null = whole system (waters included);
                              # e.g. "byres (around 6 resname LIG)" to trim it

report:
  enabled: true
  formats: [markdown, html]
  title: "Reaction geometry analysis"
"""

DATA_CONFIG = """\
# =============================================================================
# md_interactions - analysis of PRE-COMPUTED tables (no topology, no trajectory)
#
#   md-analyzer check -c config_data.yaml
#   md-analyzer run   -c config_data.yaml
#
# Input: cpptraj-style files (one row per frame), e.g.
#     #Frame     D20:OD1-R32:HH12   D20:OD2-R32:HH22
#          1               2.5962             2.3542
# Useful for Monte Carlo runs, for data measured elsewhere, or to re-plot old
# results in the same style as the rest of the figures.
# =============================================================================

data:
  # Independent runs: keep them apart instead of pooling them blindly.
  replicas:
    rep1: [dist_rep1.dat]
    rep2: [dist_rep2.dat]
    rep3: [dist_rep3.dat]
  # ...or a single set of files analysed together:
  #   files: [dist_all.dat]

  unit: "Å"                   # unit of the columns, used in the axis labels
  time:
    dt: null                  # time between rows; null -> x axis is the frame
    unit: ps
  frames:
    start: 0
    stop: null
    stride: 1

  columns: []                 # [] = every column; or a subset by name
  labels:                     # optional renaming (the new name is used everywhere)
    "D20:OD1-R32:HH12": d_saltbridge_OD1
  thresholds:                 # optional: % of frames below the value
    d_saltbridge_OD1: 2.5

  bins: 30
  kde: true
  facet: true                 # one multi-panel figure with every column
  normalization: density      # density (Å⁻¹) | percent | counts
  timeseries: true            # also plot each column against time/frame

output:
  directory: results
  formats: [png, pdf]
  dpi: 300

analyses:
  free_energy_maps:           # 2D maps work on the columns of the table
    enabled: true
    temperature: 300.0
    maps:
      - name: fes_od1_od2
        x: d_saltbridge_OD1
        y: "D20:OD2-R32:HH22"
        bins: 40
        free_energy: true
        max_energy: 3.0

report:
  enabled: true
  formats: [markdown, html]
  title: "Distance distributions"
"""
