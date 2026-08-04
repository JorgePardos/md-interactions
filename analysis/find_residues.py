"""Locate the right ARG/LYS when the nominal resid does not match.

Usage:
    python find_residues.py <ruta_al_prmtop>

Around each resid that failed in 'check' (265, 55, 36), find the nearest
residue of the expected type (ARG or LYS) in the real topology, so that
the selections of qmmm_distances.yaml can be fixed without guessing.
"""

import sys
import warnings

warnings.filterwarnings("ignore")
import MDAnalysis as mda

TOPOLOGY = sys.argv[1] if len(sys.argv) > 1 else None
if not TOPOLOGY:
    raise SystemExit("Usage: python find_residues.py <ruta_al_prmtop>")

u = mda.Universe(TOPOLOGY)

# (nominal resid that failed, expected residue type, search window)
TARGETS = [
    (265, "ARG", 15),   # TRH:O4P/O3P - R265:HH21/HE
    (55, "LYS", 15),    # 89:OE2 - K55:NZ
    (36, "LYS", 15),    # 114:OE - K36:NZ
]

for nominal, resname, window in TARGETS:
    print(f"\n=== resid {nominal} expected {resname} ===")
    around = u.select_atoms(f"resid {nominal - window}-{nominal + window}")
    print("  context (resid, resname):")
    for r in around.residues:
        marker = "  <-- type matches" if r.resname == resname else ""
        here = " (nominal resid)" if r.resid == nominal else ""
        print(f"    {r.resid:5d} {r.resname:4s}{here}{marker}")

    candidates = [r for r in around.residues if r.resname == resname]
    if candidates:
        closest = min(candidates, key=lambda r: abs(r.resid - nominal))
        print(f"  >>> nearest candidate: resid {closest.resid} "
              f"(numbering distance: {closest.resid - nominal:+d})")
    else:
        print(f"  >>> no {resname} found in that range; "
              f"try widening 'window' in the script")

print("\n--- every ARG in the protein (in case the candidate is not listed above) ---")
print([int(r.resid) for r in u.select_atoms("resname ARG").residues])
print("\n--- every LYS in the protein ---")
print([int(r.resid) for r in u.select_atoms("resname LYS").residues])
