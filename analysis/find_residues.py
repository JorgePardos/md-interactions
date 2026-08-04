"""Localiza el ARG/LYS correcto cuando el resid nominal no coincide.

Uso:
    python find_residues.py <ruta_al_prmtop>

Busca, alrededor de cada resid que falló en 'check' (265, 55, 36), el residuo
del tipo esperado (ARG o LYS) más cercano en la topología real, para poder
corregir las selecciones de qmmm_distances.yaml sin adivinar.
"""

import sys
import warnings

warnings.filterwarnings("ignore")
import MDAnalysis as mda

TOPOLOGY = sys.argv[1] if len(sys.argv) > 1 else None
if not TOPOLOGY:
    raise SystemExit("Uso: python find_residues.py <ruta_al_prmtop>")

u = mda.Universe(TOPOLOGY)

# (resid nominal que falló, tipo de residuo esperado, ventana de búsqueda)
TARGETS = [
    (265, "ARG", 15),   # TRH:O4P/O3P - R265:HH21/HE
    (55, "LYS", 15),    # 89:OE2 - K55:NZ
    (36, "LYS", 15),    # 114:OE - K36:NZ
]

for nominal, resname, window in TARGETS:
    print(f"\n=== resid {nominal} esperado {resname} ===")
    around = u.select_atoms(f"resid {nominal - window}-{nominal + window}")
    print("  contexto (resid, resname):")
    for r in around.residues:
        marker = "  <-- coincide en tipo" if r.resname == resname else ""
        here = " (resid nominal)" if r.resid == nominal else ""
        print(f"    {r.resid:5d} {r.resname:4s}{here}{marker}")

    candidates = [r for r in around.residues if r.resname == resname]
    if candidates:
        closest = min(candidates, key=lambda r: abs(r.resid - nominal))
        print(f"  >>> candidato mas cercano: resid {closest.resid} "
              f"(distancia en numeracion: {closest.resid - nominal:+d})")
    else:
        print(f"  >>> ningun {resname} encontrado en ese rango; "
              f"prueba a ampliar 'window' en el script")

print("\n--- todos los ARG de la proteina (por si el candidato no aparece arriba) ---")
print([int(r.resid) for r in u.select_atoms("resname ARG").residues])
print("\n--- todos los LYS de la proteina ---")
print([int(r.resid) for r in u.select_atoms("resname LYS").residues])
