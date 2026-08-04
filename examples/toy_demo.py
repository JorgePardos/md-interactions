"""Demostración completa sin necesidad de datos propios.

Genera un sistema de juguete (tríada Ser-His-Asp + ligando + aguas), lo escribe
en disco y lanza todos los análisis, dejando el resultado en ``demo_results/``.

    python examples/toy_demo.py
"""

from __future__ import annotations

from pathlib import Path

import md_interactions as mdi
from md_interactions.testing import write_toy_system

HERE = Path(__file__).parent
WORKDIR = HERE / "demo_data"


def main() -> None:
    topology, trajectory = write_toy_system(WORKDIR, n_frames=200)
    print(f"Sistema de juguete escrito en {WORKDIR}")

    config = mdi.Config.from_dict({
        "system": {
            "topology": str(topology),
            "trajectory": [str(trajectory)],
            "time": {"dt": 0.01, "unit": "ns"},
        },
        "output": {"directory": str(HERE / "demo_results"), "formats": ["png", "pdf"]},
        "selections": {
            "SER_OG": "resid 1 and name OG",
            "HIS_NE2": "resid 2 and name NE2",
            "ASP_OD2": "resid 3 and name OD2",
            "LIG_C1": "resname LIG and name C1",
            "LIG_O1": "resname LIG and name O1",
            "WATERS": "resname WAT and name O",
        },
        "analyses": {
            "distances": {"pairs": [
                {"name": "d_nuc", "atoms": ["SER_OG", "LIG_C1"], "threshold": 3.2},
                {"name": "d_acid", "atoms": ["SER_OG", "HIS_NE2"]},
                {"name": "d_wat", "atoms": ["LIG_C1", "WATERS"], "mode": "min"},
            ]},
            "angles": {"definitions": [
                {"name": "a_attack", "atoms": ["SER_OG", "LIG_C1", "LIG_O1"],
                 "label": "OG–C1=O1 (Bürgi–Dunitz)"},
            ]},
            "rmsd": {"groups": [
                {"name": "backbone", "selection": "backbone"},
                {"name": "ligand", "selection": "resname LIG",
                 "superposition": "backbone"},
            ]},
            "rmsf": {"enabled": True, "selection": "protein and name CA"},
            "hbonds": {"pairs": [
                {"name": "hb_ser_his", "donor": "SER_OG", "acceptor": "HIS_NE2"},
                {"name": "hb_his_asp", "donor": "HIS_NE2", "acceptor": "ASP_OD2"},
            ]},
            "free_energy_maps": {"maps": [
                {"name": "fes_dnuc_dacid", "x": "d_nuc", "y": "d_acid", "bins": 40,
                 "max_energy": 4.0},
                {"name": "fes_dnuc_attack", "x": "d_nuc", "y": "a_attack", "bins": 40},
            ]},
            "radius_of_gyration": {"groups": [
                {"name": "rg_protein", "selection": "protein"},
            ]},
            "rdf": {"pairs": [
                {"name": "rdf_lig_wat", "g1": "LIG_O1", "g2": "WATERS",
                 "nbins": 60, "range": [0.0, 12.0]},
            ]},
            "clustering": {"enabled": True, "selection": "protein and not name H*",
                           "n_clusters": 3},
        },
        "report": {"enabled": True, "title": "Demo: sistema de juguete"},
    })

    output = mdi.run_analyses(config)
    print("\nResumen:")
    print(output.summary.to_string(index=False))
    print(f"\nInforme: {output.paths.root / 'report.html'}")


if __name__ == "__main__":
    main()
