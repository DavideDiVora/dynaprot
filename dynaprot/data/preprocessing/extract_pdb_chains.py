"""Split PDB mmCIF entries into per-chain .pdb/.cif files.

Run as a script:

    python -m dynaprot.data.preprocessing.extract_pdb_chains \
        --mmcif-path /path/to/pdb_mmcif --chain-path /path/to/pdb_chains
"""
import argparse
import multiprocessing
import os

import numpy as np
from Bio import PDB
from rich.console import Console
from rich.progress import Progress

console = Console()

PROTEIN_LISTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "protein_lists")

# Set by main(); module-level so pool workers inherit them.
mmcif_path = None
chain_path = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split PDB mmCIF entries into per-chain pdb/cif files."
    )
    parser.add_argument(
        "--mmcif-path", required=True, help="Directory of mmCIF files, sharded as <xx>/<id>.cif"
    )
    parser.add_argument(
        "--chain-path", required=True, help="Output directory, sharded as <xx>/<id>_<chain>.pdb"
    )
    parser.add_argument(
        "--protein-list",
        default=os.path.join(PROTEIN_LISTS, "atlas_proteins.npy"),
        help="Path to a .npy list of protein ids (default: the ATLAS list)",
    )
    parser.add_argument("--num-workers", type=int, default=40)
    return parser.parse_args()


def parse_one_protein(protein_id):
    if "_" in protein_id:
        protein_name, chain = protein_id.split("_")
    else:
        protein_name = protein_id
    parser = PDB.MMCIFParser(QUIET=True)
    structure = parser.get_structure(
        "protein", os.path.join(mmcif_path, f"{protein_name[1:3]}/{protein_name}.cif")
    )
    # Create an instance of PDBIO to save chains separately if needed
    pdb_io = PDB.PDBIO()
    mmcif_io = PDB.MMCIFIO()

    # Iterate through models and chains
    for model in structure:
        for chain in model:
            chain_id = chain.get_id()

            new_structure = PDB.Structure.Structure(chain_id)
            new_model = PDB.Model.Model(0)
            new_model.add(chain.copy())
            new_structure.add(new_model)

            pdb_io.set_structure(new_structure)
            pdb_io.save(os.path.join(chain_path, f"{protein_id[1:3]}/{protein_name}_{chain_id}.pdb"))
            mmcif_io.set_structure(new_structure)
            mmcif_io.save(os.path.join(chain_path, f"{protein_id[1:3]}/{protein_name}_{chain_id}.cif"))

    return protein_id


def main():
    global mmcif_path, chain_path
    args = parse_args()
    mmcif_path = args.mmcif_path
    chain_path = args.chain_path

    proteins = np.load(args.protein_list)
    total_chains = len(proteins)

    with Progress() as progress:
        task = progress.add_task(
            f"[cyan]Processing chains (0/{total_chains})...", total=total_chains
        )
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            for i, protein in enumerate(pool.imap(parse_one_protein, proteins)):
                progress.update(
                    task,
                    advance=1,
                    description=f"[cyan]Processing chains ({i}/{total_chains})... Completed {protein}",
                )


if __name__ == "__main__":
    main()
