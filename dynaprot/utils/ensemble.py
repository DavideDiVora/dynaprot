"""Sampling conformational ensembles and writing them to disk.

Given the 3N x 3N joint covariance from :mod:`dynaprot.descriptors` and the input structure's
Calpha coordinates as the mean, :func:`sample_ensemble` draws conformations via the
reparameterization trick. The writers emit multi-model Calpha PDB or XTC, with optional
all-atom reconstruction through PULCHRA.
"""
import os
import shutil
import string
import subprocess
import time
from pathlib import Path
from typing import Optional

import mdtraj as md
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from dynaprot.utils.secondary_structure import assign_secondary_structures

__all__ = [
    "reconstruct_ensemble_with_pulchra",
    "sample_ensemble",
    "save_ensemble_with_ss",
    "save_ensemble_xtc",
]

def sample_ensemble(mean_3n, cov_3n, num_samples=10, timed=True):
    start = time.time() if timed else None

    device = cov_3n.device
    dtype = cov_3n.dtype
    N = cov_3n.shape[0] // 3

    if mean_3n is None:
        mean_3n = torch.zeros(3 * N, device=device, dtype=dtype)

    mvn = MultivariateNormal(loc=mean_3n, covariance_matrix=cov_3n)
    samples = mvn.rsample((num_samples,))
    samples = samples.view(num_samples, N, 3)

    if timed:
        elapsed = time.time() - start
        return samples, elapsed
    else:
        return samples


def save_ensemble_xtc(samples, ref_pdb_path, xtc_path):
    """
    Save a C-alpha ensemble as an MDTraj XTC file.

    Args:
        samples: torch.Tensor or np.ndarray of shape (N, L, 3)
                 coordinates in Å
        ref_pdb_path: path to the reference pdb (for topology)
        xtc_path: output path ending in .xtc
    """

    # Load reference PDB to get topology and CA indexing
    ref = md.load(ref_pdb_path)
    ca_indices = ref.top.select("name CA")

    # Slice reference trajectory down to CA-only to get the right topology
    ref_ca = ref.atom_slice(ca_indices)
    topo = ref_ca.top

    # Convert samples to numpy
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()

    assert samples.ndim == 3, f"Expected (N, L, 3), got {samples.shape}"

    # MDTraj expects coordinates in nanometers, so convert Å → nm
    coords_nm = samples * 0.1

    # Build a new trajectory
    traj = md.Trajectory(coords_nm, topology=topo)

    # Save XTC file
    traj.save_xtc(xtc_path)


def save_ensemble_with_ss(
    coord_ensemble: torch.Tensor,
    ref_pdb_path: str,
    out_pdb_path: str,
    assign_ss_fn = assign_secondary_structures,  # Your `assign_secondary_structures` function
    chain: Optional[str] = None,
):
    """
    Save an ensemble of structures using only Cα atoms and annotate secondary structure using a custom assignment function.

    Args:
        coord_ensemble (torch.Tensor): Tensor of shape (num_samples, N, 3), in Ångström.
        ref_pdb_path (str): Path to a reference PDB file with correct topology.
        out_pdb_path (str): Output path for the ensemble PDB.
        assign_ss_fn (function): Function to assign secondary structure from coordinates.
        chain (str, optional): Restrict the Cα selection to this chain. Defaults to None,
            i.e. every Cα in the reference.
    """
    coord_ensemble = coord_ensemble.cpu().numpy()
    num_models, N, _ = coord_ensemble.shape

    ref = md.load_pdb(ref_pdb_path)[0]
    ss_assignments =list( md.compute_dssp(ref)[0])

    # Only use Cα atoms
    selection = "name CA" if chain is None else f"name CA and chainid {chain}"
    ca_indices = ref.topology.select(selection)
    ref_calpha = ref.atom_slice(ca_indices)
    residues = list(ref_calpha.topology.residues)

    if len(ca_indices) != coord_ensemble.shape[1]:
        raise ValueError(f"Reference has {len(ca_indices)} Cα atoms, but ensemble has {coord_ensemble.shape[1]} residues")

    lines = []

    # Write HELIX/SHEET records manually for model 1
    current_ss = None
    start = None

    for i, ss in enumerate(ss_assignments + ["END"]):  # Add sentinel
        if ss != current_ss:
            if current_ss in ["H", "E"]:
                record_type = "HELIX" if current_ss == "H" else "SHEET"
                end = i
                res_start = residues[start].resSeq
                res_end   = residues[end - 1].resSeq
                chain_id = string.ascii_uppercase[residues[start].chain.index] 
                lines.append(f"{record_type:<6}    1 {chain_id} {res_start:>4}    {chain_id} {res_end:>4}")
            current_ss = ss
            start = i

    # Write each model
    atom_line_format = (
        "ATOM  {atom_idx:5d}  CA  {resname:>3} {chain_id}{resid:4d}    "
        "{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  "
    )

    resnames  = [res.name for res in residues]
    resids    = [res.resSeq for res in residues]
    chain_ids = [string.ascii_uppercase[res.chain.index]   for res in residues]
    
    for model_idx in range(num_models):
        lines.append(f"MODEL     {model_idx + 1}")
        for i in range(N):
            coord = coord_ensemble[model_idx, i]
            lines.append(atom_line_format.format(
                atom_idx=i + 1,
                resname=resnames[i],
                chain_id=chain_ids[i],
                resid=resids[i],
                x=coord[0].item(),
                y=coord[1].item(),
                z=coord[2].item()
            ))
        lines.append("ENDMDL")

    lines.append("END")

    with open(out_pdb_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def split_models(input_pdb, temp_dir):
    """Split multi-model PDB into individual models."""
    with open(input_pdb, 'r') as f:
        lines = f.readlines()

    models = []
    current_model = []
    for line in lines:
        if line.startswith("MODEL"):
            current_model = []
        elif line.startswith("ENDMDL"):
            models.append(current_model[:])
        else:
            current_model.append(line)

    paths = []
    for i, model in enumerate(models):
        model_path = temp_dir / f"model_{i+1}.pdb"
        with open(model_path, 'w') as f:
            f.write("".join(model) + "END\n")
        paths.append(model_path)

    return paths


def run_pulchra(pdb_path):
    """Run Pulchra on a single Cα-only PDB."""
    result = subprocess.run(
        ["pulchra", str(pdb_path)],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Pulchra failed on {pdb_path}: {result.stderr}")
    return pdb_path.with_suffix(".rebuilt.pdb")


def merge_models(pdb_paths, out_path):
    """Merge single-model PDBs into a multi-model PDB."""
    with open(out_path, 'w') as out_f:
        for i, pdb in enumerate(pdb_paths):
            with open(pdb, 'r') as f:
                atoms = [l for l in f if l.startswith("ATOM")]
            out_f.write(f"MODEL     {i + 1}\n")
            out_f.writelines(atoms)
            out_f.write("ENDMDL\n")
        out_f.write("END\n")


def reconstruct_ensemble_with_pulchra(input_pdb, output_pdb, save_individual_models=False, individual_out_dir=None):
    temp_dir = Path("tmp_pulchra")
    temp_dir.mkdir(exist_ok=True)

    if save_individual_models:
        individual_out_dir = Path(individual_out_dir or output_pdb).with_suffix('')  # default to output_pdb stem
        individual_out_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"[INFO] Splitting ensemble: {input_pdb}")
        model_paths = split_models(input_pdb, temp_dir)

        print(f"[INFO] Running Pulchra on {len(model_paths)} models...")
        rebuilt_paths = []
        for i, p in enumerate(model_paths):
            rebuilt = run_pulchra(p)
            rebuilt_paths.append(rebuilt)

            if save_individual_models:
                dest = individual_out_dir / f"{Path(input_pdb).stem}_model_{i+1:03d}.pdb"
                shutil.copyfile(rebuilt, dest)

        print(f"[INFO] Merging rebuilt models into: {output_pdb}")
        merge_models(rebuilt_paths, output_pdb)

    finally:
        print("[INFO] Cleaning up temp files...")
        for f in temp_dir.glob("*"):
            f.unlink()
        temp_dir.rmdir()


# def save_ensemble(
#     coord_ensemble: torch.Tensor,
#     ref_pdb_path: str,
#     out_pdb_path: str,
#     transfer_secondary_struc: bool = True,
# ):
#     """
#     Save an ensemble of structures using only Cα atoms.

#     Args:
#         coord_ensemble (torch.Tensor): Tensor of shape (num_samples, N, 3), in Ångström.
#         ref_pdb_path (str): Path to a reference PDB file with correct topology.
#         out_pdb_path (str): Output path for the ensemble PDB.
#         transfer_secondary_struc (bool): Whether to assign secondary structure using DSSP from ref PDB.
#     """
#     coord_ensemble = coord_ensemble.cpu().numpy()
#     ref = md.load_pdb(ref_pdb_path)

#     ca_indices = ref.topology.select("name CA")
#     ref_calpha = ref.atom_slice(ca_indices)

#     if len(ca_indices) != coord_ensemble.shape[1]:
#         raise ValueError(f"Reference has {len(ca_indices)} Cα atoms, but ensemble has {coord_ensemble.shape[1]} residues")

#     # Save coordinates (first model only) with SS annotation
#     coords = coord_ensemble[0]

#     ss_annotation = assign_secondary_structures(torch.from_numpy(ref.xyz*10.0), return_encodings=False)  
#     print(ss_annotation)
#     # Build HELIX and SHEET lines
#     header_lines = []
#     if transfer_secondary_struc and ss_annotation:
#         def find_regions(label):
#             regions = []
#             start = None
#             for i, c in enumerate(ss_annotation):
#                 if c == label and start is None:
#                     start = i
#                 elif c != label and start is not None:
#                     if i - start >= 3:
#                         regions.append((start, i - 1))
#                     start = None
#             if start is not None and len(ss_annotation) - start >= 3:
#                 regions.append((start, len(ss_annotation) - 1))
#             return regions

#         helix_regions = find_regions('H')
#         sheet_regions = find_regions('E')

#         for i, (start, end) in enumerate(helix_regions, 1):
#             res_start = ref_calpha.topology.residue(start)
#             res_end = ref_calpha.topology.residue(end)
#             header_lines.append(
#                 f"HELIX  {i:>3}  H{i:>3} {res_start.name:>3} A{res_start.resSeq:>4}  {res_end.name:>3} A{res_end.resSeq:>4}  1 {end-start+1:>30}"
#             )

#         for i, (start, end) in enumerate(sheet_regions, 1):
#             res_start = ref_calpha.topology.residue(start)
#             res_end = ref_calpha.topology.residue(end)
#             header_lines.append(
#                 f"SHEET  {i:>3}  S{i:>3} 1 {res_start.name:>3} A{res_start.resSeq:>4}  {res_end.name:>3} A{res_end.resSeq:>4}  0"
#             )

#     # Save annotated PDB manually
#     os.makedirs(os.path.dirname(out_pdb_path), exist_ok=True)
#     with open(out_pdb_path, "w") as f:
#         for line in header_lines:
#             f.write(line + "\n")

#         for i, atom in enumerate(ref_calpha.topology.atoms):
#             x, y, z = coords[i]
#             f.write(
#                 f"ATOM  {i+1:5d}  CA  {atom.residue.name:>3} A{atom.residue.resSeq:>4}    "
#                 f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C\n"
#             )
#         f.write("END\n")
