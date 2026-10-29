"""ATLAS dynamics labels from C-alpha coordinates instead of the all-atom trajectories.

The same labels as preprocess_atlas.process_one_trajectory_atlas, for an archive that keeps, per
protein, only the PDB (<inpath>/<prot>/<prot>.pdb) and the C-alpha of the three replicas
(<inpath>/<prot>/<prot>_R{1,2,3}_ca.npy: float32 in Angstrom, [n_frames, N, 3], the rows in the order
of the PDB C-alpha, extracted from the ATLAS protein-only xtc).

The labels depend on the trajectories only through compute_gaussians, which keeps the C-alpha and
superposes them on the C-alpha of the PDB in every frame: the superposition of all atoms done first
by the original pipeline does not change the result, which is therefore the same up to float32
rounding (check_preprocess_atlas_ca.py compares the two on proteins with their xtc). Everything after
compute_gaussians is copied from process_one_trajectory_atlas unchanged.

Usage:
    python -m dynaprot.data.preprocessing.preprocess_atlas_ca --inpath <archive> --outpath <labels>
        [--proteins <list of chains, one per line>] [--processes 8]
"""

import argparse
import multiprocessing
import os

import mdtraj as md
import numpy as np
import torch

from dynaprot.data.preprocessing.preprocess_atlas import PROTEIN_LISTS, compute_gaussians
from dynaprot.descriptors import (
    downsample_covariance_to_nxn,
    project_to_psd,
    renormalize_to_unit_diag,
    rmsf_from_covariances as compute_rmsf_from_covariances,
    sum_inner_product_heuristic,
)
from dynaprot.evaluation.metrics import matrix_sqrt_eigen
from dynaprot.utils.openfold import data_transforms
from dynaprot.utils.openfold.data_transforms import np_to_tensor_dict
from dynaprot.utils.structure import from_pdb_string, map_one_protein_local_frame

NM_PER_ANGSTROM = 0.1


def ca_trajectory(inpath, prot):
    """(traj, ref): the three replicas joined as one C-alpha trajectory (in nm, as mdtraj) and the C-alpha
    of the PDB, with the same topology."""
    pdb_path = os.path.join(inpath, prot, prot + ".pdb")
    full = md.load(pdb_path)
    ref = full.atom_slice(full.top.select("name CA"))
    xyz = np.concatenate([np.load(os.path.join(inpath, prot, f"{prot}_R{r}_ca.npy")) for r in (1, 2, 3)])
    if xyz.shape[1] != ref.n_atoms:
        raise ValueError(f"{prot}: {xyz.shape[1]} C-alpha in the trajectories, {ref.n_atoms} in the PDB")
    traj = md.Trajectory((xyz * NM_PER_ANGSTROM).astype(np.float32), ref.top)
    return traj, ref


def labels(traj, ref, pdb_path):
    """The label dictionary of process_one_trajectory_atlas from a trajectory and its reference."""
    # generate feats and process them into dicts (as in process_one_trajectory_atlas)
    feats = from_pdb_string(open(pdb_path, 'r').read())
    feats = np_to_tensor_dict(feats, feats.keys())
    feats = data_transforms.atom37_to_frames(feats)
    feats = data_transforms.get_backbone_frames(feats)
    selected_feats = {k: feats[k] for k in ["aatype", "residue_index", "all_atom_positions", "all_atom_mask"]}
    selected_feats["frames"] = feats["backbone_rigid_tensor"]

    selected_feats["dynamics_means"], selected_feats["dynamics_covars_global"], selected_feats["dynamics_fullcovar_global"] = compute_gaussians(traj, ref)

    selected_feats["dynamics_covars_local"], selected_feats["dynamics_fullcovar_local"] = map_one_protein_local_frame(selected_feats["frames"].double(), selected_feats["dynamics_covars_global"].double(), selected_feats["dynamics_fullcovar_global"].double())
    selected_feats["dynamics_covars_local_sqrt"], selected_feats["dynamics_fullcovar_local_sqrt"] = matrix_sqrt_eigen(selected_feats["dynamics_covars_local"]), matrix_sqrt_eigen(selected_feats["dynamics_fullcovar_local"])

    assert torch.allclose(selected_feats["dynamics_covars_local_sqrt"] @ selected_feats["dynamics_covars_local_sqrt"], selected_feats["dynamics_covars_local"])
    assert torch.allclose(selected_feats["dynamics_fullcovar_local_sqrt"] @ selected_feats["dynamics_fullcovar_local_sqrt"], selected_feats["dynamics_fullcovar_local"])

    selected_feats["dynamics_rmsf"] = compute_rmsf_from_covariances(selected_feats["dynamics_covars_local"])
    selected_feats["dynamics_correlations_nbyncovar"] = downsample_covariance_to_nxn(selected_feats["dynamics_fullcovar_global"])
    selected_feats["dynamics_correlations_sum"] = renormalize_to_unit_diag(project_to_psd(sum_inner_product_heuristic(selected_feats["dynamics_fullcovar_global"])))
    return selected_feats


def process_one_protein(prot, inpath, outpath):
    """Labels of one protein from its C-alpha archive, saved to <outpath>/<prot>/<prot>.pt."""
    os.makedirs(os.path.join(outpath, prot), exist_ok=True)
    traj, ref = ca_trajectory(inpath, prot)
    torch.save(labels(traj, ref, os.path.join(inpath, prot, prot + ".pdb")), os.path.join(outpath, prot, f"{prot}.pt"))
    return prot


def _worker(args):
    prot, inpath, outpath = args
    try:
        process_one_protein(prot, inpath, outpath)
        return prot, None
    except Exception as err:  # reported, the other proteins go on
        return prot, repr(err)


def main():
    parser = argparse.ArgumentParser(description="ATLAS dynamics labels from C-alpha coordinates.")
    parser.add_argument("--inpath", required=True, help="archive with <prot>/<prot>.pdb and <prot>_R{1,2,3}_ca.npy")
    parser.add_argument("--outpath", required=True, help="output directory of the labels")
    parser.add_argument("--proteins", help="text file of chains, one per line (default: atlas_proteins.npy)")
    parser.add_argument("--processes", type=int, default=8)
    args = parser.parse_args()
    if args.proteins:
        proteins = [line.strip() for line in open(args.proteins) if line.strip()]
    else:
        proteins = [str(p) for p in np.load(os.path.join(PROTEIN_LISTS, "atlas_proteins.npy"))]
    os.makedirs(args.outpath, exist_ok=True)
    failed = {}
    with multiprocessing.Pool(processes=args.processes) as pool:
        for i, (prot, error) in enumerate(pool.imap(_worker, [(p, args.inpath, args.outpath) for p in proteins])):
            if error:
                failed[prot] = error
            print(f"[{i + 1}/{len(proteins)}] {prot}: {'failed, ' + error if error else 'ok'}", flush=True)
    print(f"done: {len(proteins) - len(failed)} proteins, {len(failed)} failed", flush=True)


if __name__ == "__main__":
    main()
