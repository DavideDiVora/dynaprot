"""Structure parsing and reference-frame utilities.

PDB parsing (a modified copy of OpenFold's ``from_pdb_string``), Kabsch alignment, and the
local <-> global frame mapping that DynaProt's covariance outputs must pass through: the model
predicts in each residue's local backbone frame, so anything compared against MD -- or composed
into the joint covariance -- has to be rotated into the global frame first.
"""
import io
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from Bio.PDB import PDBParser

from dynaprot.utils.openfold import residue_constants
from dynaprot.utils.openfold.protein import Protein
from dynaprot.utils.openfold.rigid_utils import Rigid

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.
PICO_TO_ANGSTROM = 0.01

PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)
assert PDB_MAX_CHAINS == 62

__all__ = [
    "from_pdb_string",
    "kabsch_rmsd",
    "map_one_protein_global_frame",
    "map_one_protein_local_frame",
    "rmsdalign",
]

def find_rot_trans(a, b, weights=None):
    B = a.shape[:-2]
    N = a.shape[-2]
    if weights == None:
        weights = a.new_ones(*B, N)
    weights = weights.unsqueeze(-1)
    a_mean = (a * weights).sum(-2, keepdims=True) / weights.sum(-2, keepdims=True)
    a = a - a_mean
    b_mean = (b * weights).sum(-2, keepdims=True) / weights.sum(-2, keepdims=True)
    b = b - b_mean
    B = torch.einsum('...ji,...jk->...ik', weights * a, b)
    u, s, vh = torch.linalg.svd(B)

    # Correct improper rotation if necessary (as in Kabsch algorithm)
    '''
    if torch.linalg.det(u @ vh) < 0:
        s[-1] = -s[-1]
        u[:, -1] = -u[:, -1]
    '''
    sgn = torch.sign(torch.linalg.det(u @ vh))
    s[...,-1] *= sgn
    u[...,:,-1] *= sgn.unsqueeze(-1)
    C = u @ vh # c rotates B to A
    return C.mT,a_mean,b_mean

#https://github.com/scipy/scipy/blob/main/scipy/spatial/transform/_rotation.pyx


def rmsdalign(a, b, weights=None): # alignes B to A  # [*, N, 3]
    B = a.shape[:-2]
    N = a.shape[-2]
    if weights == None:
        weights = a.new_ones(*B, N)
    weights = weights.unsqueeze(-1)
    a_mean = (a * weights).sum(-2, keepdims=True) / weights.sum(-2, keepdims=True)
    a = a - a_mean
    b_mean = (b * weights).sum(-2, keepdims=True) / weights.sum(-2, keepdims=True)
    b = b - b_mean
    B = torch.einsum('...ji,...jk->...ik', weights * a, b)
    u, s, vh = torch.linalg.svd(B)

    # Correct improper rotation if necessary (as in Kabsch algorithm)
    '''
    if torch.linalg.det(u @ vh) < 0:
        s[-1] = -s[-1]
        u[:, -1] = -u[:, -1]
    '''
    sgn = torch.sign(torch.linalg.det(u @ vh))
    s[...,-1] *= sgn
    u[...,:,-1] *= sgn.unsqueeze(-1)
    C = u @ vh # c rotates B to A
    return b @ C.mT + a_mean


def kabsch_rmsd(a, b, weights=None):    # from AlphaFlow
    B = a.shape[:-2]
    N = a.shape[-2]
    if weights == None:
        weights = a.new_ones(*B, N)
    b_aligned = rmsdalign(a, b, weights)
    out = torch.square(b_aligned - a).sum(-1)
    out = (out * weights).sum(-1) / weights.sum(-1)
    return torch.sqrt(out)


def from_pdb_string(pdb_str: str, chain_id: Optional[str] = None) -> dict:
    """Takes a single chain PDB string and constructs a dictionary of protein chain features.
        Modifies the original method from openfold.np.protein

    WARNING: All non-standard residue types will be converted into UNK. All
      non-standard atoms will be ignored.

    Args:
      pdb_str: The contents of the pdb file
      chain_id: If None, then the whole pdb file is parsed. If chain_id is specified (e.g. A), then only that chain
        is parsed.

    Returns:
      dictionary of protein features
    """
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb_fh)
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(
            f"Only single model PDBs are supported. Found {len(models)} models."
        )
    model = models[0]

    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []

    for chain in model:
        if(chain_id is not None and chain.id != chain_id):
            continue

        for res in chain:
            if res.id[2] != " ":
                raise ValueError(
                    f"PDB contains an insertion code at chain {chain.id} and residue "
                    f"index {res.id[1]}. These are not supported."
                )
            res_shortname = residue_constants.restype_3to1.get(res.resname, "X")
            restype_idx = residue_constants.restype_order.get(
                res_shortname, residue_constants.restype_num
            )
            pos = np.zeros((residue_constants.atom_type_num, 3))
            mask = np.zeros((residue_constants.atom_type_num,))
            res_b_factors = np.zeros((residue_constants.atom_type_num,))
            for atom in res:
                if atom.name not in residue_constants.atom_types:
                    continue
                pos[residue_constants.atom_order[atom.name]] = atom.coord
                mask[residue_constants.atom_order[atom.name]] = 1.0
                res_b_factors[
                    residue_constants.atom_order[atom.name]
                ] = atom.bfactor
            if np.sum(mask) < 0.5:
                # If no known atom positions are reported for the residue then skip it.
                continue

            aatype.append(restype_idx)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1])
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)

    # parents = None
    # parents_chain_index = None
    # if("PARENT" in pdb_str):
    #     parents = []
    #     parents_chain_index = []
    #     chain_id = 0
    #     for l in pdb_str.split("\n"):
    #         if("PARENT" in l):
    #             if(not "N/A" in l):
    #                 parent_names = l.split()[1:]
    #                 parents.extend(parent_names)
    #                 parents_chain_index.extend([
    #                     chain_id for _ in parent_names
    #                 ])
    #             chain_id += 1

    # unique_chain_ids = np.unique(chain_ids)
    # chain_id_mapping = {cid: n for n, cid in enumerate(string.ascii_uppercase)}
    # chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

    return dict(
        aatype= np.array(aatype),
        all_atom_positions= np.array(atom_positions),
        all_atom_mask=np.array(atom_mask),
        residue_index=np.array(residue_index),
        # chain_index=chain_index,
        b_factors=np.array(b_factors),
        # parents=parents,
        # parents_chain_index=parents_chain_index,
    )


def map_one_protein_local_frame(prot_frames, prot_covars, prot_fullcovar = None):
    """Maps the per residue covariance matrices of a single protein to its local frames."""
    # prot_frames, prot_covars, prot_fullcovar = feats["frames"].double(), feats["dynamics_covars"].double(), feats["dynamics_fullcovar"].double()
    
    # print(prot_frames.shape[0], prot_covars.shape[0])
    assert prot_frames.shape[0] == prot_covars.shape[0]
    # rotations = Rigid.from_tensor_4x4(prot_frames).get_rots().get_rot_mats().double()  # Extract rotation matrices
    rotations  =  prot_frames[..., :3, :3]
    local_covars = torch.einsum("nij,njk,nlk->nil", rotations, prot_covars, rotations)
    # feats["dynamics_covars"] = local_covars

    # align full covar
    local_fullcovar = None
    if prot_fullcovar is not None:
        R_block = torch.block_diag(*[rotations[i] for i in range(rotations.shape[0])])
        local_fullcovar = R_block @ prot_fullcovar @ R_block.transpose(-1, -2)
    # feats["dynamics_fullcovar"] = local_fullcovar

    # return feats
    return local_covars, local_fullcovar


def map_one_protein_global_frame(prot_frames, prot_covars, prot_fullcovar = None):
    """Maps the per residue covariance matrices of a single protein to its global frames."""
    # prot_frames, prot_covars, prot_fullcovar = feats["frames"].double(), feats["dynamics_covars"].double(), feats["dynamics_fullcovar"].double()
    
    # print(prot_frames.shape[0], prot_covars.shape[0])
    assert prot_frames.shape[0] == prot_covars.shape[0]
    # rotations = Rigid.from_tensor_4x4(prot_frames).get_rots().get_rot_mats().double()  # Extract rotation matrices
    rotations  =  prot_frames[..., :3, :3]
    global_covars = torch.einsum("nji,njk,nkl->nil", rotations, prot_covars, rotations)
    # feats["dynamics_covars"] = global_covars

    # align full covar
    global_fullcovar = None
    if prot_fullcovar is not None:
        R_block = torch.block_diag(*[rotations[i] for i in range(rotations.shape[0])])
        global_fullcovar = R_block.transpose(-1, -2) @ prot_fullcovar @ R_block
    # feats["dynamics_fullcovar"] = global_fullcovar
    # return feats
    return global_covars, global_fullcovar
