"""Check of preprocess_atlas_ca against the original pipeline, on proteins whose all-atom xtc are available.

For every protein: the labels of preprocess_atlas.process_one_trajectory_atlas from the ATLAS protein-only
files (<xtc_inpath>/<prot>/<prot>.pdb, <prot>_prod_R{1,2,3}_fit.xtc) and those of preprocess_atlas_ca from
the C-alpha archive (<ca_inpath>/<prot>/<prot>.pdb, <prot>_R{1,2,3}_ca.npy); the C-alpha of the archive are
first compared with those of the xtc, then every label tensor (maximum absolute and relative difference,
and torch.allclose with rtol and atol).

Usage:
    python -m dynaprot.data.preprocessing.check_preprocess_atlas_ca --xtc_inpath <dir> --ca_inpath <dir>
        --outpath <dir> --proteins 3mx7_A 4obi_A [--rtol 1e-4 --atol 1e-6]
"""

import argparse
import os

import mdtraj as md
import numpy as np
import torch

from dynaprot.data.preprocessing import preprocess_atlas, preprocess_atlas_ca


def main():
    parser = argparse.ArgumentParser(description="Check preprocess_atlas_ca against the original pipeline.")
    parser.add_argument("--xtc_inpath", required=True)
    parser.add_argument("--ca_inpath", required=True)
    parser.add_argument("--outpath", required=True)
    parser.add_argument("--proteins", nargs="+", required=True)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()
    original, adapted = os.path.join(args.outpath, "original"), os.path.join(args.outpath, "ca")
    os.makedirs(original, exist_ok=True)
    preprocess_atlas.inpath, preprocess_atlas.outpath = args.xtc_inpath, original
    all_ok = True
    for prot in args.proteins:
        pdb = os.path.join(args.xtc_inpath, prot, prot + ".pdb")
        ca = md.load(pdb).top.select("name CA")
        for r in (1, 2, 3):
            xtc = md.load_xtc(os.path.join(args.xtc_inpath, prot, f"{prot}_prod_R{r}_fit.xtc"), top=pdb, atom_indices=ca)
            npy = np.load(os.path.join(args.ca_inpath, prot, f"{prot}_R{r}_ca.npy"))
            print(f"{prot} R{r}: C-alpha of the archive equal to 10 x those of the xtc: "
                  f"{np.array_equal(npy, (xtc.xyz * 10.0).astype(np.float32))}", flush=True)
        preprocess_atlas.process_one_trajectory_atlas(prot)
        preprocess_atlas_ca.process_one_protein(prot, args.ca_inpath, adapted)
        a = torch.load(os.path.join(original, prot, f"{prot}.pt"), weights_only=False)
        b = torch.load(os.path.join(adapted, prot, f"{prot}.pt"), weights_only=False)
        if set(a) != set(b):
            print(f"{prot}: different keys {sorted(set(a) ^ set(b))}", flush=True)
            all_ok = False
        for key in sorted(a):
            x, y = torch.as_tensor(a[key]).double(), torch.as_tensor(b[key]).double()
            diff = (x - y).abs()
            rel = (diff / x.abs().clamp_min(args.atol)).max().item() if diff.numel() else 0.0
            ok = x.shape == y.shape and torch.allclose(x, y, rtol=args.rtol, atol=args.atol)
            all_ok &= ok
            print(f"{prot} {key:36s} shape {tuple(x.shape)}: max |diff| {diff.max().item() if diff.numel() else 0.0:.3e}, "
                  f"max relative {rel:.3e}, allclose {ok}", flush=True)
    print(f"all labels equal within rtol {args.rtol}, atol {args.atol}: {all_ok}", flush=True)


if __name__ == "__main__":
    main()
