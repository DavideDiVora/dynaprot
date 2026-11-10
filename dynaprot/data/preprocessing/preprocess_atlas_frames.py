"""Augmentation frames of DynaProt from the all-atom ATLAS trajectories.

With `augmented: True` in the data config, the input structure of a training protein is a random MD frame,
read from <labels>/<prot>/frames/frame_*.pt (backbone rigid tensors). This module writes those files as
preprocess_atlas does: the three replicas <inpath>/<prot>/<prot>_prod_R{1,2,3}_fit.xtc joined, every frame
superposed on the PDB <inpath>/<prot>/<prot>.pdb (load_superposed_trajectory), and every stride-th frame of the
joined trajectory saved by save_frame under frame_file_name(i), i the index in the joined trajectory
(save_separate_frames).

Two ways of reading, with the same frames:
- full: the original code path, load_superposed_trajectory then save_separate_frames(prot, traj, stride);
  it holds the whole joined trajectory in memory (all atoms, all frames).
- strided (default): every replica is read only at the frames whose index in the joined trajectory is a
  multiple of stride, each frame superposed on the PDB and saved by save_frame under the same name. The
  superposition acts frame by frame, so the frames are the same as with full reading.

Frames are written to <outpath>/<prot>/frames_partial/, renamed to <outpath>/<prot>/frames/ when complete:
a protein with a frames/ folder is complete and skipped. The dataset picks any file in frames/, so nothing
else is written there.

Usage:
    python -m dynaprot.data.preprocessing.preprocess_atlas_frames --inpath <archive with pdb and xtc>
        --outpath <labels> --proteins <list .npy or .txt> [--chunk K --n_chunks N] [--stride 100]
        [--load strided|full] [--processes 1]
"""

import argparse
import multiprocessing
import os
import shutil
import time

import mdtraj as md
import numpy as np

from dynaprot.data.preprocessing.preprocess_atlas import (
    frame_file_name,
    load_superposed_trajectory,
    save_frame,
    save_separate_frames,
)

N_REPLICAS = 3


def read_proteins(path):
    """Chains of a protein list: a .npy array (the protein_lists of the package) or a text file, one per line."""
    if path.endswith(".npy"):
        return [str(p) for p in np.load(path)]
    return [line.strip() for line in open(path) if line.strip()]


def strided_superposed_trajectory(inpath, prot, stride):
    """(traj, numbers): the frames of the joined trajectory of load_superposed_trajectory whose index is a
    multiple of stride, read replica by replica without loading the others, each superposed on the PDB;
    numbers[k] is the index in the joined trajectory of traj[k]."""
    pdb_path = os.path.join(inpath, prot, prot + ".pdb")
    ref = md.load(pdb_path)
    parts, numbers, start = [], [], 0
    for r in range(1, N_REPLICAS + 1):
        with md.formats.XTCTrajectoryFile(os.path.join(inpath, prot, f"{prot}_prod_R{r}_fit.xtc"), "r") as f:
            n_frames = len(f)
            first = (-start) % stride   # first frame of this replica at a multiple of stride in the joined index
            if first < n_frames:
                f.seek(first)
                part = f.read_as_traj(ref.topology, stride=stride)
                part.superpose(ref)
                parts.append(part)
                numbers.extend(range(start + first, start + n_frames, stride))
        start += n_frames
    traj = md.join(parts)
    if len(traj) != len(numbers):
        raise RuntimeError(f"{prot}: read {len(traj)} frames, expected {len(numbers)}")
    return traj, numbers


def write_frames(prot, inpath, outpath, stride, load="strided"):
    """Frames of one protein to <outpath>/<prot>/frames/; returns the number of frames written (0 if the
    protein was already complete)."""
    final = os.path.join(outpath, prot, "frames")
    if os.path.isdir(final):
        return 0
    partial = os.path.join(outpath, prot, "frames_partial")
    if os.path.isdir(partial):
        shutil.rmtree(partial)
    os.makedirs(partial)
    if load == "full":
        traj, _ = load_superposed_trajectory(inpath, prot)
        save_separate_frames(prot, traj, stride, partial)
        n_written = len(range(0, len(traj), stride))
    elif load == "strided":
        traj, numbers = strided_superposed_trajectory(inpath, prot, stride)
        for k, i in enumerate(numbers):
            save_frame(traj[k], os.path.join(partial, frame_file_name(i)))
        n_written = len(numbers)
    else:
        raise ValueError(f"unknown load {load!r}")
    os.rename(partial, final)
    return n_written


def _worker(args):
    prot, inpath, outpath, stride, load = args
    tic = time.time()
    try:
        n = write_frames(prot, inpath, outpath, stride, load)
        return prot, n, time.time() - tic, None
    except Exception as err:  # reported, the other proteins go on
        return prot, 0, time.time() - tic, repr(err)


def main():
    parser = argparse.ArgumentParser(description="DynaProt augmentation frames from the ATLAS all-atom xtc.")
    parser.add_argument("--inpath", required=True, help="archive with <prot>/<prot>.pdb and <prot>_prod_R{1,2,3}_fit.xtc")
    parser.add_argument("--outpath", required=True, help="labels directory; frames go to <outpath>/<prot>/frames/")
    parser.add_argument("--proteins", required=True, help="protein list, .npy or text file with one chain per line")
    parser.add_argument("--chunk", type=int, default=0, help="this chunk, 0 <= chunk < n_chunks (e.g. SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--n_chunks", type=int, default=1, help="number of chunks; chunk k takes proteins k, k + n_chunks, ...")
    parser.add_argument("--stride", type=int, default=100, help="stride over the joined trajectory")
    parser.add_argument("--load", choices=["strided", "full"], default="strided")
    parser.add_argument("--processes", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.chunk < args.n_chunks:
        raise SystemExit(f"--chunk {args.chunk} outside [0, {args.n_chunks})")
    proteins = read_proteins(args.proteins)[args.chunk::args.n_chunks]
    os.makedirs(args.outpath, exist_ok=True)
    print(f"chunk {args.chunk}/{args.n_chunks}: {len(proteins)} proteins, stride {args.stride}, load {args.load}, "
          f"{args.processes} processes", flush=True)
    tasks = [(p, args.inpath, args.outpath, args.stride, args.load) for p in proteins]
    failed = []
    with multiprocessing.Pool(processes=args.processes) as pool:
        for i, (prot, n, seconds, error) in enumerate(pool.imap_unordered(_worker, tasks)):
            if error:
                failed.append(prot)
                status = f"failed, {error}"
            else:
                status = f"{n} frames" if n else "complete, skipped"
            print(f"[{i + 1}/{len(proteins)}] {prot}: {status} ({seconds:.1f} s)", flush=True)
    print(f"done: {len(proteins) - len(failed)} ok, {len(failed)} failed {failed}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
