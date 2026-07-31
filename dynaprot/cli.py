"""Command-line interface: ``dynaprot predict INPUT.pdb -o OUTDIR``."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

import dynaprot
from dynaprot.inference import (
    DEFAULT_JOINT,
    DEFAULT_MARGINAL,
    OUTPUT_TYPES,
    DynaProtPredictor,
    normalize_outputs,
    required_models,
    load_model,
    resolve_checkpoint_dir,
    write_ensemble,
    write_prediction_npz,
    write_rmsf_bfactor_pdb,
    write_rmsf_csv,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _add_predict_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "predict",
        help="predict dynamics descriptors (and optionally an ensemble) for a structure",
        description=(
            "Run DynaProt on a PDB file. Writes per-residue marginal covariances, RMSF, "
            "the residue-residue coupling matrix, and -- unless --num-samples 0 -- a "
            "sampled conformational ensemble."
        ),
    )
    p.add_argument("input", type=Path, help="input PDB file")
    p.add_argument(
        "--outputs",
        nargs="+",
        default=["all"],
        metavar="TYPE",
        choices=list(OUTPUT_TYPES) + ["all"],
        help=(
            "which descriptors to produce: " + ", ".join(OUTPUT_TYPES) + ", or all (default). "
            "Only the models actually needed are loaded and run -- e.g. --outputs rmsf skips "
            "DynaProt-J entirely. RMSF is derived from the marginals, so the marginals are "
            "always computed alongside it."
        ),
    )
    p.add_argument("-o", "--out", type=Path, required=True, help="output directory")
    p.add_argument("--chain", default=None, help="parse only this chain (required for multi-chain PDBs)")
    p.add_argument(
        "--marginal-ckpt",
        default=DEFAULT_MARGINAL,
        help="DynaProt-M checkpoint: a directory, a name under $DYNAPROT_CKPT_DIR, or 'none'",
    )
    p.add_argument(
        "--joint-ckpt",
        default=DEFAULT_JOINT,
        help="DynaProt-J checkpoint: a directory, a name under $DYNAPROT_CKPT_DIR, or 'none'",
    )
    p.add_argument(
        "-n", "--num-samples", type=int, default=250,
        help="conformations to sample (0 = descriptors only, no DynaProt-J needed)",
    )
    p.add_argument(
        "--format", default="pdb", choices=["pdb", "xtc", "both"],
        help="ensemble output format (default: pdb)",
    )
    p.add_argument(
        "--all-atom", action="store_true",
        help="reconstruct all-atom coordinates with PULCHRA (must be on PATH)",
    )
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for ensemble sampling")


def _add_visualize_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "visualize",
        help="render predicted marginal Gaussians as PyMOL ellipsoids",
        description=(
            "Turn predicted per-residue 3x3 covariances into ellipsoids for PyMOL. Each "
            "ellipsoid is the one-sigma surface of that residue's predicted positional "
            "distribution, so its size shows how much the residue moves and its shape shows "
            "in which direction."
        ),
    )
    p.add_argument(
        "prediction",
        type=Path,
        help="prediction.npz written by `dynaprot predict`, or the directory holding it",
    )
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="script to write (default: ellipsoids.py beside the prediction)")
    p.add_argument("--structure", type=Path, default=None,
                   help="PDB to show underneath the ellipsoids")
    p.add_argument("--scale", type=float, default=1.0,
                   help="scale the ellipsoid axes; 1.0 = one sigma (default)")
    p.add_argument("--transparency", type=float, default=0.3)
    p.add_argument("--resolution", type=int, default=12,
                   help="mesh subdivisions per angular direction (cost grows as resolution^2)")
    p.add_argument("--session", type=Path, default=None,
                   help="also drive PyMOL directly and save a .pse session (needs pymol importable)")
    p.add_argument("--image", type=Path, default=None,
                   help="also ray-trace a PNG (needs pymol importable)")


def cmd_visualize(args: argparse.Namespace) -> int:
    import numpy as np

    from dynaprot.visualization import render_ellipsoids, write_pymol_script

    npz_path = args.prediction / "prediction.npz" if args.prediction.is_dir() else args.prediction
    if not npz_path.is_file():
        print(f"error: no such file: {npz_path}", file=sys.stderr)
        return 1

    data = np.load(npz_path)
    if "marginal_covariances" not in data:
        print(
            f"error: {npz_path} has no 'marginal_covariances'. Re-run `dynaprot predict` "
            "with --outputs marginals (or all).",
            file=sys.stderr,
        )
        return 1

    out_path = args.out or npz_path.parent / "ellipsoids.py"
    structure_name = None
    if args.structure is not None:
        structure_name = (
            args.structure.name
            if args.structure.parent.resolve() == out_path.parent.resolve()
            else str(args.structure.resolve())
        )

    write_pymol_script(
        out_path,
        arrays_name=npz_path.name,
        structure=structure_name,
        scale=args.scale,
        transparency=args.transparency,
        resolution=args.resolution,
    )
    n = len(data["ca_coords"])
    print(f"[dynaprot] wrote {out_path} ({n} ellipsoids)")
    print(f"[dynaprot] view with:  pymol {args.structure or ''} {out_path}".replace("  ", " "))

    if args.session is not None or args.image is not None:
        render_ellipsoids(
            data["marginal_covariances"],
            data["ca_coords"],
            structure=args.structure,
            session_path=args.session,
            image_path=args.image,
            scale=args.scale,
            transparency=args.transparency,
            resolution=args.resolution,
        )
        for label, path in (("session", args.session), ("image", args.image)):
            if path is not None:
                print(f"[dynaprot] wrote {label}: {path}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    if not args.input.is_file():
        print(f"error: no such file: {args.input}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)

    outputs = normalize_outputs(args.outputs)
    if args.num_samples == 0:
        outputs = tuple(o for o in outputs if o != "ensemble")
    if not outputs:
        print("error: no outputs requested", file=sys.stderr)
        return 1

    want_marginal, want_joint = required_models(outputs)
    for name, ref, wanted in (("DynaProt-M", args.marginal_ckpt, want_marginal),
                              ("DynaProt-J", args.joint_ckpt, want_joint)):
        if wanted and str(ref).lower() == "none":
            print(
                f"error: outputs {list(outputs)} need {name}, but its checkpoint is disabled",
                file=sys.stderr,
            )
            return 1
    print(f"[dynaprot] outputs: {', '.join(outputs)}")

    checkpoints = {}
    models = {}
    for key, ref, wanted in (
        ("marginal", args.marginal_ckpt, want_marginal),
        ("joint", args.joint_ckpt, want_joint),
    ):
        if not wanted:
            models[key] = None
            continue
        ckpt_dir = resolve_checkpoint_dir(ref)
        ckpt_file = ckpt_dir / "model.ckpt" if ckpt_dir.is_dir() else ckpt_dir
        checkpoints[key] = {"path": str(ckpt_file), "sha256": _sha256(ckpt_file)}
        print(f"[dynaprot] loading {key} model from {ckpt_dir}")
        models[key] = load_model(ckpt_dir, device=args.device)

    predictor = DynaProtPredictor(
        marginal_model=models["marginal"], joint_model=models["joint"], device=args.device
    )

    print(f"[dynaprot] predicting on {args.input}" + (f" chain {args.chain}" if args.chain else ""))
    started = time.time()
    prediction = predictor.predict(args.input, chain_id=args.chain, outputs=outputs)
    print(f"[dynaprot] {prediction.num_residues} residues")

    written = {}
    written["prediction"] = write_prediction_npz(
        prediction, args.out / "prediction.npz", outputs=outputs
    )
    if "rmsf" in outputs:
        written["rmsf_csv"] = write_rmsf_csv(prediction, args.out / "rmsf.csv")
        written["rmsf_pdb"] = write_rmsf_bfactor_pdb(prediction, args.out / "rmsf_bfactor.pdb")
        print(f"[dynaprot] RMSF: mean {prediction.rmsf.mean():.2f} A, max {prediction.rmsf.max():.2f} A")

    if "ensemble" in outputs:
        print(f"[dynaprot] sampling {args.num_samples} conformations")
        samples = prediction.sample_ensemble(args.num_samples, seed=args.seed)
        formats = ("pdb", "xtc") if args.format == "both" else (args.format,)
        written.update(
            write_ensemble(
                samples,
                reference_pdb=args.input,
                out_dir=args.out,
                formats=formats,
                all_atom=args.all_atom,
                chain_id=args.chain,
            )
        )
        print(f"[dynaprot] sampled in {prediction.timings.get('sample_ensemble', 0):.3f} s")

    metadata = {
        "dynaprot_version": dynaprot.__version__,
        "input": str(args.input.resolve()),
        "chain": args.chain,
        "num_residues": prediction.num_residues,
        "sequence": prediction.sequence,
        "outputs": list(outputs),
        "num_samples": args.num_samples if "ensemble" in outputs else 0,
        "seed": args.seed,
        "device": str(args.device),
        "checkpoints": checkpoints,
        "timings_seconds": {**prediction.timings, "total": time.time() - started},
        "outputs": {k: str(Path(v).resolve()) for k, v in written.items()},
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"[dynaprot] wrote {len(written) + 1} files to {args.out}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="dynaprot",
        description="DynaProt: residue-level protein dynamics as multiscale Gaussians.",
    )
    parser.add_argument("--version", action="version", version=f"dynaprot {dynaprot.__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_predict_parser(subparsers)
    _add_visualize_parser(subparsers)

    args = parser.parse_args(argv)
    if args.command == "predict":
        return cmd_predict(args)
    if args.command == "visualize":
        return cmd_visualize(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
