"""Run DynaProt on a structure: PDB in, dynamics descriptors and ensembles out.

Typical use::

    from dynaprot.inference import DynaProtPredictor

    predictor = DynaProtPredictor.from_pretrained(device="cuda:0")
    prediction = predictor.predict("1ADE_A.pdb")

    prediction.rmsf                  # (N,)     per-residue flexibility
    prediction.marginal_covariances  # (N,3,3)  anisotropic Gaussian blobs, global frame
    prediction.correlation           # (N,N)    residue-residue coupling
    prediction.joint_covariance      # (3N,3N)  composed via the paper's Eq. 5
    prediction.sample_ensemble(250)  # (250,N,3)
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import yaml

from dynaprot.descriptors import (
    compose_joint_covariance,
    renormalize_to_unit_diag,
    rmsf_from_covariances,
)
from dynaprot.model.architecture import DynaProt
from dynaprot.utils.ensemble import (
    reconstruct_ensemble_with_pulchra,
    sample_ensemble,
    save_ensemble_with_ss,
    save_ensemble_xtc,
)
from dynaprot.utils.structure import from_pdb_string, map_one_protein_global_frame
from dynaprot.utils.openfold import data_transforms, residue_constants
from dynaprot.utils.openfold.data_transforms import np_to_tensor_dict
from dynaprot.utils.openfold.protein import Protein, to_pdb

__all__ = [
    "DynaProtPrediction",
    "DynaProtPredictor",
    "featurize_pdb",
    "load_model",
    "load_prediction",
    "resolve_checkpoint_dir",
    "write_prediction_npz",
    "write_rmsf_bfactor_pdb",
    "write_rmsf_csv",
]

#: Environment variable pointing at a directory of frozen checkpoint folders, each
#: containing ``model.ckpt`` + ``model_config.yaml`` (+ optional ``data_config.yaml``).
CKPT_DIR_ENV = "DYNAPROT_CKPT_DIR"

#: Hugging Face repo holding the released weights; used when no local path resolves.
HF_REPO = "mihirbafna/dynaprot"

DEFAULT_MARGINAL = "dynaprot-M"
DEFAULT_JOINT = "dynaprot-J"

#: Descriptors DynaProt can emit. "rmsf" is derived from the marginals, so asking for it still
#: computes -- and keeps -- the full 3x3 covariances on the prediction.
OUTPUT_TYPES = ("rmsf", "marginals", "coupling", "ensemble")

#: Which model each output needs: (DynaProt-M, DynaProt-J).
_OUTPUT_REQUIRES = {
    "rmsf": (True, False),
    "marginals": (True, False),
    "coupling": (False, True),
    "ensemble": (True, True),  # the joint covariance is composed from both readouts
}


def normalize_outputs(outputs=None) -> tuple:
    """Validate an output selection, expanding ``None``/``"all"`` to every descriptor."""
    if outputs is None:
        return tuple(OUTPUT_TYPES)
    if isinstance(outputs, str):
        outputs = [outputs]
    resolved = []
    for item in outputs:
        if item == "all":
            resolved.extend(OUTPUT_TYPES)
        elif item in OUTPUT_TYPES:
            resolved.append(item)
        else:
            raise ValueError(
                f"Unknown output type {item!r}. Choose from {', '.join(OUTPUT_TYPES)} or 'all'."
            )
    return tuple(t for t in OUTPUT_TYPES if t in resolved)


def required_models(outputs) -> tuple:
    """-> (needs_marginal, needs_joint) for a normalized output selection."""
    return (any(_OUTPUT_REQUIRES[o][0] for o in outputs),
            any(_OUTPUT_REQUIRES[o][1] for o in outputs))


# --------------------------------------------------------------------------------------
# featurization
# --------------------------------------------------------------------------------------
def featurize_pdb(pdb_path: Union[str, Path], chain_id: Optional[str] = None) -> dict:
    """Turn a PDB file into the tensors DynaProt's forward pass expects.

    Args:
        pdb_path: path to a PDB file.
        chain_id: parse only this chain. ``None`` parses the whole file, which is only
            correct for single-chain inputs.

    Returns:
        dict with ``aatype`` (N,), ``frames`` (N,4,4) backbone rigid tensors,
        ``ca_coords`` (N,3) in Angstrom, ``mask`` (N,), ``residue_index`` (N,),
        ``sequence`` (str), and ``protein`` (the parsed feature dict, for writers).
    """
    pdb_str = Path(pdb_path).read_text()
    feats = from_pdb_string(pdb_str, chain_id=chain_id)

    aatype_idx = torch.tensor(np.asarray(feats["aatype"]), dtype=torch.long)
    tensors = np_to_tensor_dict(feats, feats.keys())
    tensors["aatype"] = aatype_idx
    tensors = data_transforms.atom37_to_frames(tensors)
    tensors = data_transforms.get_backbone_frames(tensors)

    # backbone_rigid_tensor is (N, 4, 4) -- the same rank preprocess_atlas.py stores as
    # "frames" and therefore the rank the trained models expect (batched to (B, N, 4, 4)).
    frames = tensors["backbone_rigid_tensor"]
    ca_index = residue_constants.atom_order["CA"]
    # The rigid ops emit float32; keep the coordinates in the same dtype so the returned
    # tensors are internally consistent. Callers upcast to float64 for the covariance algebra.
    ca_coords = tensors["all_atom_positions"][:, ca_index, :].to(frames.dtype)

    sequence = "".join(
        residue_constants.restypes_with_x[i] if i < len(residue_constants.restypes_with_x) else "X"
        for i in aatype_idx.tolist()
    )

    return dict(
        aatype=aatype_idx,
        frames=frames,
        ca_coords=ca_coords,
        mask=torch.ones(frames.shape[0]),
        residue_index=np.asarray(feats["residue_index"]),
        sequence=sequence,
        features=feats,
    )


# --------------------------------------------------------------------------------------
# checkpoint loading
# --------------------------------------------------------------------------------------
def resolve_checkpoint_dir(name_or_path: Union[str, Path]) -> Path:
    """Resolve a checkpoint reference to a local directory.

    Resolution order:

    1. an existing local path (directory containing ``model.ckpt``, or the ``.ckpt`` itself)
    2. ``$DYNAPROT_CKPT_DIR/<name>``
    3. download ``<name>`` from the Hugging Face repo :data:`HF_REPO`
    """
    path = Path(name_or_path)
    if path.exists():
        return path

    env_dir = os.environ.get(CKPT_DIR_ENV)
    if env_dir:
        candidate = Path(env_dir) / str(name_or_path)
        if candidate.exists():
            return candidate

    from huggingface_hub import snapshot_download

    try:
        local = snapshot_download(repo_id=HF_REPO, allow_patterns=f"{name_or_path}/*")
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not resolve checkpoint '{name_or_path}'. It is not a local path, not under "
            f"${CKPT_DIR_ENV}"
            f"{f' ({env_dir})' if env_dir else ' (unset)'}, and downloading it from the Hugging "
            f"Face repo '{HF_REPO}' failed: {exc}. Either set {CKPT_DIR_ENV} to a directory of "
            f"checkpoint folders, or pass an explicit path to the folder containing model.ckpt."
        ) from exc

    resolved = Path(local) / str(name_or_path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"'{HF_REPO}' has no folder named '{name_or_path}'. Expected a directory containing "
            f"model.ckpt and model_config.yaml."
        )
    return resolved


def load_model(
    source: Union[str, Path],
    model_config: Optional[Union[str, Path, dict]] = None,
    data_config: Optional[Union[str, Path, dict]] = None,
    device: Union[str, torch.device] = "cpu",
) -> DynaProt:
    """Load a DynaProt checkpoint.

    ``DynaProt.__init__`` never called ``save_hyperparameters()``, so the config is not
    recoverable from the checkpoint and must be supplied. Passing a frozen checkpoint
    *directory* (``model.ckpt`` + ``model_config.yaml`` + ``data_config.yaml``) resolves
    it automatically; that layout is what the release ships.

    Args:
        source: checkpoint directory, a ``.ckpt`` path, or a name to resolve (see
            :func:`resolve_checkpoint_dir`).
        model_config: required only when ``source`` is a bare ``.ckpt``.
        data_config: optional; attached as ``cfg["data_config"]``.
        device: device to place the model on.
    """
    path = resolve_checkpoint_dir(source)

    if path.is_dir():
        ckpt_path = path / "model.ckpt"
        if model_config is None:
            model_config = path / "model_config.yaml"
        if data_config is None and (path / "data_config.yaml").exists():
            data_config = path / "data_config.yaml"
    else:
        ckpt_path = path
        if model_config is None:
            raise ValueError(
                f"{ckpt_path} is a bare checkpoint; pass model_config explicitly "
                "(a checkpoint cannot be loaded without its YAML)."
            )

    cfg = model_config if isinstance(model_config, dict) else yaml.safe_load(Path(model_config).read_text())
    if data_config is not None:
        cfg["data_config"] = (
            data_config if isinstance(data_config, dict) else yaml.safe_load(Path(data_config).read_text())
        )

    model = DynaProt.load_from_checkpoint(str(ckpt_path), cfg=cfg, map_location="cpu")
    return model.to(device).eval()


# --------------------------------------------------------------------------------------
# prediction
# --------------------------------------------------------------------------------------
@dataclass
class DynaProtPrediction:
    """DynaProt's outputs for one structure, all on CPU in float64.

    Covariances are in Angstrom^2 and expressed in the **global** frame.
    """

    sequence: str
    residue_index: np.ndarray
    ca_coords: torch.Tensor  # (N, 3)
    frames: torch.Tensor  # (N, 4, 4)
    marginal_covariances: Optional[torch.Tensor] = None  # (N, 3, 3)
    coupling: Optional[torch.Tensor] = None  # (N, N)
    source_pdb: Optional[str] = None
    chain_id: Optional[str] = None
    features: Optional[dict] = field(default=None, repr=False)
    outputs: tuple = ()
    timings: dict = field(default_factory=dict)
    _joint: Optional[torch.Tensor] = field(default=None, repr=False)

    @property
    def num_residues(self) -> int:
        return len(self.sequence)

    @property
    def rmsf(self) -> torch.Tensor:
        """Per-residue RMSF, sqrt(Tr(Sigma_i)), in Angstrom."""
        if self.marginal_covariances is None:
            raise ValueError("No marginal covariances: DynaProt-M was not run.")
        return rmsf_from_covariances(self.marginal_covariances)

    @property
    def correlation(self) -> torch.Tensor:
        """Unit-diagonal (correlation) form of the predicted coupling matrix."""
        if self.coupling is None:
            raise ValueError("No coupling matrix: DynaProt-J was not run.")
        return renormalize_to_unit_diag(self.coupling)

    @property
    def joint_covariance(self) -> torch.Tensor:
        """Full (3N, 3N) joint covariance, composed via the paper's Eq. 5 heuristic."""
        if self._joint is None:
            if self.marginal_covariances is None or self.coupling is None:
                raise ValueError(
                    "The joint covariance needs both readouts; run DynaProt-M and DynaProt-J."
                )
            start = time.time()
            self._joint = compose_joint_covariance(self.marginal_covariances, self.correlation)
            self.timings["compose_joint"] = time.time() - start
        return self._joint

    def sample_ensemble(self, num_samples: int = 250, seed: Optional[int] = None) -> torch.Tensor:
        """Draw ``num_samples`` conformations from N(mu, Sigma_joint) -> (num_samples, N, 3).

        The mean is the input structure's Calpha coordinates, per the paper.
        """
        if seed is not None:
            torch.manual_seed(seed)
        mean = self.ca_coords.reshape(-1).double()
        samples, elapsed = sample_ensemble(
            mean, self.joint_covariance.double(), num_samples=num_samples, timed=True
        )
        self.timings["sample_ensemble"] = elapsed
        return samples


class DynaProtPredictor:
    """Holds the DynaProt-M and/or DynaProt-J models and runs them on structures."""

    def __init__(
        self,
        marginal_model: Optional[DynaProt] = None,
        joint_model: Optional[DynaProt] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        if marginal_model is None and joint_model is None:
            raise ValueError("Provide at least one of marginal_model / joint_model.")
        self.marginal_model = marginal_model
        self.joint_model = joint_model
        self.device = torch.device(device)

    @classmethod
    def from_pretrained(
        cls,
        marginal: Optional[Union[str, Path]] = DEFAULT_MARGINAL,
        joint: Optional[Union[str, Path]] = DEFAULT_JOINT,
        device: Union[str, torch.device] = "cpu",
    ) -> "DynaProtPredictor":
        """Load the released checkpoints (see :func:`resolve_checkpoint_dir`)."""
        m = load_model(marginal, device=device) if marginal else None
        j = load_model(joint, device=device) if joint else None
        return cls(marginal_model=m, joint_model=j, device=device)

    def _run(self, model: DynaProt, aatype, frames, mask) -> dict:
        dtype = next(model.parameters()).dtype
        with torch.no_grad():
            return model(
                aatype.unsqueeze(0).to(self.device),
                frames.unsqueeze(0).to(device=self.device, dtype=dtype),
                mask.unsqueeze(0).to(device=self.device, dtype=dtype),
            )

    def predict(
        self,
        pdb_path: Union[str, Path],
        chain_id: Optional[str] = None,
        outputs=None,
    ) -> DynaProtPrediction:
        """Predict dynamics descriptors for one structure.

        Args:
            pdb_path: input structure.
            chain_id: parse only this chain.
            outputs: which descriptors are wanted -- any of :data:`OUTPUT_TYPES`. Defaults to
                everything the loaded models can produce. Only the models actually needed are
                run, so asking for just ``"rmsf"`` skips the DynaProt-J forward pass. Asking
                for ``"rmsf"`` still computes and retains the 3x3 marginals it derives from.
        """
        if outputs is None:
            outputs = tuple(
                o for o in OUTPUT_TYPES
                if (not _OUTPUT_REQUIRES[o][0] or self.marginal_model is not None)
                and (not _OUTPUT_REQUIRES[o][1] or self.joint_model is not None)
            )
        else:
            outputs = normalize_outputs(outputs)
        needs_marginal, needs_joint = required_models(outputs)
        if needs_marginal and self.marginal_model is None:
            raise ValueError(f"outputs {outputs} need DynaProt-M, which was not loaded.")
        if needs_joint and self.joint_model is None:
            raise ValueError(f"outputs {outputs} need DynaProt-J, which was not loaded.")

        feats = featurize_pdb(pdb_path, chain_id=chain_id)
        aatype, frames, mask = feats["aatype"], feats["frames"], feats["mask"]

        timings: dict = {}
        marginal_covariances = None
        coupling = None

        if needs_marginal:
            start = time.time()
            out = self._run(self.marginal_model, aatype, frames, mask)
            marginal_local = out["marginal_covars"].squeeze(0).cpu().double()
            # The model predicts in the local backbone frame; rotate into the global frame
            # before anything is compared to MD or composed into the joint.
            marginal_covariances, _ = map_one_protein_global_frame(
                frames.cpu().double(), marginal_local
            )
            timings["marginal"] = time.time() - start

        if needs_joint:
            start = time.time()
            out = self._run(self.joint_model, aatype, frames, mask)
            coupling = out["joint_covar"].squeeze(0).cpu().double()
            timings["joint"] = time.time() - start

        return DynaProtPrediction(
            sequence=feats["sequence"],
            residue_index=feats["residue_index"],
            ca_coords=feats["ca_coords"].cpu().double(),
            frames=frames.cpu().double(),
            marginal_covariances=marginal_covariances,
            coupling=coupling,
            source_pdb=str(pdb_path),
            chain_id=chain_id,
            features=feats["features"],
            outputs=outputs,
            timings=timings,
        )


# --------------------------------------------------------------------------------------
# writers
# --------------------------------------------------------------------------------------
#: Shape/unit of every array `prediction.npz` can hold, mirrored into metadata.json so the
#: file is self-documenting (an .npz carries no units of its own).
ARRAY_SPEC = {
    "marginal_covariances": ("(N, 3, 3) per-residue marginal covariance, global frame", "A^2"),
    "rmsf": ("(N,) per-residue RMSF, sqrt(Tr(Sigma_i))", "A"),
    "coupling": ("(N, N) scalar residue-residue coupling covariance", "A^2"),
    "correlation": ("(N, N) unit-diagonal form of `coupling`", "-"),
    "ca_coords": ("(N, 3) input-structure CA coordinates (the Gaussian means)", "A"),
    "residue_index": ("(N,) residue numbering from the input PDB", "-"),
    "sequence": ("one-letter amino acid sequence", "-"),
}


def write_prediction_npz(
    prediction: DynaProtPrediction, path: Union[str, Path], outputs=None
) -> Path:
    """Write the predicted arrays to a single ``.npz``.

    Only the requested descriptors are stored; ``ca_coords``, ``residue_index`` and
    ``sequence`` are always included, since the descriptors are meaningless without the
    structure they refer to. See :data:`ARRAY_SPEC` for shapes and units.
    """
    outputs = normalize_outputs(outputs)
    arrays = dict(
        sequence=np.array(prediction.sequence),
        residue_index=np.asarray(prediction.residue_index),
        ca_coords=prediction.ca_coords.numpy(),
    )
    if prediction.marginal_covariances is not None:
        if "marginals" in outputs:
            arrays["marginal_covariances"] = prediction.marginal_covariances.numpy()
        if "rmsf" in outputs:
            arrays["rmsf"] = prediction.rmsf.numpy()
    if prediction.coupling is not None and "coupling" in outputs:
        arrays["coupling"] = prediction.coupling.numpy()
        arrays["correlation"] = prediction.correlation.numpy()
    np.savez_compressed(path, **arrays)
    return Path(path)


def load_prediction(path: Union[str, Path]) -> dict:
    """Load a ``prediction.npz`` back into plain Python/NumPy types.

    NumPy stores a Python string as a 0-d ``<U`` array, so ``np.load(...)["sequence"]`` comes
    back as an array rather than a ``str``. This loader unwraps it, so callers never need
    ``str(...)`` ceremony:

        >>> pred = load_prediction("out/")           # or "out/prediction.npz"
        >>> pred["sequence"][:8]
        'GNNVVVLG'
        >>> pred["rmsf"].shape
        (431,)

    Args:
        path: the ``.npz``, or a directory containing ``prediction.npz``.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "prediction.npz"
    with np.load(path) as data:
        loaded = {key: data[key] for key in data.files}
    if "sequence" in loaded:
        loaded["sequence"] = str(loaded["sequence"])
    return loaded


def write_rmsf_csv(prediction: DynaProtPrediction, path: Union[str, Path]) -> Path:
    """Write a residue-indexed RMSF table."""
    rmsf = prediction.rmsf.numpy()
    lines = ["residue_index,residue,rmsf"]
    lines += [
        f"{int(ri)},{aa},{value:.6f}"
        for ri, aa, value in zip(prediction.residue_index, prediction.sequence, rmsf)
    ]
    Path(path).write_text("\n".join(lines) + "\n")
    return Path(path)


def write_rmsf_bfactor_pdb(prediction: DynaProtPrediction, path: Union[str, Path]) -> Path:
    """Write the input structure with predicted RMSF in the B-factor column.

    Makes a prediction directly viewable -- e.g. ``spectrum b`` in PyMOL colours the
    structure by predicted flexibility.
    """
    feats = prediction.features
    if feats is None:
        feats = from_pdb_string(Path(prediction.source_pdb).read_text(), chain_id=prediction.chain_id)

    rmsf = prediction.rmsf.numpy()
    b_factors = np.zeros_like(np.asarray(feats["b_factors"], dtype=float))
    b_factors[:] = rmsf[:, None]

    protein = Protein(
        atom_positions=np.asarray(feats["all_atom_positions"]),
        aatype=np.asarray(feats["aatype"]),
        atom_mask=np.asarray(feats["all_atom_mask"]),
        residue_index=np.asarray(feats["residue_index"]),
        b_factors=b_factors,
        chain_index=np.zeros(len(rmsf), dtype=int),
    )
    Path(path).write_text(to_pdb(protein))
    return Path(path)


def write_ensemble(
    samples: torch.Tensor,
    reference_pdb: Union[str, Path],
    out_dir: Union[str, Path],
    stem: str = "ensemble",
    formats=("pdb",),
    all_atom: bool = False,
    chain_id: Optional[str] = None,
) -> dict:
    """Write a sampled ensemble as multi-model PDB and/or XTC, optionally all-atom.

    All-atom reconstruction shells out to PULCHRA, which must be on PATH.
    """
    out_dir = Path(out_dir)
    written: dict = {}

    if "pdb" in formats or all_atom:
        pdb_path = out_dir / f"{stem}.pdb"
        save_ensemble_with_ss(samples, str(reference_pdb), str(pdb_path), chain=chain_id)
        written["pdb"] = pdb_path

    if "xtc" in formats:
        xtc_path = out_dir / f"{stem}.xtc"
        save_ensemble_xtc(samples, str(reference_pdb), str(xtc_path))
        written["xtc"] = xtc_path

    if all_atom:
        aa_path = out_dir / f"{stem}_allatom.pdb"
        reconstruct_ensemble_with_pulchra(str(written["pdb"]), str(aa_path))
        written["allatom"] = aa_path
        if "pdb" not in formats:
            written.pop("pdb").unlink(missing_ok=True)

    return written
