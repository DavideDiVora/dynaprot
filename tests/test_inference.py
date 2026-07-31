"""Featurization and end-to-end inference tests.

The model tests need checkpoints; they are skipped unless DYNAPROT_CKPT_DIR is set (or the
checkpoints are otherwise resolvable). The featurization tests run everywhere.
"""
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from dynaprot.descriptors import compose_joint_covariance
from dynaprot.inference import DynaProtPredictor, featurize_pdb, load_model

PDB = Path(__file__).parent / "data" / "1ADE_A.pdb"

pytestmark = pytest.mark.skipif(not PDB.is_file(), reason="test PDB not available")


def checkpoints_available():
    try:
        from dynaprot.inference import resolve_checkpoint_dir

        resolve_checkpoint_dir("dynaprot-M")
        return True
    except Exception:
        return False


needs_ckpts = pytest.mark.skipif(
    not checkpoints_available(), reason="set DYNAPROT_CKPT_DIR to run model tests"
)


def test_featurize_pdb_shapes():
    feats = featurize_pdb(PDB)
    N = feats["frames"].shape[0]
    assert N > 0
    # (N, 4, 4) is the rank preprocess_atlas.py stores and the models were trained on
    assert feats["frames"].shape == (N, 4, 4)
    assert feats["ca_coords"].shape == (N, 3)
    assert feats["aatype"].shape == (N,)
    assert len(feats["sequence"]) == N
    assert feats["mask"].sum() == N


def test_featurize_pdb_frames_are_valid_rigids():
    feats = featurize_pdb(PDB)
    rots = feats["frames"][:, :3, :3]
    eye = torch.eye(3, dtype=rots.dtype).expand_as(rots)
    assert torch.allclose(rots @ rots.transpose(-1, -2), eye, atol=1e-4), "rotations not orthogonal"
    assert torch.allclose(torch.linalg.det(rots), torch.ones(len(rots), dtype=rots.dtype), atol=1e-4)


def test_featurize_pdb_ca_matches_structure():
    """The Calpha coordinates are the means of the predicted Gaussians, so they must be right."""
    feats = featurize_pdb(PDB)
    translations = feats["frames"][:, :3, 3]
    assert torch.allclose(feats["ca_coords"], translations, atol=1e-3)


@needs_ckpts
def test_predict_outputs_are_spd():
    predictor = DynaProtPredictor.from_pretrained(device="cpu")
    prediction = predictor.predict(PDB)
    N = prediction.num_residues

    assert prediction.marginal_covariances.shape == (N, 3, 3)
    assert torch.linalg.eigvalsh(prediction.marginal_covariances).min() > 0

    assert prediction.coupling.shape == (N, N)
    assert torch.linalg.eigvalsh(prediction.coupling).min() > 0

    assert prediction.rmsf.shape == (N,)
    assert (prediction.rmsf > 0).all()


@needs_ckpts
def test_joint_covariance_is_spd_and_consistent():
    predictor = DynaProtPredictor.from_pretrained(device="cpu")
    prediction = predictor.predict(PDB)
    N = prediction.num_residues

    joint = prediction.joint_covariance
    assert joint.shape == (3 * N, 3 * N)
    assert torch.allclose(joint, joint.T, atol=1e-8)
    assert torch.linalg.eigvalsh(joint).min() > 0

    expected = compose_joint_covariance(prediction.marginal_covariances, prediction.correlation)
    assert torch.allclose(joint, expected)


@needs_ckpts
def test_sample_ensemble_is_deterministic_and_centred():
    predictor = DynaProtPredictor.from_pretrained(device="cpu")
    prediction = predictor.predict(PDB)
    N = prediction.num_residues

    a = prediction.sample_ensemble(64, seed=0)
    b = prediction.sample_ensemble(64, seed=0)
    assert a.shape == (64, N, 3)
    assert torch.allclose(a, b), "sampling is not reproducible for a fixed seed"

    # the ensemble mean should sit near the input structure (the Gaussian's mean)
    drift = (a.mean(dim=0) - prediction.ca_coords).norm(dim=-1)
    assert drift.mean() < prediction.rmsf.mean()


@needs_ckpts
def test_marginal_only_prediction_needs_no_joint_model():
    predictor = DynaProtPredictor.from_pretrained(joint=None, device="cpu")
    prediction = predictor.predict(PDB)
    assert prediction.marginal_covariances is not None
    assert prediction.coupling is None
    with pytest.raises(ValueError):
        _ = prediction.joint_covariance
