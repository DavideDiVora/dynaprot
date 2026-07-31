"""Tests for ensemble sampling and the on-disk writers.

These need no checkpoints -- they sample from a synthetic covariance -- so they always run.
The writers were previously exercised only by hand, which let a missing import survive.
"""
from pathlib import Path

import mdtraj as md
import numpy as np
import pytest
import torch

from dynaprot.descriptors import compose_joint_covariance, renormalize_to_unit_diag
from dynaprot.utils.ensemble import sample_ensemble, save_ensemble_with_ss, save_ensemble_xtc
from dynaprot.utils.structure import from_pdb_string

PDB = Path(__file__).parent / "data" / "1ADE_A.pdb"

pytestmark = pytest.mark.skipif(not PDB.is_file(), reason="test PDB not available")


@pytest.fixture(scope="module")
def synthetic():
    """A small synthetic ensemble built on the test structure's real Calpha coordinates."""
    feats = from_pdb_string(PDB.read_text())
    ca = torch.tensor(np.asarray(feats["all_atom_positions"])[:, 1, :], dtype=torch.float64)
    N = ca.shape[0]

    g = torch.Generator().manual_seed(0)
    A = torch.randn(N, 3, 3, generator=g, dtype=torch.float64)
    marginals = A @ A.transpose(-1, -2) + 3 * torch.eye(3, dtype=torch.float64)
    B = torch.randn(N, N, generator=g, dtype=torch.float64)
    coupling = B @ B.T + N * torch.eye(N, dtype=torch.float64)

    joint = compose_joint_covariance(marginals, renormalize_to_unit_diag(coupling))
    samples = sample_ensemble(ca.reshape(-1), joint, num_samples=8, timed=False)
    return ca, samples


def test_sample_ensemble_shape_and_timing(synthetic):
    ca, samples = synthetic
    assert samples.shape == (8, ca.shape[0], 3)

    timed = sample_ensemble(ca.reshape(-1), torch.eye(3 * ca.shape[0], dtype=torch.float64),
                            num_samples=2, timed=True)
    assert isinstance(timed, tuple) and len(timed) == 2
    assert timed[0].shape == (2, ca.shape[0], 3)


def test_save_ensemble_with_ss_writes_readable_multimodel_pdb(synthetic, tmp_path):
    ca, samples = synthetic
    out = tmp_path / "ensemble.pdb"
    save_ensemble_with_ss(samples, str(PDB), str(out))

    assert out.is_file()
    traj = md.load(str(out))
    assert traj.n_frames == 8
    assert traj.n_atoms == ca.shape[0]
    assert all(a.name == "CA" for a in traj.top.atoms)


def test_save_ensemble_xtc_roundtrips(synthetic, tmp_path):
    ca, samples = synthetic
    out = tmp_path / "ensemble.xtc"
    save_ensemble_xtc(samples, str(PDB), str(out))

    assert out.is_file()
    ref_ca = md.load(str(PDB)).atom_slice(md.load(str(PDB)).top.select("name CA"))
    traj = md.load(str(out), top=ref_ca.top)
    assert traj.n_frames == 8
    # writer converts Angstrom -> nm; check a coordinate survives the round trip
    np.testing.assert_allclose(traj.xyz[0] * 10.0, samples[0].numpy(), rtol=1e-3, atol=1e-2)
