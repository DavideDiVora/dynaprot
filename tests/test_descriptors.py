"""Tests for the dynamics-descriptor algebra in dynaprot.descriptors."""
import pytest
import torch

from dynaprot.descriptors import (
    cholesky_blockwise,
    compose_joint_covariance,
    downsample_covariance_to_nxn,
    extract_3x3_block_diagonal,
    renormalize_to_unit_diag,
    rmsf_from_covariances,
)


def random_spd(n, dim, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, dim, dim, generator=g, dtype=torch.float64)
    return A @ A.transpose(-1, -2) + dim * torch.eye(dim, dtype=torch.float64)


def test_rmsf_is_sqrt_trace():
    covs = random_spd(7, 3)
    expected = torch.sqrt(torch.stack([torch.trace(c) for c in covs]))
    assert torch.allclose(rmsf_from_covariances(covs), expected)


def test_cholesky_blockwise_reconstructs_blocks():
    covs = random_spd(5, 3)
    L = cholesky_blockwise(covs)
    assert torch.allclose(L @ L.transpose(-1, -2), covs, atol=1e-10)


def test_compose_joint_covariance_is_spd_and_symmetric():
    """Proposition 3.1: the Eq. 5 reconstruction preserves symmetry and positive definiteness."""
    N = 9
    marginals = random_spd(N, 3, seed=1)
    corr = renormalize_to_unit_diag(random_spd(1, N, seed=2)[0])

    joint = compose_joint_covariance(marginals, corr)

    assert joint.shape == (3 * N, 3 * N)
    assert torch.allclose(joint, joint.T, atol=1e-10), "not symmetric"
    assert torch.linalg.eigvalsh(joint).min() > 0, "not positive definite"


def test_compose_joint_covariance_recovers_marginals_on_diagonal():
    """With a unit-diagonal correlation, the diagonal blocks are exactly the marginals."""
    N = 6
    marginals = random_spd(N, 3, seed=3)
    corr = renormalize_to_unit_diag(random_spd(1, N, seed=4)[0])

    joint = compose_joint_covariance(marginals, corr)
    assert torch.allclose(extract_3x3_block_diagonal(joint), marginals, atol=1e-9)


def test_compose_joint_matches_notebook_heuristic():
    """Guards the port of `global_cov_heuristic` out of experiments/eval_pairwise.ipynb."""
    N = 5
    marginals = random_spd(N, 3, seed=5)
    corr = renormalize_to_unit_diag(random_spd(1, N, seed=6)[0])

    # verbatim from the notebook (cell 14)
    I3 = torch.eye(3, device=corr.device, dtype=marginals.dtype)
    kron_corr = torch.kron(corr, I3)
    sqrt_blocks = torch.stack([torch.linalg.cholesky(b) for b in marginals])
    L = torch.block_diag(*sqrt_blocks)
    expected = L @ kron_corr @ L.T

    assert torch.allclose(compose_joint_covariance(marginals, corr), expected, atol=1e-12)


def test_renormalize_to_unit_diag():
    C = random_spd(1, 8, seed=7)[0]
    R = renormalize_to_unit_diag(C)
    assert torch.allclose(torch.diag(R), torch.ones(8, dtype=torch.float64), atol=1e-6)
    assert R.abs().max() <= 1.0 + 1e-6


def test_downsample_preserves_psd():
    full = random_spd(1, 12, seed=8)[0]  # 3N x 3N with N = 4
    C = downsample_covariance_to_nxn(full)
    assert C.shape == (4, 4)
    assert torch.allclose(C, C.T, atol=1e-10)
    assert torch.linalg.eigvalsh(C).min() >= -1e-9


def test_downsample_entry_is_block_mean_times_three():
    """Each entry is the 3x3 block mean, scaled by 3 (P uses 1/sqrt(3) rows)."""
    full = random_spd(1, 9, seed=9)[0]
    C = downsample_covariance_to_nxn(full)
    for i in range(3):
        for j in range(3):
            block_mean = full[3 * i : 3 * i + 3, 3 * j : 3 * j + 3].mean()
            assert torch.isclose(C[i, j], block_mean * 3, atol=1e-9)


def test_extract_block_diagonal_roundtrip():
    N = 4
    marginals = random_spd(N, 3, seed=10)
    full = torch.block_diag(*marginals)
    assert torch.allclose(extract_3x3_block_diagonal(full), marginals)
