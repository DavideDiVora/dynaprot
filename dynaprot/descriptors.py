"""Dynamics descriptors: deriving, projecting and composing them.

Shared by preprocessing (building training labels), inference (model outputs) and evaluation.
These operations implement the four levels of the paper's Gaussian view of dynamics:

* level 1 -- per-residue scalar RMSF_i = sqrt(Tr(Sigma_i))          :func:`rmsf_from_covariances`
* level 2 -- per-residue 3x3 marginals (the diagonal blocks)        :func:`extract_3x3_block_diagonal`
* level 3 -- N x N scalar coupling from the full joint              :func:`downsample_covariance_to_nxn`
* level 4 -- the full 3N x 3N joint, composed from levels 2 and 3   :func:`compose_joint_covariance`

"""
from __future__ import annotations

import torch

__all__ = [
    "cholesky_blockwise",
    "compose_joint_covariance",
    "downsample_covariance_to_nxn",
    "extract_3x3_block_diagonal",
    "project_to_psd",
    "renormalize_to_unit_diag",
    "rmsf_from_covariances",
    "sum_inner_product_heuristic",
]


def rmsf_from_covariances(cov_matrices: torch.Tensor) -> torch.Tensor:
    """RMSF_i = sqrt(Tr(Sigma_i)) for a stack of (N, 3, 3) covariances."""
    trace = cov_matrices.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    return torch.sqrt(trace)


def extract_3x3_block_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    """Pull the N diagonal 3x3 blocks out of a (3N, 3N) covariance -> (N, 3, 3)."""
    N = matrix.shape[0] // 3
    blocks = matrix.view(N, 3, N, 3)
    return blocks[torch.arange(N), :, torch.arange(N), :]


def cholesky_blockwise(blocks: torch.Tensor) -> torch.Tensor:
    """Cholesky factor of each (3, 3) block in an (N, 3, 3) stack."""
    return torch.stack([torch.linalg.cholesky(block) for block in blocks])


def compose_joint_covariance(
    marginal_covariances: torch.Tensor, correlation_matrix: torch.Tensor
) -> torch.Tensor:
    """Compose the full 3N x 3N joint covariance from the two DynaProt readouts.

    Implements the paper's joint reconstruction heuristic (Eq. 5)::

        Sigma_joint = L_marginal (C_tilde kron I_3) L_marginal^T

    where ``L_marginal`` is the block diagonal of the per-residue Cholesky factors and
    ``C_tilde`` is the unit-diagonal (correlation) form of the predicted coupling matrix.
    By Proposition 3.1 the result is SPD whenever both inputs are, since this is a
    congruence transformation of an SPD matrix by a full-rank matrix.

    Args:
        marginal_covariances: (N, 3, 3) SPD per-residue covariances, in the **global**
            frame (see :func:`dynaprot.utils.structure.map_one_protein_global_frame`).
        correlation_matrix: (N, N) SPD correlation matrix, unit diagonal (see
            :func:`renormalize_to_unit_diag`).

    Returns:
        (3N, 3N) SPD joint covariance.
    """
    I3 = torch.eye(
        3, device=correlation_matrix.device, dtype=marginal_covariances.dtype
    )
    kron_corr = torch.kron(correlation_matrix, I3)  # (3N, 3N)
    sqrt_blocks = cholesky_blockwise(marginal_covariances)
    L_marginal = torch.block_diag(*sqrt_blocks)
    return L_marginal @ kron_corr @ L_marginal.T


def downsample_covariance_to_nxn(cov_3n: torch.Tensor) -> torch.Tensor:
    """Project a (3N, 3N) covariance to the (N, N) scalar coupling matrix.

    Each entry is the mean of the corresponding 3x3 block, applied as a projection
    ``P Sigma P^T`` with ``P`` rows of ``1/sqrt(3)`` so that positive semi-definiteness
    is preserved. This produces the ``dynamics_correlations_nbyncovar`` training label
    for DynaProt-J.
    """
    assert cov_3n.ndim == 2 and cov_3n.shape[0] % 3 == 0
    N = cov_3n.shape[0] // 3
    device = cov_3n.device

    # Each row of P averages the corresponding 3D block
    P = torch.zeros(N, 3 * N, device=device, dtype=cov_3n.dtype)
    avg_vec = torch.ones(3, device=device) / 3.0**0.5

    for i in range(N):
        P[i, 3 * i : 3 * i + 3] = avg_vec

    # Project: [N x 3N] x [3N x 3N] x [3N x N] -> [N x N]
    return P @ cov_3n @ P.T


def sum_inner_product_heuristic(full_cov: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Alternative (N, N) reduction: block sums normalized by per-residue block sums.

    Produces the ``dynamics_correlations_sum`` label. Kept for reproducibility of the
    preprocessing outputs; the models are trained on
    :func:`downsample_covariance_to_nxn` instead.
    """
    N = full_cov.shape[0] // 3
    C = torch.zeros((N, N), device=full_cov.device, dtype=full_cov.dtype)

    sums = torch.tensor(
        [full_cov[3 * i : 3 * i + 3, 3 * i : 3 * i + 3].sum() for i in range(N)],
        device=full_cov.device,
    )

    for i in range(N):
        for j in range(i, N):
            Sigma_ij = full_cov[3 * i : 3 * i + 3, 3 * j : 3 * j + 3]
            numer = Sigma_ij.sum()
            denom = torch.sqrt(sums[i] * sums[j]) + eps
            val = numer / denom
            C[i, j] = val
            C[j, i] = val

    return C


def project_to_psd(C: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Nearest PSD matrix by clipping eigenvalues at ``eps``."""
    eigvals, eigvecs = torch.linalg.eigh(C)
    eigvals_clipped = torch.clamp(eigvals, min=eps)
    return eigvecs @ torch.diag(eigvals_clipped) @ eigvecs.T


def renormalize_to_unit_diag(C: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Standardize a covariance matrix into a correlation matrix (unit diagonal)."""
    diag = torch.diag(C)
    norm_factor = torch.sqrt(diag[:, None] * diag[None, :]) + eps
    return C / norm_factor
