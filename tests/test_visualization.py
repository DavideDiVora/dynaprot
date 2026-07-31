"""Tests for the PyMOL ellipsoid visualization.

PyMOL is not a dependency, so these cover the parts that do not need it: the ellipsoid
geometry, and that the generated script is valid Python referring to the right arrays.
"""
import ast
import math
from pathlib import Path

import numpy as np
import pytest

from dynaprot.visualization import _ellipsoid_vertex, write_pymol_script


def _surface_points(covariance, center, n=64):
    """Sample points on the one-sigma ellipsoid surface the renderer draws."""
    w, V = np.linalg.eigh(np.asarray(covariance, dtype=float))
    a1, a2, a3 = np.sqrt(np.clip(w, 1e-12, None))
    pts = []
    rng = np.random.default_rng(0)
    for u, v in zip(rng.uniform(-math.pi / 2, math.pi / 2, n), rng.uniform(0, 2 * math.pi, n)):
        p, _ = _ellipsoid_vertex(a1, a2, a3, u, v, V, np.asarray(center, dtype=float))
        pts.append(p)
    return np.array(pts)


def test_ellipsoid_surface_is_the_one_sigma_level_set():
    """Every drawn vertex must satisfy (p-c)^T Sigma^-1 (p-c) = 1."""
    rng = np.random.default_rng(1)
    A = rng.normal(size=(3, 3))
    cov = A @ A.T + 3 * np.eye(3)
    center = np.array([10.0, -4.0, 2.5])

    pts = _surface_points(cov, center)
    inv = np.linalg.inv(cov)
    d = np.einsum("ni,ij,nj->n", pts - center, inv, pts - center)
    np.testing.assert_allclose(d, 1.0, rtol=1e-8, atol=1e-8)


def test_ellipsoid_axes_match_covariance_eigenvalues():
    """A diagonal covariance diag(1,4,9) must give semi-axes 1, 2, 3."""
    cov = np.diag([1.0, 4.0, 9.0])
    center = np.zeros(3)
    pts = _surface_points(cov, center, n=400)
    extent = np.abs(pts).max(axis=0)
    np.testing.assert_allclose(extent, [1.0, 2.0, 3.0], rtol=0.05)


def test_ellipsoid_follows_rotation():
    """Rotating the covariance rotates the ellipsoid by the same amount."""
    cov = np.diag([1.0, 4.0, 9.0])
    theta = math.pi / 3
    R = np.array([[math.cos(theta), -math.sin(theta), 0],
                  [math.sin(theta), math.cos(theta), 0],
                  [0, 0, 1]])
    rotated = R @ cov @ R.T

    inv = np.linalg.inv(rotated)
    pts = _surface_points(rotated, np.zeros(3))
    d = np.einsum("ni,ij,nj->n", pts, inv, pts)
    np.testing.assert_allclose(d, 1.0, rtol=1e-8, atol=1e-8)


def test_write_pymol_script_is_valid_python(tmp_path):
    out = write_pymol_script(tmp_path / "ellipsoids.py", arrays_name="prediction.npz",
                             structure="1ADE_A.pdb", scale=1.5)
    assert out.is_file()
    source = out.read_text()

    ast.parse(source)  # raises SyntaxError if the template produced invalid code
    assert 'covariances = _data["marginal_covariances"]' in source
    assert 'ca_coords = _data["ca_coords"]' in source
    assert '"prediction.npz"' in source
    assert "SCALE = 1.5" in source
    assert "'1ADE_A.pdb'" in source


def test_write_pymol_script_without_structure(tmp_path):
    out = write_pymol_script(tmp_path / "e.py")
    source = out.read_text()
    ast.parse(source)
    assert "STRUCTURE = None" in source
