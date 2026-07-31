"""Render predicted marginal Gaussians as PyMOL ellipsoids.

Each per-residue 3x3 covariance is an ellipsoid centred on that residue's Calpha, with axes
along the eigenvectors and semi-axis lengths ``sqrt(eigenvalue)`` -- i.e. a one-sigma surface
of the predicted positional distribution. This is the rendering used for Figure 4 of the paper.

"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

__all__ = ["ellipsoid_mesh", "render_ellipsoids", "write_pymol_script"]

PREDICTED_COLOR = (1.0, 20 / 255, 147 / 255)
REFERENCE_COLOR = (0.0, 0.545, 0.545)


def _ellipsoid_vertex(a1, a2, a3, u, v, M, r0):
    """A point and its normal on the ellipsoid surface, in the rotated frame ``M``."""
    vertex = M.dot(np.array([
        a1 * math.cos(u) * math.cos(v),
        a2 * math.cos(u) * math.sin(v),
        a3 * math.sin(u),
    ]))
    normal = M.dot(np.array([
        math.cos(u) * math.cos(v) / a1,
        math.cos(u) * math.sin(v) / a2,
        math.sin(u) / a3,
    ]))
    return vertex + r0, normal


def ellipsoid_mesh(
    center: Sequence[float],
    covariance: np.ndarray,
    color: Sequence[float] = PREDICTED_COLOR,
    scale: float = 1.0,
    resolution: int = 12,
) -> list:
    """Build the PyMOL CGO triangle mesh for one covariance ellipsoid.

    Args:
        center: (3,) Calpha coordinate the ellipsoid is centred on, in Angstrom.
        covariance: (3, 3) SPD covariance in Angstrom^2.
        color: RGB in [0, 1].
        scale: multiplies the semi-axes; 1.0 is the one-sigma surface.
        resolution: subdivisions per angular direction; cost grows as resolution^2.

    Returns:
        A CGO list suitable for ``pymol.cmd.load_cgo``.
    """
    from pymol.cgo import BEGIN, COLOR, END, NORMAL, TRIANGLES, VERTEX

    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(covariance, dtype=float))
    # eigh can return tiny negatives for near-singular inputs; clamp so sqrt is real
    a1, a2, a3 = scale * np.sqrt(np.clip(eigenvalues, 1e-12, None))
    M = eigenvectors
    r0 = np.asarray(center, dtype=float)

    mesh = [BEGIN, TRIANGLES, COLOR, *color]
    dU = math.pi / resolution
    dV = 2 * math.pi / resolution
    U = -math.pi / 2
    for _ in range(resolution):
        V = math.pi
        for _ in range(resolution):
            (x1, y1, z1), (n1x, n1y, n1z) = _ellipsoid_vertex(a1, a2, a3, U, V, M, r0)
            (x2, y2, z2), (n2x, n2y, n2z) = _ellipsoid_vertex(a1, a2, a3, U + dU, V, M, r0)
            (x3, y3, z3), (n3x, n3y, n3z) = _ellipsoid_vertex(a1, a2, a3, U + dU, V + dV, M, r0)
            (x4, y4, z4), (n4x, n4y, n4z) = _ellipsoid_vertex(a1, a2, a3, U, V + dV, M, r0)

            mesh.extend([NORMAL, n1x, n1y, n1z, VERTEX, x1, y1, z1])
            mesh.extend([NORMAL, n2x, n2y, n2z, VERTEX, x2, y2, z2])
            mesh.extend([NORMAL, n4x, n4y, n4z, VERTEX, x4, y4, z4])
            mesh.extend([NORMAL, n2x, n2y, n2z, VERTEX, x2, y2, z2])
            mesh.extend([NORMAL, n3x, n3y, n3z, VERTEX, x3, y3, z3])
            mesh.extend([NORMAL, n4x, n4y, n4z, VERTEX, x4, y4, z4])
            V += dV
        U += dU
    mesh.append(END)
    return mesh


def render_ellipsoids(
    covariances: np.ndarray,
    ca_coords: np.ndarray,
    structure: Optional[Union[str, Path]] = None,
    session_path: Optional[Union[str, Path]] = None,
    image_path: Optional[Union[str, Path]] = None,
    color: Sequence[float] = PREDICTED_COLOR,
    scale: float = 1.0,
    transparency: float = 0.3,
    resolution: int = 12,
    residues: Optional[Sequence[int]] = None,
) -> None:
    """Draw the ellipsoids in a live PyMOL session (requires ``pymol`` importable).

    Args:
        covariances: (N, 3, 3) predicted marginal covariances.
        ca_coords: (N, 3) Calpha coordinates.
        structure: optional PDB to load underneath the ellipsoids.
        session_path: save a ``.pse`` session here.
        image_path: ray-trace a PNG here.
        residues: 0-based indices to draw; default all.
    """
    try:
        from pymol import cmd
    except ImportError as exc:  # pragma: no cover - PyMOL is not a hard dependency
        raise ImportError(
            "render_ellipsoids needs PyMOL importable as a Python module "
            "(conda install -c conda-forge pymol-open-source). To avoid that dependency, "
            "use write_pymol_script() and run the generated script inside PyMOL instead."
        ) from exc

    covariances = np.asarray(covariances)
    ca_coords = np.asarray(ca_coords)
    if covariances.shape[0] != ca_coords.shape[0]:
        raise ValueError(
            f"{covariances.shape[0]} covariances vs {ca_coords.shape[0]} coordinates."
        )
    indices = range(len(ca_coords)) if residues is None else residues

    cmd.reinitialize()
    if structure is not None:
        cmd.load(str(structure))
        cmd.orient(Path(structure).stem)

    for i in indices:
        name = f"ellipsoid_{i:04d}"
        cmd.load_cgo(
            ellipsoid_mesh(ca_coords[i], covariances[i], color=color,
                           scale=scale, resolution=resolution),
            name,
        )
        cmd.set("cgo_transparency", transparency, name)

    cmd.bg_color("white")
    cmd.set("ray_trace_mode", 0)
    cmd.set("depth_cue", "off")
    cmd.set("ray_shadows", "off")
    cmd.zoom("all", buffer=10)

    if session_path is not None:
        cmd.save(str(session_path))
    if image_path is not None:
        cmd.png(str(image_path), width=1600, height=1200, dpi=300, ray=1)


_SCRIPT_TEMPLATE = '''"""Render DynaProt marginal Gaussians as ellipsoids. Generated by dynaprot visualize.

Run inside PyMOL:

    pymol {structure_hint} {script_name}

or, from the PyMOL prompt:

    run {script_name}
"""
import os

import numpy as np
from pymol import cmd
from pymol.cgo import BEGIN, COLOR, END, NORMAL, TRIANGLES, VERTEX

HERE = os.path.dirname(os.path.abspath(__file__))
ARRAYS = os.path.join(HERE, "{arrays_name}")
STRUCTURE = {structure_repr}
COLOR_RGB = {color!r}
SCALE = {scale!r}
TRANSPARENCY = {transparency!r}
RESOLUTION = {resolution!r}

_data = np.load(ARRAYS)
covariances = _data["{cov_key}"]
ca_coords = _data["ca_coords"]


def _vertex(a1, a2, a3, u, v, M, r0):
    v_ = M.dot(np.array([a1 * np.cos(u) * np.cos(v),
                         a2 * np.cos(u) * np.sin(v),
                         a3 * np.sin(u)]))
    n_ = M.dot(np.array([np.cos(u) * np.cos(v) / a1,
                         np.cos(u) * np.sin(v) / a2,
                         np.sin(u) / a3]))
    return v_ + r0, n_


def ellipsoid(center, cov, color, scale=1.0, resolution=12):
    w, V = np.linalg.eigh(np.asarray(cov, dtype=float))
    a1, a2, a3 = scale * np.sqrt(np.clip(w, 1e-12, None))
    r0 = np.asarray(center, dtype=float)
    mesh = [BEGIN, TRIANGLES, COLOR] + list(color)
    dU, dV = np.pi / resolution, 2 * np.pi / resolution
    U = -np.pi / 2
    for _ in range(resolution):
        W = np.pi
        for _ in range(resolution):
            p1, n1 = _vertex(a1, a2, a3, U, W, V, r0)
            p2, n2 = _vertex(a1, a2, a3, U + dU, W, V, r0)
            p3, n3 = _vertex(a1, a2, a3, U + dU, W + dV, V, r0)
            p4, n4 = _vertex(a1, a2, a3, U, W + dV, V, r0)
            for n, p in ((n1, p1), (n2, p2), (n4, p4), (n2, p2), (n3, p3), (n4, p4)):
                mesh += [NORMAL, n[0], n[1], n[2], VERTEX, p[0], p[1], p[2]]
            W += dV
        U += dU
    mesh.append(END)
    return mesh


if STRUCTURE is not None and os.path.exists(os.path.join(HERE, STRUCTURE)):
    cmd.load(os.path.join(HERE, STRUCTURE))

for i in range(len(ca_coords)):
    name = "ellipsoid_%04d" % i
    cmd.load_cgo(ellipsoid(ca_coords[i], covariances[i], COLOR_RGB, SCALE, RESOLUTION), name)
    cmd.set("cgo_transparency", TRANSPARENCY, name)

cmd.bg_color("white")
cmd.set("depth_cue", "off")
cmd.set("ray_shadows", "off")
cmd.zoom("all", buffer=10)
print("[dynaprot] drew %d ellipsoids" % len(ca_coords))
'''


def write_pymol_script(
    out_path: Union[str, Path],
    arrays_name: str = "prediction.npz",
    cov_key: str = "marginal_covariances",
    structure: Optional[str] = None,
    color: Sequence[float] = PREDICTED_COLOR,
    scale: float = 1.0,
    transparency: float = 0.3,
    resolution: int = 12,
) -> Path:
    """Write a standalone PyMOL script that draws the ellipsoids.

    Needs nothing installed here -- the script reads the arrays when PyMOL runs it, so the
    output directory stays self-contained and portable.

    Args:
        out_path: where to write the ``.py`` script.
        arrays_name: the ``.npz`` sitting next to it, holding ``cov_key`` and ``ca_coords``.
        cov_key: array name holding the (N, 3, 3) covariances.
        structure: optional PDB filename (relative to the script) to load underneath.
    """
    out_path = Path(out_path)
    script = _SCRIPT_TEMPLATE.format(
        script_name=out_path.name,
        structure_hint=structure or "structure.pdb",
        structure_repr=repr(structure),
        arrays_name=arrays_name,
        cov_key=cov_key,
        color=tuple(float(c) for c in color),
        scale=float(scale),
        transparency=float(transparency),
        resolution=int(resolution),
    )
    out_path.write_text(script)
    return out_path
