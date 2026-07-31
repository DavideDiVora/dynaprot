"""Vendored subset of OpenFold (https://github.com/aqlaboratory/openfold).

OpenFold is not distributed on PyPI and its full installation requires building CUDA
extensions, so the handful of modules DynaProt actually needs are vendored here instead
of being taken as a dependency. See ``LICENSE`` in this directory (Apache 2.0) and
``PROVENANCE.md`` for what was copied and what was changed.

Vendored from OpenFold 2.2.0:
  - ``residue_constants``  (verbatim, except ``load_stereo_chemical_props`` which now raises)
  - ``rigid_utils``        (verbatim: ``Rigid``, ``Rotation``)
  - ``protein``            (verbatim, except its ``residue_constants`` import)
  - ``data_transforms``    (trimmed: ``atom37_to_frames``, ``get_backbone_frames``,
                            ``batched_gather``, ``np_to_tensor_dict``)
"""
from dynaprot.utils.openfold import residue_constants
from dynaprot.utils.openfold.data_transforms import (
    atom37_to_frames,
    batched_gather,
    get_backbone_frames,
    np_to_tensor_dict,
)
from dynaprot.utils.openfold.protein import Protein
from dynaprot.utils.openfold.rigid_utils import Rigid, Rotation

__all__ = [
    "Protein",
    "Rigid",
    "Rotation",
    "atom37_to_frames",
    "batched_gather",
    "get_backbone_frames",
    "np_to_tensor_dict",
    "residue_constants",
]
