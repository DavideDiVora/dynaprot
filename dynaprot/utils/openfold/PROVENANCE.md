# Vendored OpenFold subset

Upstream: [aqlaboratory/openfold](https://github.com/aqlaboratory/openfold), version **2.2.0**.
Licensed under Apache License 2.0 — see `LICENSE` in this directory. Copyright 2021
AlQuraishi Laboratory and 2021 DeepMind Technologies Limited.

DynaProt needs four small pieces of OpenFold. OpenFold is not published on PyPI and a full
install requires compiling CUDA extensions, which makes it a poor dependency for a package
whose only use of it is structure parsing and backbone-frame construction. Those pieces are
therefore vendored here.

| File | Upstream path | Changes |
|---|---|---|
| `residue_constants.py` | `openfold/np/residue_constants.py` | Verbatim, except `load_stereo_chemical_props()` now raises `NotImplementedError` — it reads `openfold/resources/stereo_chemical_props.txt`, which upstream downloads via a setup script and which no DynaProt code path reaches. |
| `rigid_utils.py` | `openfold/utils/rigid_utils.py` | Verbatim (`Rigid`, `Rotation`). Has no OpenFold-internal imports. |
| `protein.py` | `openfold/np/protein.py` | Verbatim, except `from openfold.np import residue_constants` → `from dynaprot.utils.openfold import residue_constants`. |
| `data_transforms.py` | `openfold/data/data_transforms.py` + `openfold/utils/tensor_utils.py` + `openfold/data/feature_pipeline.py` | Trimmed to `atom37_to_frames`, `get_backbone_frames`, `batched_gather`, `np_to_tensor_dict`. The multimer branches of `atom37_to_frames` (which depended on `openfold.utils.geometry`) are removed; a multimer input now raises instead of silently taking a different branch. DynaProt only processes single chains. |

Note that `dynaprot/utils/structure.py` carries its own modified `from_pdb_string`, adapted from
`openfold/np/protein.py`; the vendored `protein.py` is kept for the `Protein` dataclass it
returns. `dynaprot/data/transforms.py` is likewise a pre-existing DynaProt adaptation of
OpenFold's cropping/padding transforms and is not part of this directory.

To re-sync with a newer OpenFold release, re-copy the three verbatim files, re-apply the two
import rewrites above, and re-extract the four functions in `data_transforms.py`.
