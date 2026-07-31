"""Shared utilities: structure I/O, ensemble sampling, secondary structure, vendored OpenFold.

- :mod:`dynaprot.utils.structure` -- PDB parsing, Kabsch alignment, and the local <-> global
  backbone-frame mapping the model's covariance outputs must pass through.
- :mod:`dynaprot.utils.ensemble` -- sampling conformations from a joint covariance and writing
  them as multi-model PDB / XTC, optionally all-atom via PULCHRA.
- :mod:`dynaprot.utils.secondary_structure` -- P-SEA secondary-structure assignment from
  Calpha coordinates.
- :mod:`dynaprot.utils.openfold` -- vendored subset of OpenFold (Apache 2.0); see its
  ``PROVENANCE.md``.
"""
from dynaprot.utils.ensemble import (
    reconstruct_ensemble_with_pulchra,
    sample_ensemble,
    save_ensemble_with_ss,
    save_ensemble_xtc,
)
from dynaprot.utils.structure import (
    from_pdb_string,
    kabsch_rmsd,
    map_one_protein_global_frame,
    map_one_protein_local_frame,
    rmsdalign,
)

__all__ = [
    "from_pdb_string",
    "kabsch_rmsd",
    "map_one_protein_global_frame",
    "map_one_protein_local_frame",
    "reconstruct_ensemble_with_pulchra",
    "rmsdalign",
    "sample_ensemble",
    "save_ensemble_with_ss",
    "save_ensemble_xtc",
]
