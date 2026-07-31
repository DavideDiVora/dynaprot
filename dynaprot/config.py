"""Config loading with portable paths.

Configs use ``${ENV_VAR}`` placeholders and package-relative paths, which this module
resolves:

* ``${DYNAPROT_DATA_DIR}`` and any other ``${VAR}`` / ``~`` are expanded.
* ``protein_chains_path`` resolves against the installed package, so the split lists work
  regardless of the current working directory.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Union

import yaml

__all__ = ["expand_paths", "load_config", "resolve_protein_list"]

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

#: Split lists shipped inside the package.
PROTEIN_LISTS = Path(__file__).parent / "data" / "preprocessing" / "protein_lists"


def _expand(value: str) -> str:
    def replace(match: re.Match) -> str:
        name = match.group(1)
        env = os.environ.get(name)
        if env is None:
            raise KeyError(
                f"Config references ${{{name}}} but that environment variable is not set. "
                f"Export it, e.g. `export {name}=/path/to/data`."
            )
        return env

    return os.path.expanduser(_ENV_PATTERN.sub(replace, value))


def expand_paths(config: dict) -> dict:
    """Expand ``${ENV_VAR}`` and ``~`` in every string value, recursively."""
    out = {}
    for key, value in config.items():
        if isinstance(value, str):
            out[key] = _expand(value) if ("${" in value or value.startswith("~")) else value
        elif isinstance(value, dict):
            out[key] = expand_paths(value)
        else:
            out[key] = value
    return out


def resolve_protein_list(path: Union[str, Path]) -> Path:
    """Resolve a protein-list path, falling back to the copy shipped in the package."""
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    packaged = PROTEIN_LISTS / candidate.name
    if packaged.is_file():
        return packaged
    raise FileNotFoundError(
        f"Protein list not found: {path} (also looked for {packaged}). "
        f"Available: {sorted(p.name for p in PROTEIN_LISTS.glob('*.npy'))}"
    )


def load_config(path: Union[str, Path]) -> dict:
    """Load a YAML config and expand its path placeholders."""
    return expand_paths(yaml.safe_load(Path(path).read_text()))
