"""Detect incompatible Windows OpenMP runtimes before native imports."""

from __future__ import annotations

import importlib.util
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple


def _package_dir(name: str) -> Optional[Path]:
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, AttributeError, ValueError):
        return None
    if spec is None:
        return None
    locations = spec.submodule_search_locations
    if locations:
        return Path(next(iter(locations)))
    return Path(spec.origin).parent if spec.origin else None


def find_conflicting_runtimes() -> Tuple[Optional[Path], Optional[Path]]:
    torch_dir = _package_dir("torch")
    llama_dir = _package_dir("llama_cpp")
    intel = torch_dir / "lib" / "libiomp5md.dll" if torch_dir else None
    llvm = llama_dir / "lib" / "libomp140.x86_64.dll" if llama_dir else None
    return (
        intel if intel and intel.is_file() else None,
        llvm if llvm and llvm.is_file() else None,
    )


def warn_if_conflicting_openmp_runtimes() -> bool:
    if sys.platform != "win32":
        return False

    intel, llvm = find_conflicting_runtimes()
    if not (intel and llvm):
        return False

    bypass = os.environ.get("KMP_DUPLICATE_LIB_OK", "").strip().upper() in {
        "1", "TRUE", "YES", "ON",
    }
    bypass_note = (
        "\nKMP_DUPLICATE_LIB_OK is enabled, but remains an unsupported workaround."
        if bypass else ""
    )
    warnings.warn(
        "[ComfyUI-llama-cpp] Incompatible Windows OpenMP runtimes detected "
        "before native imports:\n"
        f"  PyTorch: {intel}\n"
        f"  llama.cpp: {llvm}\n"
        "Image decoding may terminate ComfyUI with OMP Error #15. Install a "
        "llama-cpp-python wheel built with -DGGML_OPENMP=OFF; see "
        "docs/windows-openmp.md."
        f"{bypass_note}",
        RuntimeWarning,
        stacklevel=2,
    )
    return True
