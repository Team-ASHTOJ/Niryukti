"""Resolve installed native artifacts, explicit overrides, or developer builds."""
import os
import shutil
import sys
from pathlib import Path

def executable(override=None):
    explicit = override or os.environ.get("VANTAGE_BINARY")
    if explicit:
        return str(explicit)
    root = Path(__file__).resolve().parent
    for path in (root / "_native/bin/vantage", root.parents[1] / "build/vantage"):
        if path.is_file():
            return str(path)
    found = shutil.which("vantage-engine")
    if found:
        return found
    raise FileNotFoundError("VANTAGE engine missing. Install a native vantage-opt wheel or set VANTAGE_BINARY.")

def shared_library(override=None):
    explicit = override or os.environ.get("VANTAGE_LIBRARY")
    if explicit:
        return str(explicit)
    name = "libvantage_c.dylib" if sys.platform == "darwin" else "libvantage_c.so"
    root = Path(__file__).resolve().parent
    for path in (root / "_native/lib" / name, root.parents[1] / "build" / name):
        if path.is_file():
            return str(path)
    raise FileNotFoundError("Native VANTAGE library missing; install a native wheel or set VANTAGE_LIBRARY.")
