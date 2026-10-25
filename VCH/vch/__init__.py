"""VCH - tank coil design simulation (P6 engine + desktop front end)."""
import os
from pathlib import Path

__version__ = "1.1.0"

# All processes (the app and its sweep workers) share one on-disk Numba cache, so the
# solver is compiled once and the workers only *load* it.  This must be set before
# numba is first imported, which is why it lives here (vch is imported first in every
# process, including the spawned workers).
if "NUMBA_CACHE_DIR" not in os.environ:
    _base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / ".cache"))
    _cache = _base / "TankCoilSim" / "numba_cache"
    try:
        _cache.mkdir(parents=True, exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = str(_cache)
    except OSError:
        pass  # numba falls back to its own default location
