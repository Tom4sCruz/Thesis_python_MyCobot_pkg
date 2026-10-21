"""
COMPILED RUNS -- save a profile's rehearsed tape, reuse it until the code changes.
==================================================================================

Shared by HighH-LowR.py / HighH-HighR.py / LowH-HighR_v2.py / LowH-LowR.py.

Each of those scripts rehearses its whole run on a simulated arm before
anything moves (see THE TAPE in any of them). That tape depends only on the
source code and RANDOM_SEED -- never on the real arm -- so the same code
always produces the same tape. Instead of re-rehearsing on every launch, the
tape is saved to compiled/<script>.json together with a fingerprint of:

  * the script itself,
  * every armik/*.py (DH table, IK, planning, every config constant),
  * every scripts/Profiles/_*.py helper except _rviz_bridge.py (visual only),
  * the RANDOM_SEED.

On the next launch, a matching fingerprint means the saved tape is loaded and
the rehearsal is skipped; ANY edit to those files (even a comment) means it
is rehearsed again and re-saved. RANDOM_SEED = None (a fresh random run each
time) is never cached.

JSON, not pickle: the compiled files are committed to git so a run can be
compiled on the laptop and only PLAYED on the Jetson, and pickled numpy
arrays do not load across the two machines' numpy versions.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os

import numpy as np

from armik import Plan

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(os.path.dirname(_HERE))
COMPILED_DIR = os.path.join(_HERE, "compiled")
_FORMAT = 1                       # bump if the JSON layout below changes


def _source_files(script_path):
    armik_dir = os.path.join(_PKG_ROOT, "armik")
    files = [os.path.abspath(script_path)]
    files += sorted(os.path.join(armik_dir, f) for f in os.listdir(armik_dir) if f.endswith(".py"))
    files += sorted(os.path.join(_HERE, f) for f in os.listdir(_HERE)
                    if f.startswith("_") and f.endswith(".py") and f != "_rviz_bridge.py")
    return files


def source_key(script_path, seed):
    """SHA-256 over every source file the tape depends on, plus the seed."""
    h = hashlib.sha256(f"format={_FORMAT} seed={seed!r}\n".encode())
    for path in _source_files(script_path):
        h.update(os.path.relpath(path, _PKG_ROOT).encode() + b"\0")
        with open(path, "rb") as fh:
            h.update(fh.read())
        h.update(b"\0")
    return h.hexdigest()


def _path_for(script_path):
    stem = os.path.splitext(os.path.basename(script_path))[0]
    return os.path.join(COMPILED_DIR, stem + ".json")


# ---- JSON encoding of a tape (tuples, Plans, numpy arrays/scalars) ----------

def _enc(obj):
    if isinstance(obj, Plan):
        return {"__plan__": {f.name: _enc(getattr(obj, f.name)) for f in dataclasses.fields(obj)}}
    if isinstance(obj, np.ndarray):
        return {"__nd__": obj.tolist(), "dtype": str(obj.dtype)}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, tuple):
        return {"__tuple__": [_enc(v) for v in obj]}
    if isinstance(obj, list):
        return [_enc(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _enc(v) for k, v in obj.items()}
    return obj


def _dec(obj):
    if isinstance(obj, list):
        return [_dec(v) for v in obj]
    if not isinstance(obj, dict):
        return obj
    if "__plan__" in obj:
        return Plan(**{k: _dec(v) for k, v in obj["__plan__"].items()})
    if "__nd__" in obj:
        return np.asarray(obj["__nd__"], dtype=obj["dtype"])
    if "__tuple__" in obj:
        return tuple(_dec(v) for v in obj["__tuple__"])
    return {k: _dec(v) for k, v in obj.items()}


# ---- public API --------------------------------------------------------------

def load(script_path, seed):
    """The saved payload if it was compiled from exactly the current sources,
    else None (and a one-line reason is printed)."""
    if seed is None:
        print("RANDOM_SEED is None -- a fresh random run every time, so nothing is cached")
        return None
    path = _path_for(script_path)
    rel = os.path.relpath(path, _PKG_ROOT)
    if not os.path.exists(path):
        print(f"{rel}: not compiled yet")
        return None
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"{rel}: unreadable ({exc}) -- recompiling")
        return None
    if data.get("key") != source_key(script_path, seed):
        print(f"{rel}: source changed since it was compiled -- recompiling")
        return None
    try:
        return _dec(data["payload"])
    except Exception as exc:                       # e.g. a Plan field was renamed
        print(f"{rel}: could not be decoded ({exc}) -- recompiling")
        return None


def save(script_path, seed, payload):
    """Write the payload (overwriting the previous compile). No-op when seed
    is None. Returns the path written, or None."""
    if seed is None:
        return None
    os.makedirs(COMPILED_DIR, exist_ok=True)
    path = _path_for(script_path)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"script": os.path.basename(script_path), "seed": seed,
                   "key": source_key(script_path, seed), "payload": _enc(payload)},
                  fh, separators=(",", ":"))
    os.replace(tmp, path)
    return path
