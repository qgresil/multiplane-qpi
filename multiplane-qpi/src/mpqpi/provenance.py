"""Provenance block written into every output file: code version, parameters (with tags), inputs, computer, time."""
import json, platform, subprocess, time
from pathlib import Path

from .params import REG
from .config import REPO


def git_state():
    try:
        rev = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        return {"commit": rev or None, "uncommitted_changes": bool(dirty)}
    except OSError:
        return {"commit": None, "uncommitted_changes": None}


def stamp(step, inputs):
    import numpy, scipy
    return {"step": step, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "host": platform.node(),
            "python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
            "code": git_state(), "inputs": {k: str(v) for k, v in inputs.items()}, "parameters": REG.to_dict()}


def write_json(path, payload, step, inputs):
    """Write payload + provenance. Every downstream notebook reads these files, never re-derives upstream results."""
    payload = dict(payload); payload["provenance"] = stamp(step, inputs)
    Path(path).write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=float))
    return path
