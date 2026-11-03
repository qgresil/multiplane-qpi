"""Raw acquisitions -> two camera streams (A, B) with their stage z, independent of the file format.

Formats
  .dat + .info   LabVIEW acquisition: big-endian uint16, frames of both cameras interleaved; .info = JSON with the ROI, exposure
                 and Seq Params (ZstackParams StartZ / EndZ / NZ in nm).
  TIFF folder    Python controller: <camera>_<color>.tif per camera + acquisition.json (params.z_offsets_nm, per-frame z).
Stage z is always the *commanded* position: the stage read-outs in the metadata are not reliable (they jump by ~1 µm).
"""
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np

from .params import REG


@dataclass
class Acquisition:
    path: Path
    fmt: str                                   # "dat" or "tiff"
    streams: dict                              # "A"/"B" -> array (T, H, W), memory-mapped where possible
    z_um: dict                                 # "A"/"B" -> commanded stage z per frame (µm), or None for movies
    exposure_s: float
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self):
        return min(len(v) for v in self.streams.values())

    def step_um(self):
        z = self.z_um["A"]
        return None if z is None or len(z) < 2 else float(np.median(np.diff(z)))


def load(path, instrument, z_end_inclusive=None, color=None):
    """Open a .dat file or a TIFF acquisition folder. Stream assignment and the camera-B z shift come from the instrument file."""
    path = Path(path)
    shift = instrument.value("cameras.stream_b_z_shift_steps")
    if path.suffix == ".dat":
        acq = _load_dat(path, instrument, z_end_inclusive)
    elif path.is_dir():
        acq = _load_tiff(path, instrument, color)
    else:
        raise ValueError(f"{path}: expected a .dat file or a TIFF acquisition folder")
    if acq.z_um["B"] is not None and shift:
        step = acq.step_um()
        acq.z_um["B"] = acq.z_um["B"] + shift * step
    return acq


def _load_dat(path, instrument, z_end_inclusive):
    info = json.loads(path.with_suffix(".info").read_text())
    H, W = int(info["ROI CAM0"]["VWid"]), int(info["ROI CAM0"]["HWid"])
    n = path.stat().st_size // (H * W * 2)
    raw = np.memmap(path, dtype=np.dtype(">u2"), mode="r", shape=(n, H, W))
    first, second = instrument.value("cameras.dat_stream_order")
    streams = {first: raw[0:n - n % 2:2], second: raw[1:n - n % 2:2]}
    zp = info.get("Seq Params", {}).get("ZstackParams", {})
    z = None
    if info.get("Seq Params", {}).get("SeqType") == "ZSTACK" and int(zp.get("NZ", 0)) > 1:
        nz = int(zp["NZ"])
        incl = REG.declare("z_end_inclusive", bool(z_end_inclusive) if z_end_inclusive is not None else False, "heuristic",
                           ".dat z-stacks: step = (EndZ − StartZ) / NZ (False) or / (NZ − 1) (True); acquisition-software convention",
                           note="unverified: check against the stage or a known-thickness sample")
        step = (float(zp["EndZ"]) - float(zp["StartZ"])) * 1e-3 / ((nz - 1) if incl else nz)
        z = float(zp["StartZ"]) * 1e-3 + step * np.arange(nz)
        streams = {k: v[:nz] for k, v in streams.items()}
    return Acquisition(path, "dat", streams, {"A": z, "B": None if z is None else z.copy()},
                       float(info.get("ExpoCam0", np.nan)), {"info": info})


def _load_tiff(path, instrument, color):
    import tifffile
    acq = json.loads((path / "acquisition.json").read_text())
    files = instrument.value("cameras.tiff_files")
    colors = list(acq["params"].get("colors", {}).keys()) or [None]
    color = color or colors[0]
    streams = {}
    for s, cam in files.items():
        cands = sorted(path.glob(f"{cam}_{color}.tif")) if color else sorted(path.glob(f"{cam}*.tif"))
        if not cands:
            raise FileNotFoundError(f"no TIFF for {cam} ({color}) in {path}")
        streams[s] = tifffile.memmap(cands[0])
    zoff = acq["params"].get("z_offsets_nm")
    z = None if not zoff or len(zoff) < 2 else np.asarray(zoff, float) * 1e-3
    if z is not None and acq["params"].get("order", "c-z-xy") != "c-z-xy":
        raise NotImplementedError("only frame order c-z-xy with one colour and one position is handled")
    exp_ms = acq.get("state_start", {}).get("cameras", {}).get(files["A"], {}).get("exposure_ms", np.nan)
    return Acquisition(path, "tiff", streams, {"A": z, "B": None if z is None else z.copy()}, exp_ms * 1e-3, {"acquisition": acq})


def to_signal(frames, offset_counts):
    """Counts -> offset-subtracted float32 (the camera offset must be removed before any intensity normalisation, eq. S28)."""
    return np.asarray(frames, np.float32) - np.float32(offset_counts)
