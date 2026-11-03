"""Instrument description, local paths and output folders.

Local settings live in ~/.mpqpi.yaml (one per computer, never in git), e.g.
    data_root: /Volumes/T7/POSTDOC        # Mac
    data_root: D:/Quentin                 # Windows
    uipsf_python: /opt/homebrew/anaconda3/envs/uiPSF/bin/python
Notebooks refer to data relative to data_root, so the same notebook runs on every computer.
"""
from pathlib import Path
import yaml

from .params import REG

REPO = Path(__file__).resolve().parents[2]
DEFAULT_INSTRUMENT = REPO / "instruments" / "lob_multiplane.yaml"
LOCAL_FILE = Path.home() / ".mpqpi.yaml"


def local():
    """Per-computer settings (data_root, uipsf_python, ...). Empty dict if ~/.mpqpi.yaml does not exist."""
    if not LOCAL_FILE.exists():
        return {}
    return yaml.safe_load(LOCAL_FILE.read_text()) or {}


def data_path(rel):
    """Absolute path of a data file given relative to data_root (absolute paths are returned unchanged)."""
    p = Path(rel)
    if p.is_absolute():
        return p
    root = local().get("data_root")
    if root is None:
        raise RuntimeError(f"set data_root in {LOCAL_FILE} (see config/local.example.yaml)")
    return Path(root) / p


class Instrument:
    """Instrument YAML. `inst.value("objectives.60x_water.na")` returns the value and declares it in the registry."""

    def __init__(self, path=DEFAULT_INSTRUMENT):
        self.path = Path(path)
        self.tree = yaml.safe_load(self.path.read_text())

    def entry(self, key):
        node = self.tree
        for k in key.split("."):
            node = node[k]
        if not (isinstance(node, dict) and "value" in node and "tag" in node):
            raise KeyError(f"{key} is not a parameter entry in {self.path.name}")
        return node

    def value(self, key, name=None):
        e = self.entry(key)
        return REG.declare(name or key, e["value"], e["tag"], e["source"], e.get("unit", ""), e.get("note", ""))

    def pixel_um(self, objective):
        """Back-projected pixel: measured value if present, else camera pitch / (f_TL / f_obj · f_T2 / f_T1) (SI §6.1)."""
        meas = self.entry(f"objectives.{objective}.pixel_um_measured")
        if meas["value"] is not None:
            return self.value(f"objectives.{objective}.pixel_um_measured", "pixel_um")
        pitch = self.value("cameras.pixel_pitch_um")
        f_obj = self.value("detection.olympus_reference_tube_mm") / self.value(f"objectives.{objective}.magnification")
        mag = self.value("detection.tube_lens_mm") / f_obj * self.value("detection.relay_t2_mm") / self.value("detection.relay_t1_mm")
        return REG.declare("pixel_um", pitch / mag, "derived",
                           "camera pitch / (f_TL/f_obj · f_T2/f_T1), SI §6.1; nominal, not measured", "µm")


def output_dir(data_file, step):
    """Output folder next to the data: <stem>_<step>/ (e.g. ..._01_registration)."""
    data_file = Path(data_file)
    out = data_file.parent / f"{data_file.stem}_{step}"
    out.mkdir(parents=True, exist_ok=True)
    return out
