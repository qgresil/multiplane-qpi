"""Parameter registry: every number that affects a result is declared here with its origin, so that assumptions are visible.

Tags
  paper         stated in Descloux et al. 2018 (SI); `source` gives the section / equation / figure
  measured      measured on this instrument (`source` says how, where and when)
  manufacturer  spec sheet or nominal design value
  derived       computed from other declared values (`source` gives the formula)
  user          set by the experimenter for this sample / acquisition (sample index, file, objective ...)
  heuristic     an implementation choice that is neither in the paper nor measured; `source` gives the rationale.
                These are the values to review; every notebook prints them.

Usage
  from mpqpi.params import REG
  na_ill = REG.declare("na_ill", 0.26, "paper", "SI §2, eq. S24 (NA_ill ≈ 0.26)", unit="")
  ...
  REG.banner()          # prints all values used so far, heuristics first
  REG.to_dict()         # stored in every output file (provenance)
"""
from dataclasses import dataclass, asdict, field
import json

TAGS = ("paper", "measured", "manufacturer", "derived", "user", "heuristic")


@dataclass
class Param:
    name: str
    value: object
    tag: str
    source: str
    unit: str = ""
    note: str = ""


@dataclass
class Registry:
    params: dict = field(default_factory=dict)

    def declare(self, name, value, tag, source, unit="", note=""):
        """Register `name` and return its value. Re-declaring a name (re-running a cell) replaces the entry."""
        if tag not in TAGS:
            raise ValueError(f"{name}: tag must be one of {TAGS}, got {tag!r}")
        if not source:
            raise ValueError(f"{name}: a source (reference, measurement or rationale) is required")
        self.params[name] = Param(name, value, tag, source, unit, note)
        return value

    def get(self, name):
        return self.params[name].value

    def clear(self):
        self.params.clear()

    def to_dict(self):
        return {k: {**asdict(p), "value": _jsonable(p.value)} for k, p in self.params.items()}

    def banner(self, show=("heuristic", "user", "measured", "paper", "manufacturer", "derived")):
        """Print the declared parameters, grouped by tag (heuristics first)."""
        lines = []
        for tag in show:
            ps = [p for p in self.params.values() if p.tag == tag]
            if not ps:
                continue
            head = f"{tag.upper()} ({len(ps)})" + ("  <- choices made in the code, review these" if tag == "heuristic" else "")
            lines.append(head)
            for p in ps:
                v = _short(p.value)
                lines.append(f"  {p.name:28s} = {v:>14s} {p.unit:6s} {p.source}" + (f"  [{p.note}]" if p.note else ""))
        print("\n".join(lines))

    def heuristics(self):
        return [p for p in self.params.values() if p.tag == "heuristic"]


def _short(v):
    s = json.dumps(_jsonable(v), ensure_ascii=False)
    return s if len(s) <= 14 else s[:13] + "…"


def _jsonable(v):
    try:
        import numpy as np
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
    except ImportError:
        pass
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


REG = Registry()
