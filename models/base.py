from __future__ import annotations
from dataclasses import dataclass, field, asdict, fields, is_dataclass
from typing import Any, Dict, List, Optional, ClassVar, Tuple, Union, get_args, get_origin
import logging

from .axis import Axis
from .axis_variable import AxisVariable
from .cross_section import CrossSection

log = logging.getLogger(__name__)

VERBOSE = False

# Base class for SPOT objects
@dataclass
class BaseObject:
    """Base class for all SPOT objects with common functionality."""
    
    # Core identification fields that most objects have
    name: str = ""
    axis_name: str = ""
    
    # Resolved references (set by configure methods)
    axis_obj: Optional[Axis] = None
    
    def set_axis(self, axis_data: List[Dict]) -> None:
        """Set the axis object from axis data."""
        if not self.axis_name or not axis_data:
            return
            
        # Find matching axis
        for axis_row in axis_data:
            if (axis_row.get("Class") == "Axis" and 
                str(axis_row.get("Name")) == self.axis_name):
                # Create axis object using existing function
                self.axis_obj = load_axis_from_rows(axis_data, self.axis_name)
                break

def _build_axis_index(axis_data: List[Dict], axis_map: Dict) -> Dict[str, Axis]:
    """Build an index of axis objects by name."""
    index = {}
    for row in axis_data:
        if row.get("Class") == "Axis":
            name = str(row.get("Name", ""))
            if name:
                try:
                    index[name] = load_axis_from_rows(axis_data, name)
                except Exception as e:
                    log.warning(f"Failed to create axis {name}: {e}")
    return index

# models/base.py
from typing import Dict, List, Optional, Tuple
import os, json
from models.axis import Axis
from models.cross_section import CrossSection

# ---------- existing ----------
def load_axis_from_rows(axis_rows: List[Dict], axis_name: str) -> Axis:
    row = next((r for r in axis_rows if r.get("Class") == "Axis" and str(r.get("Name")) == axis_name), None)
    if row is None:
        raise RuntimeError(f"Axis '{axis_name}' not found.")
    s = [float(x) for x in row.get("StaionValue", [])]
    x = [float(v) for v in row.get("CurvCoorX", [])]
    y = [float(v) for v in row.get("CurvCoorY", [])]
    z = [float(v) for v in row.get("CurvCoorZ", [])]
    return Axis(s, x, y, z, units="m")

# ---------- NEW: index cross-sections by NCS ----------
def index_cross_sections_by_ncs(cross_rows: List[Dict]) -> Dict[int, Dict]:
    by_ncs: Dict[int, Dict] = {}
    for r in cross_rows:
        if r.get("Class") != "CrossSection":
            continue
        try:
            ncs = int(r.get("NCS"))
        except Exception:
            continue
        by_ncs[ncs] = r
    return by_ncs

def choose_section_path_for_ncs(cs_row: Dict, fallback_path: str) -> str:
    jnames = cs_row.get("JSON_name") or []
    if isinstance(jnames, list) and jnames:
        cand = str(jnames[0]).replace("\\", "/")
        if os.path.isabs(cand):
            return cand
        base_dir = os.path.dirname(fallback_path) or "."
        return os.path.join(base_dir, os.path.basename(cand))
    return fallback_path

def load_section_for_ncs(ncs: int, by_ncs: Dict[int, Dict], fallback_path: str) -> CrossSection:
    row = by_ncs.get(int(ncs))
    if not row:
        return CrossSection.from_file(fallback_path, name=f"CS_NCS_{ncs}")
    path = choose_section_path_for_ncs(row, fallback_path)
    nm = str(row.get("Name") or f"CS_NCS_{ncs}")
    return CrossSection.from_file(path, name=nm)

# ---------- NEW: step function from Deck row (stations → NCS) ----------
def cs_steps_from_deck_row(deck_row: Dict) -> List[Tuple[float, int]]:
    xs = [float(x) for x in (deck_row.get("StationValue") or [])]
    ns = [int(n)   for n in (deck_row.get("CrossSection@Ncs") or [])]
    steps = [(x, n) for x, n in zip(xs, ns)]
    steps.sort(key=lambda t: t[0])
    return steps

def ncs_at_station(s_m: float, steps: List[Tuple[float, int]]) -> Optional[int]:
    if not steps:
        return None
    active = steps[0][1]
    for x, n in steps:
        if s_m >= x:
            active = n
        else:
            break
    return active

# ---------- unchanged helper ----------
def _get_cls_map(cls, mapping_config: dict) -> dict:
    if cls in mapping_config: return mapping_config[cls] or {}
    name = getattr(cls, "__name__", str(cls))
    if name in mapping_config: return mapping_config[name] or {}
    return {}

# ---------- (optional) legacy maker by name still here if needed ----------
# Removed make_linear_object function as LinearObject is now a base class


def _eval_spec(row: dict, spec):
    """
    Evaluate a mapping spec against a data row.
    - callable -> call(row)
    - str -> row.get(str)
    - (list/tuple) -> first present key in row
    - else -> None
    """
    if callable(spec):
        try:
            return spec(row)
        except Exception as e:
            log.debug("mapping callable raised: %s", e, exc_info=True)
            return None
    if isinstance(spec, (list, tuple)):
        for k in spec:
            if k in row:
                v = row.get(k)
                if v not in (None, ""):
                    return v
        return None
    if isinstance(spec, str):
        return row.get(spec)
    return None

def _coerce_value(val: Any, tp):
    origin = get_origin(tp) or getattr(tp, "__origin__", None)
    args   = get_args(tp)   or getattr(tp, "__args__", ())

    def _to_num(v, caster):
        try:
            if isinstance(v, str) and not v.strip():
                return v
            return caster(v)
        except Exception:
            return v

    if tp in (int, float, bool):
        caster = {int: int, float: float, bool: lambda x: bool(int(x)) if isinstance(x, (str, int)) else bool(x)}[tp]
        return _to_num(val, caster)

    if origin is Union:
        for a in args:
            if a is type(None):
                continue
            return _coerce_value(val, a)

    if origin in (list, set, tuple):
        inner = args[0] if args else Any
        # accept singletons & sequences
        if isinstance(val, (list, tuple, set)):
            seq = list(val)
        elif val is None:
            seq = []
        else:
            seq = [val]
        coerced = [_coerce_value(v, inner) for v in seq]
        # preserve container type if already a sequence
        if isinstance(val, tuple):
            return tuple(coerced)
        if isinstance(val, set):
            return set(coerced)
        return coerced

    return val

def from_dict(cls, row: dict, mapping_config: dict, *, axis_data=None, verbose: bool=False):
    """
    Generic factory that works with your current py->JSON mapping:
    mapping[field] may be a callable(row), a header name, or a list/tuple of header candidates.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")

    cls_map = _get_cls_map(cls, mapping_config)
    fdefs = {f.name: f for f in fields(cls)}

    kwargs = {}

    for fname, fdef in fdefs.items():
        spec = cls_map.get(fname)
        if spec is None:
            continue
        raw_val = _eval_spec(row, spec)
        kwargs[fname] = _coerce_value(raw_val, fdef.type)

        if verbose and fname in (
            "station_value", "ref_station_offset", "top_cross_section_ncs",
            "bot_cross_section_ncs", "internal_cross_section_ncs",
            "bot_zoffset", "top_zoffset"
        ):
            log.debug("[from_dict] %s.%s <- %r -> %r", getattr(cls, "__name__", cls), fname, raw_val, kwargs[fname])

    # Don’t wire axis_obj here; your VisoContext already does that reliably.
    # Keep axis_name if present in kwargs.

    if verbose:
        log.debug("[from_dict] Creating %s with kwargs keys=%s", getattr(cls, "__name__", cls), list(kwargs.keys()))
    return cls(**kwargs)

# ----------------------------------------------------------------------------
# Dedicated PierObject loader (bypasses generic mapping when needed)
# ----------------------------------------------------------------------------
def simple_load_piers(rows: List[dict]) -> List["PierObject"]:
    """Ultra-minimal PierObject loader.

    Accepts rows where Class == 'PierObject' and Type == 'Pier'. Extracts only fields
    actually used downstream: name, axis_name, station_value, top/bot NCS + point ids,
    offsets, elevation, axis variables.
    """
    from .pier_object import PierObject
    out: List[PierObject] = []
    for r in rows or []:
        if r.get("Class") != "PierObject" or r.get("Type") != "Pier":
            continue
        # Infer height if top/bot z offsets present
        try:
            top_z = float(r.get("Top-Zoffset", 0.0) or 0.0)
            bot_z = float(r.get("Bot-Zoffset", 0.0) or 0.0)
            inferred_height = abs(top_z - bot_z)
        except Exception:
            inferred_height = 0.0
        # Parse internal offsets list
        raw_offs = r.get("Internal_Ref-StationOffset") or []
        internal_offs: List[float] = []
        if isinstance(raw_offs, list):
            for v in raw_offs:
                try:
                    fv = float(v)
                    internal_offs.append(fv)
                except Exception:
                    continue
        # Build internal NCS + offsets
        internal_ncs = [int(v) for v in (r.get("Internal-CrossSection@Ncs") or []) if isinstance(v, (int, float, str))]
        p = PierObject(
            name=str(r.get("Name", "")),
            axis_name=str(r.get("Axis@Name", "")),
            station_value=float(r.get("StationValue", 0.0) or 0.0),
            height_m=inferred_height if inferred_height > 0 else None,
            top_cross_section_points_name=str(r.get("Top-CrossSection_Points@Name", "")),
            bot_cross_section_points_name=str(r.get("Bot-CrossSection_Points@Name", "")),
            top_cross_section_ncs=int(r.get("Top-CrossSection@Ncs", 0) or 0),
            bot_cross_section_ncs=int(r.get("Bot-CrossSection@Ncs", 0) or 0),
            internal_cross_section_ncs=internal_ncs,
            internal_ref_station_offset=internal_offs,
        )
        # Pre-compute ncs_steps if we have offsets and internal ncs count matches offsets
        try:
            if internal_ncs and internal_offs and len(internal_ncs) == len(internal_offs):
                steps = [(0.0, int(p.top_cross_section_ncs or internal_ncs[0]))]
                # clamp offsets to height if known
                h_m = float(p.height_m) if p.height_m else max(internal_offs) if internal_offs else 0.0
                cleaned_pairs = []
                for off, n in zip(internal_offs, internal_ncs):
                    try:
                        offc = max(0.0, min(h_m if h_m > 0 else off, float(off)))
                        cleaned_pairs.append((offc, int(n)))
                    except Exception:
                        continue
                for offc, n in sorted(cleaned_pairs, key=lambda x: x[0]):
                    if (offc, n) not in steps:
                        steps.append((float(offc), int(n)))
                bot_code = int(p.bot_cross_section_ncs or (internal_ncs[-1] if internal_ncs else p.top_cross_section_ncs))
                if not steps or steps[-1][0] != h_m or steps[-1][1] != bot_code:
                    if h_m > 0:
                        steps.append((float(h_m), bot_code))
                p.ncs_steps = steps
        except Exception:
            pass
        av = r.get("AxisVariables")
        if isinstance(av, list):
            p.axis_variables = av
        out.append(p)
    return out

