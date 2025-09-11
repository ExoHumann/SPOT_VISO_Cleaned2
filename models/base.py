from __future__ import annotations
from dataclasses import dataclass, field, asdict, fields
from typing import Any, Dict, List, Optional, ClassVar, Tuple, Union, get_args, get_origin
import logging

from .axis import Axis
from .axis_variable import AxisVariable
from .cross_section import CrossSection

log = logging.getLogger(__name__)

VERBOSE = False

from dataclasses import fields, is_dataclass, asdict
from typing import Any, Iterable, List, Union, get_args, get_origin
import logging

log = logging.getLogger(__name__)

def _get_cls_map(cls, mapping_config: dict) -> dict:
    """Find the mapping block for a class, accepting class object or name."""
    if cls in mapping_config:
        return mapping_config[cls] or {}
    name = getattr(cls, "__name__", str(cls))
    if name in mapping_config:
        return mapping_config[name] or {}
    return {}

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

def _build_axis_index(axis_data, axis_map=None):
    """
    Minimal compatibility stub used by some modules.
    Returns a dict: {axis_name: axis_dict}
    """
    idx = {}
    if not axis_data:
        return idx
    try:
        for ax in axis_data:
            if isinstance(ax, dict):
                name = ax.get("name")
                if name:
                    idx[str(name)] = ax
            else:
                name = getattr(ax, "name", None)
                if name:
                    idx[str(name)] = asdict(ax)
    except Exception:
        pass
    return idx

@dataclass(slots=True)
class BaseObject:
    """
    Shared behavior & state for DeckObject / PierObject / FoundationObject.
    This class is now storage-agnostic: it only holds python attributes
    and relies on the loader/context for wiring.
    """
    # identity
    name: str = ""
    axis_name: str = ""

    # axis & variables (wired by loader/context)
    axis_obj: Optional[Axis] = None
    axis_variables: List[Dict] = field(default_factory=list)            # raw rows (already python keys)
    axis_variables_obj: List[AxisVariable] = field(default_factory=list)

    # user display / config
    user_stations: List[float] = field(default_factory=list)
    axis_rotation: float = 0.0
    colors: Dict[str, str] = field(default_factory=lambda: {
        "axis": "black",
        "first_station_points": "blue",
        "cross_section_loops": "red",
        "longitudinal_lines": "gray",
    })

    # memo & caches
    _memo_cache: Dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    _TRANSFORM_CACHE: ClassVar[Dict] = {}
    _TRANSFORM_CACHE_MAX: ClassVar[int] = 256

    # ------- tiny utils -------
    def memo(self, key, builder):
        m = self._memo_cache
        if key not in m:
            m[key] = builder()
        return m[key]

    def debug(self, *a, **k):
        if VERBOSE:
            print(*a, **k)

    # ------- axis variables -------
    def set_axis_variables(self, axis_var_map: dict):
        """
        Build AxisVariable objects from self.axis_variables (raw python dicts).
        Skip recompute if signature unchanged.
        """
        try:
            raw = self.axis_variables or []
            sig = _axis_vars_signature(raw)
            if getattr(self, "_axis_vars_sig", None) == sig and self.axis_variables_obj:
                return
            self.axis_variables_obj = create_axis_variables(raw, axis_var_map)
            self._axis_vars_sig = sig
        except Exception as e:
            log.warning("Axis variables parse failed for %s: %s", getattr(self, 'name', '<unnamed>'), e)
            self.axis_variables_obj = []
            self._axis_vars_sig = None

    def evaluate_axis_vars_at_stations(self, stations_m: List[float]) -> List[Dict[str, float]]:
        if not stations_m:
            return []
        self.set_axis_variables(axis_var_map={})
        return AxisVariable.evaluate_at_stations_cached(self.axis_variables_obj or [], stations_m)

    # ------- transform caching -------
    def _axis_signature(self) -> Tuple[str, str, str, str, str]:
        if not self.axis_obj:
            return ("", "", "", "", "m")
        return (
            _tiny_hash(self.axis_obj.stations),
            _tiny_hash(self.axis_obj.x_coords),
            _tiny_hash(self.axis_obj.y_coords),
            _tiny_hash(self.axis_obj.z_coords),
            "mm",
        )

    @staticmethod
    def _points_signature(nested_points: List[List[Dict[str, float]]]) -> str:
        flat = []
        for ring in (nested_points or []):
            for p in (ring or []):
                flat.append(f"{float(p.get('x',0)):.6f}:{float(p.get('y',0)):.6f}")
            flat.append("|")
        import hashlib
        return hashlib.blake2b(",".join(flat).encode("utf-8"), digest_size=16).hexdigest()

    def transform_points_cached(
        self,
        local_points: List[List[Dict[str, float]]],
        stations_to_plot: List[float],
        rotation_angle: Optional[float] = None,
    ):
        if self.axis_obj is None:
            raise RuntimeError("axis_obj is not set (loader/context should wire it).")
        rot = self.axis_rotation if rotation_angle is None else float(rotation_angle)
        key = (
            self._axis_signature(),
            tuple(round(float(s), 8) for s in (stations_to_plot or [])),
            round(rot, 6),
            self._points_signature(local_points),
        )
        hit = BaseObject._TRANSFORM_CACHE.get(key)
        if hit is not None:
            return hit

        result = self.axis_obj.transform_points(local_points, stations_to_plot, rotation_angle=rot)
        if len(BaseObject._TRANSFORM_CACHE) >= BaseObject._TRANSFORM_CACHE_MAX:
            BaseObject._TRANSFORM_CACHE.clear()
        BaseObject._TRANSFORM_CACHE[key] = result
        return result

    # ------- visualization input -------
    def get_input_for_visualisation(
        self,
        *,
        cross_section_objects: Optional[List[CrossSection]] = None,
        axis_rotation: Optional[float] = None,
        colors: Optional[Dict[str, str]] = None,
    ) -> dict:
        """
        Delegate to your existing vis adapter. No raw JSON here.
        Requires axis_obj to be set (or use context helper below).
        """
        from .vis_adapter import create_input_for_visualisation
        return create_input_for_visualisation(
            obj=self,
            axis_data=None,  # not needed anymore
            cross_section_objects=cross_section_objects,
            json_file_path=None,
            axis_rotation=axis_rotation,
            colors=colors,
        )

    # ------- helpers for dependent lookups -------
    def _resolve_cross_sections_from_ncs(self, ctx) -> List[CrossSection]:
        ncs_list = getattr(self, "cross_section_ncs", []) or []
        out: List[CrossSection] = []
        for n in ncs_list:
            try:
                cs = ctx.crosssec_by_ncs.get(int(n))
            except Exception:
                cs = None
            if cs:
                out.append(cs)
        return out
    
    @classmethod
    def from_row(cls, row: dict, mapping_config: dict, *, axis_data=None, verbose: bool=False):
        return from_dict(cls, row, mapping_config, axis_data=axis_data, verbose=verbose)



# ── helpers used above ─────────────────────────────────────────────────────────
def _tiny_hash(arr) -> str:
    import hashlib
    if arr is None:
        arr = []
    data = (','.join(str(float(v)) for v in arr)).encode('utf-8')
    return hashlib.blake2b(data, digest_size=12).hexdigest()

def _as_floats(val) -> List[float]:
    out = []
    if val is None: 
        return out
    if not isinstance(val, (list, tuple)):
        val = [val]
    for v in val:
        try:
            out.append(float(v))
        except Exception:
            pass
    return out

def _as_str_list(val) -> List[str]:
    if val is None:
        return []
    if isinstance(val, str):
        return [val]
    if isinstance(val, (list, tuple)):
        return [str(x) for x in val]
    return [str(val)]

def _axis_vars_signature(rows: List[Dict]) -> str:
    import hashlib, json
    try:
        blob = json.dumps(rows or [], sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        blob = str(rows)
    return hashlib.blake2b(blob.encode("utf-8"), digest_size=16).hexdigest()

def create_axis_variables(json_data: List[Dict], var_map: Dict) -> List[AxisVariable]:
    # Uses your existing AxisVariable API; this expects keys already normalized.

    name_k = var_map.get("VariableName",       "VariableName")
    xs_k   = var_map.get("StationValue",       "StationValue")      # <-- was "VariableStations"
    ys_k   = var_map.get("VariableValues",     "VariableValues")
    typ_k  = var_map.get("VariableIntTypes",   "VariableIntTypes")
    desc_k = var_map.get("VariableDescription","VariableDescription")
    ...

    normalized: List[Dict] = []
    for row in (json_data or []):
        mapped = { var_map.get(k, k): v for k, v in row.items() }
        name = str(mapped.get(name_k, "")).strip()
        xs   = _as_floats(mapped.get(xs_k))
        ys   = _as_floats(mapped.get(ys_k))
        tys  = _as_str_list(mapped.get(typ_k))
        desc = mapped.get(desc_k, "")
        if len(xs) != len(ys):
            n = min(len(xs), len(ys))
            xs, ys = xs[:n], ys[:n]
        pairs = sorted(zip(xs, ys), key=lambda t: t[0])
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        if not name or not xs:
            continue
        normalized.append({
            name_k: name,
            xs_k:   xs,
            ys_k:   ys,
            typ_k:  tys or ["linear"],
            desc_k: desc,
        })
    return AxisVariable.create_axis_variables(normalized)

#── Multi-candidate raw-key resolver ────────────────────────────────────────────
def _raw_candidates_for(internal: str, cls_map: dict[str, str]) -> list[str]:
    """
    All raw keys that map to this internal field (handles aliases like
    'Top-CrossSection@Ncs' and 'Top-CrossSection@NCS').
    """
    return [rk for rk, v in (cls_map or {}).items() if v == internal]

def _pick_raw_key(internal: str, cls_map: dict[str, str], data: dict, fallback_same_name=True) -> str | None:
    """
    Pick the first raw key for 'internal' that exists in 'data'.
    If none exists and fallback_same_name is True, try the internal name itself.
    """
    for k in _raw_candidates_for(internal, cls_map):
        if k in data:
            return k
    if fallback_same_name and internal in data:
        return internal
    return None

# ── Robust Axis type detector ───────────────────────────────────────────────────
def _is_axis_type(tp) -> bool:
    """
    Returns True if the type annotation contains models.axis.Axis anywhere:
    Axis, Optional[Axis], Union[Axis, None], List[Axis], Tuple[Axis, ...],
    or a forward-ref string like "Axis" / "models.axis.Axis".
    """
    if tp is None:
        return False

    # string forward-ref
    if isinstance(tp, str):
        return tp.split(".")[-1] == "Axis"

    AxisClass = None
    try:
        from models.axis import Axis as AxisClass  # your real Axis class
    except Exception:
        pass

    if AxisClass is not None and tp is AxisClass:
        return True

    origin = get_origin(tp) or getattr(tp, "__origin__", None)
    args   = get_args(tp)   or getattr(tp, "__args__", ())

    if origin is None:
        # tolerate direct class compare by name
        return getattr(tp, "__name__", "") == "Axis"

    # containers
    if origin in (list, set, tuple):
        return any(_is_axis_type(a) for a in args)

    # Optional/Union
    if origin is Union:
        return any(_is_axis_type(a) for a in args if a is not type(None))

    return False




# from __future__ import annotations
# from copy import deepcopy
# from dataclasses import dataclass, field, asdict, fields, is_dataclass
# import json
# from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, ClassVar, Type, Union
# from venv import logger

# from .mapping import mapping
# from .axis import Axis
# from .axis_variable import AxisVariable
# from .cross_section import CrossSection
# if TYPE_CHECKING:
#     from .viso_context import VisoContext
#     from .main_station import MainStation

# VERBOSE = False

# @dataclass(slots=True)
# class BaseObject:
#     """
#     Shared behavior & state for DeckObject / PierObject / FoundationObject.
#     """
#     # core identity
#     axis_name: str = ""

#     # persisted state
#     axis_variables: List[Dict] = field(default_factory=list)            # raw JSON rows
#     axis_variables_obj: List[AxisVariable] = field(default_factory=list)
#     axis_obj: Optional[Axis] = None

#     user_stations: List[float] = field(default_factory=list)
#     colors: Dict[str, str] = field(default_factory=lambda: {
#         "axis": "black",
#         "first_station_points": "blue",
#         "cross_section_loops": "red",
#         "longitudinal_lines": "gray",
#     })
#     axis_rotation: float = 0.0

#     # per-instance memo cache
#     _memo_cache: Dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

#     # class-wide bounded transform cache
#     _TRANSFORM_CACHE: ClassVar[Dict] = {}
#     _TRANSFORM_CACHE_MAX: ClassVar[int] = 256

#     # ---- tiny utilities ----
#     def memo(self, key, builder):
#         m = self._memo_cache
#         if key not in m:
#             m[key] = builder()
#         return m[key]

#     def log(self, *a, **k):
#         flag = getattr(self, 'debug', None)
#         if flag is None:
#             flag = VERBOSE
#         if flag:
#             print(*a, **k)

#     # ---------- convenience ----------
#     def get_object_metada(self) -> Dict[str, Any]:
#         data = asdict(self)
#         # tidy noisy fields
#         data['axis_variables'] = f"<{len(self.axis_variables_obj)} axis variables>"
#         data['axis_variables_obj'] = f"<{len(self.axis_variables_obj)} axis variable objects>"
#         data['axis_obj'] = "<Axis object>" if self.axis_obj is not None else None
#         # drop runtime-only
#         for k in ('colors', 'user_stations', 'axis_obj', 'axis_variables_obj', 'axis_rotation', '_memo_cache'):
#             data.pop(k, None)
#         return data
    
#     def evaluate_axis_vars_at_stations(self, stations_m: List[float]) -> List[Dict[str, float]]:
#         """
#         Evaluate this object's axis variables at the given stations (meters on input).
#         Returns a list of dicts [{varname: value, ...}] aligned to stations.
#         Uses the global cached evaluation in AxisVariable.
#         """
#         from .axis_variable import AxisVariable
#         if not stations_m:
#             return []
#         # Ensure axis_variables_obj are built
#         self.set_axis_variables(axis_var_map={})
#         return AxisVariable.evaluate_at_stations_cached(self.axis_variables_obj or [], stations_m)


#     # ---------- axis variables ----------
#     def set_axis_variables(self, axis_var_map: dict):
#         try:
#             # normalize & cache: rebuild only if raw changed
#             raw = self.axis_variables or []
#             sig = _axis_vars_signature(raw)
#             if getattr(self, "_axis_vars_sig", None) == sig and self.axis_variables_obj:
#                 return
#             self.axis_variables_obj = create_axis_variables(raw, axis_var_map)
#             self._axis_vars_sig = sig
#         except Exception as e:
#             logger.warning("Axis variables parse failed for %s: %s", getattr(self, 'name', '<unnamed>'), e)
#             self.axis_variables_obj = []
#             self._axis_vars_sig = None


#     # ---------- axis lookup & creation ----------
#     def _axis_dict_from(self, axis_data: Optional[List[Dict]] = None,
#                         axis_index: Optional[Dict[str, Dict]] = None) -> Optional[Dict]:
#         if not self.axis_name:
#             return None
#         key = str(self.axis_name).strip().lower()
#         if axis_index:
#             return axis_index.get(key)
#         axis_map = mapping.get(Axis, {})
#         class_key = axis_map.get('class', 'Class')
#         name_key  = axis_map.get('name',  'Name')
#         for d in axis_data or []:
#             if d.get(class_key) == 'Axis' and str(d.get(name_key, '')).strip().lower() == key:
#                 return d
#         return None

#     def set_axis(self, axis_name: Optional[str], axis_data: Optional[List[Dict]] = None,
#                  axis_index: Optional[Dict[str, Dict]] = None):
#         if axis_name:
#             self.axis_name = axis_name

#         axis_dict = self._axis_dict_from(axis_data, axis_index)
#         if not axis_dict:
#             logger.warning("No Axis data found for axis_name=%r", self.axis_name)
#             self.axis_obj = None
#             return

#         axis_map = mapping.get(Axis, {})
#         stations_key = axis_map.get('stations',  'StaionValue')
#         x_coords_key = axis_map.get('x_coords', 'CurvCoorX')
#         y_coords_key = axis_map.get('y_coords', 'CurvCoorY')
#         z_coords_key = axis_map.get('z_coords', 'CurvCoorZ')

#         try:
#             stations = [float(s) for s in axis_dict.get(stations_key, [])]
#             x_coords = [float(x) for x in axis_dict.get(x_coords_key, [])]
#             y_coords = [float(y) for y in axis_dict.get(y_coords_key, [])]
#             z_coords = [float(z) for z in axis_dict.get(z_coords_key, [])]
#             self.axis_obj = get_axis_cached(self.axis_name, stations, x_coords, y_coords, z_coords)
#         except Exception as e:
#             logger.warning("Failed to create Axis object for %s: %s", self.axis_name, e)
#             self.axis_obj = None

#     # ---------- transform caching ----------
#     def _axis_signature(self) -> Tuple[str, str, str, str, str]:
#         if not self.axis_obj:
#             return ("", "", "", "", "m")
#         return (
#             _tiny_hash(self.axis_obj.stations),
#             _tiny_hash(self.axis_obj.x_coords),
#             _tiny_hash(self.axis_obj.y_coords),
#             _tiny_hash(self.axis_obj.z_coords),
#             "mm",
#         )

#     @staticmethod
#     def _points_signature(nested_points: List[List[Dict[str, float]]]) -> str:
#         flat = []
#         for ring in (nested_points or []):
#             for p in (ring or []):
#                 flat.append(f"{float(p.get('x',0)):.6f}:{float(p.get('y',0)):.6f}")
#             flat.append("|")
#         import hashlib
#         return hashlib.blake2b(",".join(flat).encode("utf-8"), digest_size=16).hexdigest()

#     def transform_points_cached(
#         self,
#         local_points: List[List[Dict[str, float]]],
#         stations_to_plot: List[float],
#         rotation_angle: Optional[float] = None,
#     ):
#         if self.axis_obj is None:
#             raise RuntimeError("axis_obj is not set. Call set_axis(...) first.")
#         rot = self.axis_rotation if rotation_angle is None else float(rotation_angle)
#         key = (
#             self._axis_signature(),
#             tuple(round(float(s), 8) for s in (stations_to_plot or [])),
#             round(rot, 6),
#             self._points_signature(local_points),
#         )
#         hit = BaseObject._TRANSFORM_CACHE.get(key)
#         if hit is not None:
#             return hit

#         result = self.axis_obj.transform_points(local_points, stations_to_plot, rotation_angle=rot)
#         if len(BaseObject._TRANSFORM_CACHE) >= BaseObject._TRANSFORM_CACHE_MAX:
#             BaseObject._TRANSFORM_CACHE.clear()
#         BaseObject._TRANSFORM_CACHE[key] = result
#         return result

#     # ---------- visualization input ----------
#     # inside BaseObject
#     def get_input_for_visualisation(
#         self,
#         *,
#         axis_data=None,
#         cross_section_objects=None,
#         json_file_path=None,
#         axis_rotation=None,
#         colors=None
#     ):
#         from .vis_adapter import create_input_for_visualisation
#         return create_input_for_visualisation(
#             obj=self,
#             axis_data=axis_data,
#             cross_section_objects=cross_section_objects,
#             json_file_path=json_file_path,
#             axis_rotation=axis_rotation,
#             colors=colors,
#         )

    
#     def with_axis(self, new_axis: Axis):
#         c = deepcopy(self)
#         c.axis_obj = new_axis
#         return c

#     def with_mainstations(self, new_ms: MainStation):
#         c = deepcopy(self)
#         c._mainstations = new_ms
#         return c

#     def _resolve_cross_sections_from_ncs(self, ctx: VisoContext) -> List[CrossSection]:
#         ncs_list = getattr(self, "cross_section_ncs", []) or []
#         cs_objs = []
#         for n in ncs_list:
#             cs = ctx.crosssec_by_ncs.get(int(n))
#             if cs: cs_objs.append(cs)
#         return cs_objs


# def _tiny_hash(arr) -> str:
#     import hashlib
#     if arr is None:
#         arr = []
#     data = (','.join(str(float(v)) for v in arr)).encode('utf-8')
#     return hashlib.blake2b(data, digest_size=12).hexdigest()

# # ---- Fast JSON lib (optional) ----
# try:
#     import orjson as _fastjson_AC
# except Exception:
#     _fastjson_AC = None

# # ------- Axis object cache (dedupe identical axes across objects) -------
# _AXIS_CACHE: dict = {}

# def get_axis_cached(axis_name, stations, x_coords, y_coords, z_coords):
#     """
#     Return a cached Axis with identical data (by value), or create & cache it.
#     axis_name participates in the key to avoid accidental collisions across
#     differently named axes that happen to share identical arrays.
#     """
#     # Normalize + hashable key
#     name_key = (axis_name or "").strip().lower()
#     key = (
#         name_key,
#         tuple(float(s) for s in (stations or [])),
#         tuple(float(v) for v in (x_coords or [])),
#         tuple(float(v) for v in (y_coords or [])),
#         tuple(float(v) for v in (z_coords or [])),
#     )
#     ax = _AXIS_CACHE.get(key)
#     if ax is None:
#         ax = Axis(stations=stations, x_coords=x_coords, y_coords=y_coords, z_coords=z_coords)
#         # keep cache bounded
#         if len(_AXIS_CACHE) > 128:
#             _AXIS_CACHE.clear()
#         _AXIS_CACHE[key] = ax
#     return ax

# # def build_axis_index(axis_data: List[Dict]) -> Dict[str, Dict]:
# #         axis_map = mapping.get(Axis, {})
# #         class_key = axis_map.get('class', 'Class')
# #         name_key  = axis_map.get('name',  'Name')
# #         idx = {}
# #         for d in axis_data or []:
# #             if d.get(class_key) == 'Axis':
# #                 name = str(d.get(name_key, '')).strip().lower()
# #                 if name:
# #                     idx[name] = d
# #         return idx

# # def _build_axis_index(axis_rows: list[dict], *, axis_map: dict) -> dict[str, dict]:
# #     """
# #     Returns { axis_name(str) : normalized_axis_dict }
# #     axis_map is your raw->internal keys map for "Axis" (from the facade).
# #     """
# #     # Tolerate the misspelling "StaionValue" in the mapping on purpose
# #     key_class    = next((k for k, v in axis_map.items() if v.lower() == 'class'), None) or 'Class'
# #     key_name     = next((k for k, v in axis_map.items() if v.lower() == 'name'), None) or 'Name'
# #     key_station  = next((k for k, v in axis_map.items() if v.lower() in ('stationvalue', 'staionvalue')), None) or 'StaionValue'
# #     key_x        = next((k for k, v in axis_map.items() if v.lower() == 'x_coords'), None) or 'CurvCoorX'
# #     key_y        = next((k for k, v in axis_map.items() if v.lower() == 'y_coords'), None) or 'CurvCoorY'
# #     key_z        = next((k for k, v in axis_map.items() if v.lower() == 'z_coords'), None) or 'CurvCoorZ'

# #     idx: dict[str, dict] = {}
# #     for row in axis_rows or []:
# #         if (row.get(key_class) or '').strip().lower() != 'axis':
# #             continue
# #         name = str(row.get(key_name, '')).strip()
# #         if not name:
# #             continue

# #         # Normalize to your canonical structure once
# #         stations = row.get(key_station, []) or []
# #         x = row.get(key_x, []) or []
# #         y = row.get(key_y, []) or []
# #         z = row.get(key_z, []) or []
# #         # ensure numeric lists
# #         def _nums(seq):
# #             out = []
# #             for v in seq:
# #                 try:
# #                     out.append(float(v))
# #                 except Exception:
# #                     pass
# #             return out

# #         norm = {
# #             'class': 'Axis',
# #             'name': name,
# #             'stations': _nums(stations),
# #             'x_coords': _nums(x),
# #             'y_coords': _nums(y),
# #             'z_coords': _nums(z),
# #         }
# #         idx[name] = norm

# #     return idx

# def build_cross_section_index(cross_sections: List[CrossSection]) -> Dict[int, CrossSection]:
#         out = {}
#         for cs in cross_sections or []:
#             try:
#                 out[int(cs.ncs)] = cs
#             except Exception:
#                 pass
#         return out    

# def build_viso_object(
#     obj: BaseObject, 
#     ctx: VisoContext,
#     *,
#     axis: Optional[Axis] = None,
#     mainstations: Optional[MainStation] = None,
#     cross_sections_override: Optional[List[CrossSection]] = None,
#     mapping_cfg: Dict[type, Dict[str, str]] = None
# ) -> dict:
#     """
#     Resolve dependencies for `obj` and return the classic vis dict
#     through your existing create_input_for_visualisation path.
#     No writes, no JSON mutation.
#     """
#     mapping_cfg = mapping_cfg or mapping

#     # Axis: explicit override > ctx by name > existing self.axis_obj
#     ax = axis or getattr(obj, "axis_obj", None)
#     if ax is None:
#         ax_name = getattr(obj, "axis_name", None)
#         if ax_name:
#             ax = ctx.axes_by_name.get(str(ax_name).strip())
#     obj.axis_obj = ax  # safe to set; this is the in-memory dataclass

#     # MainStations if you model them later (optional for now)
#     if mainstations is not None:
#         obj._mainstations = mainstations

#     # CrossSections (objects, not filenames)
#     if cross_sections_override is not None:
#         obj._cross_sections = list(cross_sections_override)
#     else:
#         obj._cross_sections = getattr(obj, "cross_sections", None) or obj._resolve_cross_sections_from_ncs(ctx)

#     # Axis variables: keep your current raw list and let your existing path build objects
#     if hasattr(obj, "axis_variables") and isinstance(obj.axis_variables, list):
#         obj.set_axis_variables(mapping_cfg.get(AxisVariable, {}))


#     vis_row = obj.get_input_for_visualisation(
#         axis_data=None,                      # or whatever you called your raw axis dicts
#         cross_section_objects=obj._cross_sections,   # your list of CrossSection objects
#         json_file_path=None,
#         axis_rotation=getattr(obj, "axis_rotation", 0),
#         colors=getattr(obj, "colors", None)
#     )

#     # # This helper exists in your tree and already supports passing cross_section_objects.
#     # vis_row = obj.get_input_for_visualisation(
#     #     cross_section_objects=obj._cross_sections,
#     #     axis_data=None,                   # not needed if axis_obj is set
#     #     json_file_path=None,              # keep file path resolution as-is
#     #     axis_rotation=getattr(obj, "axis_rotation", None),
#     #     colors=getattr(obj, "colors", None),
#     # )

#     return vis_row

# # --- helpers for AxisVariables ---
# def _as_floats(val) -> list[float]:
#     out = []
#     if val is None: 
#         return out
#     if not isinstance(val, (list, tuple)):
#         val = [val]
#     for v in val:
#         try:
#             out.append(float(v))
#         except Exception:
#             pass
#     return out

# def _as_str_list(val) -> list[str]:
#     if val is None:
#         return []
#     if isinstance(val, str):
#         return [val]
#     if isinstance(val, (list, tuple)):
#         return [str(x) for x in val]
#     return [str(val)]

# def _axis_vars_signature(rows: list[dict]) -> str:
#     # stable signature so we can skip recompute
#     import hashlib, json
#     try:
#         blob = json.dumps(rows or [], sort_keys=True, ensure_ascii=False, default=str)
#     except Exception:
#         blob = str(rows)
#     return hashlib.blake2b(blob.encode("utf-8"), digest_size=16).hexdigest()

# def create_axis_variables(json_data: list[dict], var_map: dict) -> list[AxisVariable]:
#     """
#     Normalize arbitrary AxisVariable rows using mapping keys and build objects.
#     Expects these *post-mapped* keys to exist (your mapping already uses them):
#       - "VariableName", "VariableStations", "VariableValues", "VariableIntTypes", "VariableDescription"
#     """
#     name_k = var_map.get("VariableName",      "VariableName")
#     xs_k   = var_map.get("VariableStations",  "VariableStations")
#     ys_k   = var_map.get("VariableValues",    "VariableValues")
#     typ_k  = var_map.get("VariableIntTypes",  "VariableIntTypes")
#     desc_k = var_map.get("VariableDescription","VariableDescription")

#     normalized: list[dict] = []
#     for row in (json_data or []):
#         # first apply mapping (identity-friendly)
#         mapped = { var_map.get(k, k): v for k, v in row.items() }

#         name = str(mapped.get(name_k, "")).strip()
#         xs   = _as_floats(mapped.get(xs_k))
#         ys   = _as_floats(mapped.get(ys_k))
#         tys  = _as_str_list(mapped.get(typ_k))
#         desc = mapped.get(desc_k, "")

#         # align lengths defensively
#         if len(xs) != len(ys):
#             n = min(len(xs), len(ys))
#             xs, ys = xs[:n], ys[:n]

#         # sort by station
#         pairs = sorted(zip(xs, ys), key=lambda t: t[0])
#         xs = [p[0] for p in pairs]
#         ys = [p[1] for p in pairs]

#         if not name or not xs:
#             continue

#         normalized.append({
#             name_k: name,
#             xs_k:   xs,
#             ys_k:   ys,
#             typ_k:  tys or ["linear"],
#             desc_k: desc,
#         })

#     # hand off to your class factory
#     return AxisVariable.create_axis_variables(normalized)

# # # --- typing helpers for robust type checks ---
# # from typing import get_origin, get_args, Union, ForwardRef

# # def _is_axis_annotation(t) -> bool:
# #     """True if 't' is Axis, Optional[Axis], Union[..., Axis, ...] or a ForwardRef('Axis')."""
# #     try:
# #         from models.axis import Axis
# #     except Exception:
# #         Axis = None  # graceful if import path changes

# #     if t is None or Axis is None:
# #         return False

# #     # direct match
# #     if t is Axis:
# #         return True

# #     # typing.Optional / typing.Union
# #     origin = get_origin(t)
# #     if origin is Union:
# #         return any(_is_axis_annotation(a) for a in get_args(t))

# #     # ForwardRef('Axis') or 'Axis' as string
# #     if isinstance(t, ForwardRef):
# #         return t.__forward_arg__ == 'Axis'
# #     if isinstance(t, str):
# #         return t.split('.')[-1] == 'Axis'

# #     # subclass (defensive)
# #     try:
# #         return issubclass(t, Axis)
# #     except Exception:
# #         pass

# #     # last-resort: name comparison (handles duplicated modules)
# #     return getattr(t, '__name__', '') == 'Axis'


# # def _is_list_of_dict_annotation(t) -> bool:
# #     """True if t is List[Dict] or list[dict]."""
# #     origin = get_origin(t)
# #     if origin not in (list, List):
# #         return False
# #     args = get_args(t) or ()
# #     return any(a in (dict, Dict) for a in args)

# # def from_dict(cls: Type, data: Dict, mapping_config: Dict[Type, Dict[str, str]], axis_data: List[Dict] = None):
# #     m = mapping_config  # (or your facade) 
# #     key_map = m.get(cls, {})

# #     # Build axis index once
# #     if axis_data:
# #         axis_map = m.get(Axis, {})
# #         if axis_map is not None and "_axis_index" not in axis_map:
# #             axis_map["_axis_index"] = _build_axis_index(axis_data, axis_map)
# #             m[Axis] = axis_map  # store back

# #     field_info = {f.name: f.type for f in fields(cls)}
# #     init_kwargs = {}

# #     # JSON key for axis_name (e.g., "Axis@Name")
# #     reverse_key_map = {v: k for k, v in key_map.items()}
# #     axis_name_key = reverse_key_map.get('axis_name', 'axis_name')
# #     print(f"Using axis_name_key: {axis_name_key}")  # Debug


# #     # First pass: map raw fields
# #     for json_key, value in data.items():
# #         field_name = key_map.get(json_key, json_key)
# #         if field_name not in field_info:
# #             continue

# #         annot = field_info[field_name]

# #         if field_name == 'axis_variables' and isinstance(value, list) and _is_list_of_dict_annotation(annot):
# #             # keep raw; you already build AxisVariable objects later
# #             init_kwargs[field_name] = value
# #             print(f"Processing axis_variables: {value}")  # Debug

# #         elif field_name == 'axis_obj' and _is_axis_annotation(annot):
# #             # store name only; actual Axis object is built after instance alloc
# #             init_kwargs['axis_name'] = data.get(axis_name_key, init_kwargs.get('axis_name', ''))
# #             # leave axis_obj unset to be populated later
# #             # (dataclass default None is fine)

# #         elif is_dataclass(annot) and isinstance(value, dict):
# #             init_kwargs[field_name] = from_dict(annot, value, m, axis_data)

# #         else:
# #             init_kwargs[field_name] = value
# #             print(f"Skipping unmapped field: {field_name}")  # Debug


# #     # Some files omit a few optional keys for PierObject; keep your existing defaults
# #     if cls.__name__ == "PierObject":
# #         init_kwargs.setdefault("top_cross_section_points_name", "")
# #         init_kwargs.setdefault("bot_cross_section_points_name", "")
# #         init_kwargs.setdefault("internal_station_value", [])

# #     # Create the instance
# #     instance = cls(**init_kwargs)

# #     # If this class has an Axis, build it now
# #     if 'axis_obj' in field_info and _is_axis_annotation(field_info['axis_obj']):
# #         axis_index = (m.get(Axis) or {}).get("_axis_index")
# #         instance.set_axis(getattr(instance, 'axis_name', ''), axis_data, axis_index)

# #     print(f"Creating {cls.__name__} with kwargs: {init_kwargs}")  # Debug
# #     return instance


# # ── base.py ─────────────────────────────────────────────────────────────────────
# import os
# from dataclasses import fields
# from typing import Any, Union, get_origin, get_args

# # Global toggle (or pass debug=True when you call from_dict)
# DEBUG_FROM_DICT = os.environ.get("SPOT_DEBUG_FROM_DICT", "0").strip() in ("1", "true", "True")
# DEBUG_FROM_DICT = True
# def _dbg(msg: str):
#     if DEBUG_FROM_DICT:
#         print(msg)

# # ── Robust Axis type detector ───────────────────────────────────────────────────
# def _is_axis_type(tp) -> bool:
#     """
#     Returns True if the type annotation contains models.axis.Axis anywhere:
#     Axis, Optional[Axis], Union[Axis, None], List[Axis], Tuple[Axis, ...],
#     or a forward-ref string like "Axis" / "models.axis.Axis".
#     """
#     if tp is None:
#         return False

#     # string forward-ref
#     if isinstance(tp, str):
#         return tp.split(".")[-1] == "Axis"

#     AxisClass = None
#     try:
#         from models.axis import Axis as AxisClass  # your real Axis class
#     except Exception:
#         pass

#     if AxisClass is not None and tp is AxisClass:
#         return True

#     origin = get_origin(tp) or getattr(tp, "__origin__", None)
#     args   = get_args(tp)   or getattr(tp, "__args__", ())

#     if origin is None:
#         # tolerate direct class compare by name
#         return getattr(tp, "__name__", "") == "Axis"

#     # containers
#     if origin in (list, set, tuple):
#         return any(_is_axis_type(a) for a in args)

#     # Optional/Union
#     if origin is Union:
#         return any(_is_axis_type(a) for a in args if a is not type(None))

#     return False


# # ── Axis index builder ──────────────────────────────────────────────────────────
# def _build_axis_index(axis_rows: list[dict], *, axis_map: dict[str, str]) -> dict[str, dict]:
#     """
#     Returns { axis_name(str) : normalized_axis_dict } using the raw->internal map for Axis.
#     Normalizes keys: name, stations, x_coords, y_coords, z_coords.
#     """
#     # find raw keys with tolerance to typos/case
#     def _raw_for(internal_name: str, fallback: str) -> str:
#         for rk, iv in (axis_map or {}).items():
#             if str(iv).lower() == internal_name.lower():
#                 return rk
#         return fallback

#     key_class = _raw_for("Class",        "Class")
#     key_name  = _raw_for("Name",         "Name")
#     key_stat  = _raw_for("stations",     "StaionValue")  # note: 'StaionValue' in mapping
#     key_x     = _raw_for("x_coords",     "CurvCoorX")
#     key_y     = _raw_for("y_coords",     "CurvCoorY")
#     key_z     = _raw_for("z_coords",     "CurvCoorZ")

#     def _nums(seq):
#         out = []
#         for v in (seq or []):
#             try:
#                 out.append(float(v))
#             except Exception:
#                 pass
#         return out

#     idx: dict[str, dict] = {}
#     for row in axis_rows or []:
#         if str(row.get(key_class, "")).strip().lower() != "axis":
#             continue
#         name = str(row.get(key_name, "")).strip()
#         if not name:
#             continue
#         norm = {
#             "class": "Axis",
#             "name": name,
#             "stations": _nums(row.get(key_stat)),
#             "x_coords": _nums(row.get(key_x)),
#             "y_coords": _nums(row.get(key_y)),
#             "z_coords": _nums(row.get(key_z)),
#         }
#         idx[name] = norm
#     return idx


# # ── Multi-candidate raw-key resolver ────────────────────────────────────────────
# def _raw_candidates_for(internal: str, cls_map: dict[str, str]) -> list[str]:
#     """
#     All raw keys that map to this internal field (handles aliases like
#     'Top-CrossSection@Ncs' and 'Top-CrossSection@NCS').
#     """
#     return [rk for rk, v in (cls_map or {}).items() if v == internal]

# def _pick_raw_key(internal: str, cls_map: dict[str, str], data: dict, fallback_same_name=True) -> str | None:
#     """
#     Pick the first raw key for 'internal' that exists in 'data'.
#     If none exists and fallback_same_name is True, try the internal name itself.
#     """
#     for k in _raw_candidates_for(internal, cls_map):
#         if k in data:
#             return k
#     if fallback_same_name and internal in data:
#         return internal
#     return None


# # ── Safe, light coercion ───────────────────────────────────────────────────────
# def _coerce_value(val: Any, tp):
#     origin = get_origin(tp) or getattr(tp, "__origin__", None)
#     args   = get_args(tp)   or getattr(tp, "__args__", ())

#     def _to_num(v, caster):
#         try:
#             if isinstance(v, str) and not v.strip():
#                 return v
#             return caster(v)
#         except Exception:
#             return v

#     if tp in (int, float, bool):
#         caster = {int: int, float: float, bool: lambda x: bool(int(x)) if isinstance(x, (str, int)) else bool(x)}[tp]
#         return _to_num(val, caster)

#     if origin is Union:
#         for a in args:
#             if a is type(None):
#                 continue
#             return _coerce_value(val, a)

#     if origin in (list, set, tuple):
#         inner = args[0] if args else Any
#         seq = list(val) if isinstance(val, (list, tuple, set)) else [val]
#         coerced = [_coerce_value(v, inner) for v in seq]
#         return type(val)(coerced) if isinstance(val, (list, tuple, set)) else coerced

#     return val


# # ── The function you call from load_from_json ───────────────────────────────────
# def from_dict(cls, data: dict, mapping_config, axis_data=None, debug: bool | None = None):
#     """
#     Create an instance of `cls` from a raw row `data` using `mapping_config`.
#     - Detects and resolves an Axis field if present (no mutation of mapping facade).
#     - Uses class-specific raw->internal key mapping with multi-key candidates.
#     - Lightly coerces simple numeric types.
#     """
#     if debug is None:
#         debug = DEBUG_FROM_DICT

#     clsname = getattr(cls, "__name__", str(cls))
#     cls_map = mapping_config.get(cls, {}) or {}
#     if debug:
#         _dbg(f"[from_dict] class={clsname}  keys_in_row={list(data.keys())[:6]}...  axis_data provided: {bool(axis_data)}")

#     # Collect dataclass field defs
#     field_defs = {f.name: f for f in fields(cls)}

#     # ── Axis wiring (if any) ────────────────────────────────────────────────────
#     axis_field_name = next((n for n, f in field_defs.items() if _is_axis_type(f.type)), None)
#     if axis_field_name and axis_data:
#         if debug:
#             _dbg(f"[from_dict] Axis typed field detected: {axis_field_name} : {field_defs[axis_field_name].type}")

#         # pick raw key for axis name (aliases supported)
#         raw_axis_key = _pick_raw_key("axis_name", cls_map, data) or "Axis@Name"
#         axis_name_value = data.get(raw_axis_key)

#         if debug:
#             _dbg(f"[from_dict] axis_name candidates={_raw_candidates_for('axis_name', cls_map)}  picked={raw_axis_key}  value={axis_name_value!r}")

#         # build local index & instantiate Axis
#         axis_map = mapping_config.get("Axis", {})  # your facade allows key by string name
#         axis_index = _build_axis_index(axis_data, axis_map=axis_map)
#         axis_dict = axis_index.get(str(axis_name_value)) if axis_name_value is not None else None

#         if axis_dict:
#             try:
#                 from models.axis import Axis as AxisCls
#                 obj_axis = AxisCls(**axis_dict)
#                 _dbg(f"[from_dict] axis resolved & instantiated: {axis_name_value!r}")
#             except Exception as e:
#                 obj_axis = axis_dict
#                 _dbg(f"[from_dict] axis resolved to dict (fallback). reason={e}")
#             # set axis field
#             obj_kwargs = {axis_field_name: obj_axis}
#         else:
#             obj_kwargs = {}
#             if debug:
#                 _dbg(f"[from_dict] axis name {axis_name_value!r} not found in axis index.")
#         # keep the name too if there is a field for it
#         if "axis_name" in field_defs and axis_name_value is not None:
#             obj_kwargs["axis_name"] = axis_name_value
#     else:
#         obj_kwargs = {}

#     # ── Map all other fields using multi-candidate lookup ───────────────────────
#     for fname, fdef in field_defs.items():
#         if fname in obj_kwargs:
#             continue  # already set (axis fields)

#         raw_key = _pick_raw_key(fname, cls_map, data)
#         if raw_key is None:
#             continue

#         val = data.get(raw_key)
#         coerced = _coerce_value(val, fdef.type)
#         obj_kwargs[fname] = coerced

#         if debug and fname in (
#             "top_cross_section_ncs", "bot_cross_section_ncs", "internal_cross_section_ncs",
#             "internal_station_value", "ref_station_offset", "station_value",
#             "bot_zoffset", "top_zoffset"
#         ):
#             _dbg(f"[from_dict] {clsname}.{fname} <- {raw_key}  value={coerced!r}")

#     # ── Pier/Foundation sanity logs (helpful for your current issue) ────────────
#     if debug and clsname in ("PierObject", "FoundationObject"):
#         critical = [k for k in ("top_cross_section_ncs", "bot_cross_section_ncs", "internal_cross_section_ncs") if k in field_defs]
#         for k in critical:
#             if k not in obj_kwargs:
#                 _dbg(f"[WARN] {clsname} missing field '{k}'. candidates={_raw_candidates_for(k, cls_map)}  row_has={list(data.keys())}")

#     if debug:
#         _dbg(f"[from_dict] Creating {clsname} with kwargs: {obj_kwargs}")

#     # Construct the dataclass object
#     return cls(**obj_kwargs)



# @staticmethod
# def load_from_json(
#     cls: Type,
#     data_or_file: Union[str, List[Dict], Dict],
#     mapping_config: Dict[Type, Dict[str, str]] = None,
#     axis_data: List[Dict] = None
# ) -> Tuple[Union[List, object], List[Dict]]:
#     """
#     Load objects of type `cls` from a JSON file, list of dicts, or a single dict.
#     Filters out entries where Class is "ClassInfo" and returns both objects and filtered data.
#     """
#     mapping_config = mapping_config or mapping  # Use global mapping if none provided

#     if isinstance(data_or_file, str):
#         if _fastjson_AC:
#             with open(data_or_file, "rb") as f:
#                 data = _fastjson_AC.loads(f.read())
#         else:
#             with open(data_or_file, "r", encoding="utf-8") as f:
#                 data = json.load(f)
#     else:
#         data = data_or_file

#     print(f"axis_data provided: {axis_data is not None}")  # Debug

#     if isinstance(data, list):
#         # Filter out entries where Class is "ClassInfo"
#         filtered_data = [d for d in data if d.get("Class") != "ClassInfo"]
#         objects = [from_dict(cls, obj, mapping_config, axis_data) for obj in filtered_data]
#         return objects, filtered_data
#     elif isinstance(data, dict):
#         if data.get("Class") == "ClassInfo":
#             return None, []
#         obj = from_dict(cls, data, mapping_config, axis_data)
#         return obj, [data]
#     else:
#         raise ValueError("data_or_file must be a JSON file path, a list of dicts, or a dict.")

# def create_input_for_visualisation(
#     obj: object,
#     *,
#     axis_data: list | None = None,
#     cross_section_objects: list | None = None,
#     json_file_path: str | None = None,
#     axis_rotation: float | None = None,
#     colors: dict | None = None,
# ) -> dict:
#     """
#     Create input for visualisation with a reproducible axis signature.
#     """
#     import hashlib
#     import numpy as np  # <— add this

#     # Colors & rotation defaults
#     colors = colors or getattr(obj, 'colors', {
#         "axis": "black",
#         "first_station_points": "blue",
#         "cross_section_loops": "red",
#         "longitudinal_lines": "gray"
#     })
#     axis_rotation = axis_rotation if axis_rotation is not None else getattr(obj, 'axis_rotation', 0)

#     # inside create_input_for_visualisation
#     def _to_list(x):
#         # robust against numpy arrays, scalars, None, strings
#         try:
#             import numpy as np
#             if isinstance(x, np.ndarray): 
#                 return [float(v) for v in x.tolist()]
#         except Exception:
#             pass
#         if x is None or x == "":
#             return []
#         if isinstance(x, (list, tuple, set)):
#             return [float(v) for v in x]
#         if isinstance(x, (int, float)):
#             return [float(x)]
#         # last ditch: ignore
#         return []

#     # combine possible station sources
#     station_value_main   = _to_list(getattr(obj, 'station_value', None))
#     station_value_inner  = _to_list(getattr(obj, 'internal_station_value', None))
#     user_stations        = _to_list(getattr(obj, 'user_stations', None))

#     base_breaks = sorted(set(station_value_main + station_value_inner + user_stations))

#     # Interpolate segments between "breakpoints" to get dense stations_to_plot
#     def interpolate_segments(breakpoints: List[float]) -> List[float]:
#         if len(breakpoints) < 2:
#             return breakpoints
#         out = []
#         for a, b in zip(breakpoints[:-1], breakpoints[1:]):
#             dist = b - a
#             if dist <= 0:
#                 continue
#             if   dist <   5: steps = 2
#             elif dist <  10: steps = 3
#             elif dist <  50: steps = 10
#             elif dist < 100: steps = 20
#             elif dist < 200: steps = 40
#             elif dist < 300: steps = 30
#             elif dist < 400: steps = 40
#             elif dist < 500: steps = 50
#             else:             steps = 500
#             step = dist / steps
#             out.extend(round(a + j * step, 8) for j in range(steps))
#         out.append(round(breakpoints[-1], 8))
#         return out
#     stations_to_plot = interpolate_segments(base_breaks)

#     # Map Axis keys from mapping config
    
#     axis_map = mapping.get(Axis, {})
#     class_key     = axis_map.get('class', 'Class')
#     name_key      = axis_map.get('name', 'Name')
#     stations_key  = axis_map.get('stations', 'StaionValue')
#     x_coords_key  = axis_map.get('x_coords', 'CurvCoorX')
#     y_coords_key  = axis_map.get('y_coords', 'CurvCoorY')
#     z_coords_key  = axis_map.get('z_coords', 'CurvCoorZ')

#     ...
#     # Prefer an already-resolved Axis object; otherwise fall back to raw axis_data
#     def _as_float_list(val):
#         """Robustly turn None / scalar / list / tuple / np.ndarray into List[float]."""
#         if val is None:
#             return []
#         if isinstance(val, np.ndarray):
#             val = val.ravel().tolist()
#         elif isinstance(val, (list, tuple)):
#             val = list(val)
#         else:
#             val = [val]
#         out = []
#         for v in val:
#             try:
#                 out.append(float(v))
#             except Exception:
#                 # keep going; skip non-numeric junk rather than crashing
#                 pass
#         return out

#     stations = x_coords = y_coords = z_coords = []
#     if getattr(obj, "axis_obj", None) is not None:
#         ax = obj.axis_obj
#         stations = _as_float_list(getattr(ax, "stations", None))
#         x_coords = _as_float_list(getattr(ax, "x_coords", None))
#         y_coords = _as_float_list(getattr(ax, "y_coords", None))
#         z_coords = _as_float_list(getattr(ax, "z_coords", None))
#     else:
#         # Find axis dict from raw data (axis_data may be None)
#         axis_dict = None
#         axis_name = getattr(obj, 'axis_name', '') or ''
#         for data in (axis_data or []):  # safe when axis_data is None
#             if (data.get(class_key) == 'Axis' and str(data.get(name_key, '')).lower() == axis_name.lower()):
#                 axis_dict = data
#                 break
#         if axis_dict:
#             stations = _as_float_list(axis_dict.get(stations_key))
#             x_coords = _as_float_list(axis_dict.get(x_coords_key))
#             y_coords = _as_float_list(axis_dict.get(y_coords_key))
#             z_coords = _as_float_list(axis_dict.get(z_coords_key))

#     # AxisVariables to expected shape (keep yours)
#     axis_variables = [
#         {
#             "VariableName": var.name,
#             "VariableStations": var.xs,
#             "VariableValues": [str(v * 1000) for v in var.ys],  # as in your original
#             "VariableIntTypes": var.types,
#             "VariableDescription": "empty"
#         }
#         for var in getattr(obj, 'axis_variables_obj', [])
#     ]

#     # # Resolve cross-section JSON
#     # ncs_list = getattr(obj, 'cross_section_ncs', None)
#     # if not isinstance(ncs_list, list) or not ncs_list:
#     #     ncs_list = getattr(obj, 'internal_cross_section_ncs', None)
#     # final_json_file = "no MASTER SECTION defined!"
#     # if isinstance(ncs_list, list) and ncs_list:
#     #     ncs_value = ncs_list[0]
#     #     for cs_obj in cross_section_objects:
#     #         if cs_obj.ncs == ncs_value and cs_obj.json_name:
#     #             final_json_file = cs_obj.json_name[0]
#     #             break

#     # Resolve cross-section JSON
#     final_json_file = None
#     ncs_list = _to_list(getattr(obj, 'cross_section_ncs', None)) or _to_list(getattr(obj, 'internal_cross_section_ncs', None))

#     # Prefer exact ncs hit
#     if cross_section_objects:
#         cs_by_ncs  = {getattr(cs, 'ncs', None): cs for cs in cross_section_objects if getattr(cs, 'ncs', None) is not None}
#         cs_by_name = {str(getattr(cs, 'name', '')).lower(): cs for cs in cross_section_objects}

#         for ncs in ncs_list:
#             cs = cs_by_ncs.get(ncs)
#             if cs and getattr(cs, 'json_name', None):
#                 final_json_file = cs.json_name[0]
#                 break

#         # Fallback: try by name 'MASTER_*' if exact NCS has empty json_name
#         if not final_json_file:
#             # best-effort guesses based on object class/type
#             wanted = None
#             if getattr(obj, 'class_name', '') == 'DeckObject':
#                 wanted = 'MASTER_Deck'
#             elif getattr(obj, 'class_name', '') == 'PierObject':
#                 wanted = 'MASTER_Pier'
#             elif getattr(obj, 'class_name', '') == 'FoundationObject':
#                 wanted = 'MASTER_Foundation'

#             if wanted:
#                 cs = cs_by_name.get(wanted.lower())
#                 if cs and getattr(cs, 'json_name', None):
#                     final_json_file = cs.json_name[0]

#     # If still nothing, leave None; attach step won’t crash and you’ll still see guide lines
#     row_json_file = final_json_file or ""

#     final_json_file_full = find_relative_file(final_json_file)
#     # NEW: fallback to the relative path so SpotLoader can join with branch folder
#     json_file_out = final_json_file_full or final_json_file

#     # Create an axis signature to help Axis cache hits later
#     def tiny_hash(arr):
#         data = (','.join(str(float(v)) for v in (arr or []))).encode('utf-8')
#         return hashlib.blake2b(data, digest_size=12).hexdigest()
#     axis_sig = (tiny_hash(stations), tiny_hash(x_coords), tiny_hash(y_coords), tiny_hash(z_coords), 'm')

#     return {
#         "json_file": json_file_out,
#         "stations_axis": stations,
#         "x_coords": x_coords,
#         "y_coords": y_coords,
#         "z_coords": z_coords,
#         "stations_to_plot": stations_to_plot,
#         "AxisVariables": axis_variables,
#         "name": getattr(obj, 'name', ''),
#         "colors": colors,
#         "AxisRotation": axis_rotation,
#         "axis_signature": axis_sig,   # <--- NEW
#     }

# from functools import lru_cache
# @lru_cache(maxsize=512)
# def find_relative_file(rel_path: str, verbose: bool = False) -> str | None:
#     """
#     Fast path: direct joins against likely bases (script/exe/MEIPASS) + filename match.
#     Fallback: your original multi-level search (kept intact).
#     Results are cached so repeated lookups are near O(1).
#     """
#     if not rel_path:
#         return None

#     import os, sys
#     rel_path = os.path.normpath(rel_path)
#     file_name = os.path.basename(rel_path)

#     # 1) Quick bases
#     bases = []
#     if getattr(sys, "frozen", False):
#         bases.extend([getattr(sys, "_MEIPASS", ""), os.path.dirname(sys.executable)])
#     else:
#         bases.append(os.path.abspath(os.path.dirname(__file__)))
#     bases = [b for b in bases if b]

#     # try direct join (fast)
#     for base in bases:
#         p = os.path.join(base, rel_path)
#         if os.path.exists(p):
#             return os.path.abspath(p)

#     # also try “found by filename” inside base (1–2 levels down)
#     for base in bases:
#         for dirpath, _, filenames in os.walk(base):
#             # shallow to keep fast
#             depth = len(os.path.relpath(dirpath, base).split(os.sep))
#             if depth > 2:
#                 continue
#             if file_name.lower() in (f.lower() for f in filenames):
#                 candidate = os.path.join(dirpath, file_name)
#                 # make sure the tail matches the relative path shape
#                 if os.path.normpath(candidate).lower().endswith(os.path.normpath(rel_path).lower()):
#                     return os.path.abspath(candidate)

#     # 2) Fallback to your full (slower but thorough) search — unchanged logic
#     # --- begin fallback (your previous implementation) ---
#     # (Copy of your original deep search trimmed into a helper)
#     def _deep_search():
#         base_dirs = []
#         if getattr(sys, 'frozen', False):
#             exe_dir = os.path.dirname(sys.executable)
#             base_dirs += [getattr(sys, "_MEIPASS", ""), exe_dir,
#                           os.path.dirname(exe_dir),
#                           os.path.dirname(os.path.dirname(exe_dir)),
#                           os.path.dirname(os.path.dirname(os.path.dirname(exe_dir))),
#                           os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(exe_dir))))]
#         else:
#             script_dir = os.path.abspath(os.path.dirname(__file__))
#             base_dirs += [script_dir,
#                           os.path.dirname(script_dir),
#                           os.path.dirname(os.path.dirname(script_dir)),
#                           os.path.dirname(os.path.dirname(os.path.dirname(script_dir))),
#                           os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))]
#         base_dirs = list(dict.fromkeys([b for b in base_dirs if b]))

#         def check_file_path(full_path: str) -> str | None:
#             if not os.path.exists(full_path):
#                 return None
#             return os.path.abspath(full_path)

#         # Search down first (up to 4 levels), then up (up to 4 levels)
#         for base_dir in base_dirs:
#             current_dir = base_dir
#             for level in range(5):
#                 if level == 0:
#                     full_path = os.path.join(current_dir, rel_path)
#                     res = check_file_path(full_path)
#                     if res: return res
#                 else:
#                     try:
#                         subdirs = [d for d in os.listdir(current_dir)
#                                    if os.path.isdir(os.path.join(current_dir, d))]
#                     except OSError:
#                         break
#                     for subdir in subdirs:
#                         full_path = os.path.join(current_dir, subdir, rel_path)
#                         res = check_file_path(full_path)
#                         if res: return res
#                     if not subdirs: break
#                     current_dir = os.path.join(current_dir, subdirs[0])

#         for base_dir in base_dirs:
#             current_dir = base_dir
#             for _ in range(4):
#                 parent = os.path.dirname(current_dir)
#                 if parent == current_dir: break
#                 full_path = os.path.join(parent, rel_path)
#                 res = check_file_path(full_path)
#                 if res: return res
#                 current_dir = parent

#         return None
#     # --- end fallback ---

#     res = _deep_search()
#     if not res and verbose:
#         print(f"[find_relative_file] Not found: {rel_path}")
#     return res

