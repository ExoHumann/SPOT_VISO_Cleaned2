from __future__ import annotations
# models/cross_section.py
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math
import numpy as np
import logging
import json, os

# Reuse your safe-eval & math maps
from Utils import (
    _compile_expr,
    _sanitize_vars,
    _SCALAR_FUNCS,
    _VECTOR_FUNCS,
    _RESERVED_FUNC_NAMES,
)

logger = logging.getLogger(__name__)


# ---------- small helpers (robust to many JSON shapes) ----------
def _deep_find_first(obj, key_names):
    """
    BFS search for the first list under any key matching one of key_names
    (case-insensitive). Returns (value, key_path_str) or (None, None).
    """
    key_names = {k.lower() for k in key_names}
    from collections import deque
    dq = deque([([], obj)])
    while dq:
        path, cur = dq.popleft()
        if isinstance(cur, dict):
            for k, v in cur.items():
                if k.lower() in key_names and isinstance(v, list) and v:
                    return v, ".".join(path + [k])
                dq.append((path + [k], v))
        elif isinstance(cur, list):
            for i, v in enumerate(cur):
                dq.append((path + [f"[{i}]"], v))
    return None, None

def _as_point_id_list(seq):
    """
    Accept loop 'Points' shapes like:
    - list of dicts with any of: Id, id, PointName, Name, PointId, PointID
    - list of strings
    - list of numbers
    Returns list[str]
    """
    out = []
    for p in (seq or []):
        if isinstance(p, dict):
            for k in ("Id","id","PointName","Name","PointId","PointID","Point"):
                if k in p and p[k] not in (None, ""):
                    out.append(str(p[k]))
                    break
        else:
            out.append(str(p))
    return [s for s in out if s]

def _normalize_loops_value(value):
    """
    Given whatever we found under Loops/Contours/Polygons/etc., normalize to:
      List[List[str]]  (each inner list is the ordered point ids for one loop)
    """
    loops = []

    if not isinstance(value, list):
        return loops

    # Case A: vanilla [{"Points":[...]}, ...]
    if value and all(isinstance(x, dict) for x in value):
        for item in value:
            pts = item.get("Points") or item.get("Point") or item.get("Vertices") or item.get("Ids")
            if pts:
                loops.append(_as_point_id_list(pts))
            elif isinstance(item.get("Loop"), list):
                loops.append(_as_point_id_list(item["Loop"]))
            elif isinstance(item.get("Ring"), list):
                loops.append(_as_point_id_list(item["Ring"]))
        if loops:
            return loops

    # Case B: directly a list of lists (each is a loop)
    if value and all(isinstance(x, list) for x in value):
        for seq in value:
            loops.append(_as_point_id_list(seq))
        return loops

    # Case C: list of dicts, where each dict is itself a loop mapping of index->point
    if value and all(isinstance(x, dict) for x in value):
        for item in value:
            # try any obvious fields again
            pts = item.get("Points") or item.get("Point") or item.get("Vertices") or item.get("Ids")
            if pts:
                loops.append(_as_point_id_list(pts))
    return loops

def _points_to_table(value: Any) -> Dict[str, Dict[str, Any]]:
    """
    Normalize "Points" to a dict: {id: {...raw point dict...}}
    Accepts dict or list-of-dicts.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(value, dict):
        # Already {id: {...}}
        for pid, row in value.items():
            out[str(pid)] = row if isinstance(row, dict) else {"Raw": row}
        return out

    if isinstance(value, list):
        for row in value:
            if not isinstance(row, dict):
                continue
            pid = (row.get("Id") or row.get("id") or row.get("Name") or
                   row.get("PointName") or row.get("PointId") or row.get("PointID"))
            if pid is not None:
                out[str(pid)] = row
    return out


@dataclass(kw_only=True)
class CrossSection:
    """
    CrossSection now *owns* all 2D point evaluation:

    - Variable defaults
    - DAG build & topological sort
    - Vectorized expression evaluation
    - (Deprecated) scalar reference frame paths
    - Vectorized local frame transforms (C/Euclid, P/Polar, CY, CZ)
    - Public API:
        get_defaults()
        build_dag()
        eval_expressions_vectorized(...)
        get_coordinates_vectorized(...)
        compute_local_points(...)              -> (ids, X_mm, Y_mm, loops_idx)
        compute_local_points_scalar(...)       -> legacy single-slice path
        compute_embedded_points(...)           -> (ids, stations_mm, P_mm, X_mm, Y_mm, loops_idx)
    """
    no: Optional[str] = None
    class_name: Optional[str] = None
    type: Optional[str] = None
    description: Optional[str] = None
    name: Optional[str] = None
    inactive: Optional[str] = None
    ncs: Optional[int] = None
    material1: Optional[int] = None
    material2: Optional[int] = None
    material_reinf: Optional[int] = None
    json_name: Union[str, List[str], None] = None      # e.g. "SectionData.json" or ["A.json","B.json"]
    points: Union[List[Dict[str, Any]], Dict[str, Any], None] = None  # optional inline points
    variables: Union[List[Dict[str, Any]], Dict[str, Any], None] = None
    sofi_code: str | None = None
    cross_section_types: list[str] | None = None
    axis_variables: list | None = None

    # runtime / enrichment
    json_data: Optional[Dict[str, Any]] = field(default=None, init=False, repr=False)
    json_source: Optional[str] = field(default=None, init=False, repr=False)

    # caches
    _loops_cache: Dict[Tuple[str, ...], List[np.ndarray]] = field(default_factory=dict, init=False, repr=False)
    # cached parse (by object identity)
    _dag_cache_key: Optional[str] = field(default=None, init=False, repr=False)
    _dag_cache_val: Optional[Tuple[List[str], Dict[str, dict]]] = field(default=None, init=False, repr=False)
    #_dag_order: List[str] = field(default_factory=list, init=False, repr=False)
    #_dag_by_id: Dict[str, dict] = field(default_factory=dict, init=False, repr=False)
    _loops_cache: Dict[Tuple[str, ...], List[np.ndarray]] = field(default_factory=dict, init=False, repr=False)


    # ---------- basic hygiene ----------
    def __post_init__(self) -> None:
        # Coerce numerics if present
        try:
            if self.ncs is not None and str(self.ncs).strip() != "":
                self.ncs = int(self.ncs)
        except Exception:
            logger.debug("CrossSection(%r): failed to int(ncs=%r)", self.name, self.ncs)

        for attr in ("material1", "material2", "material_reinf"):
            v = getattr(self, attr)
            try:
                if v is not None and str(v).strip() != "":
                    setattr(self, attr, int(v))
            except Exception:
                logger.debug("CrossSection(%r): failed to int(%s=%r)", self.name, attr, v)

    # ---------- defaults / variables ----------
    def get_defaults(self) -> Dict[str, float]:
        """
        Variable defaults in mm. Prefer JSON->Variables; fallback to row->variables.
        Supports dict or list-of-dicts [{'VariableName','VariableValue'}, ...]
        """
        # Prefer attached JSON
        if isinstance(self.json_data, dict):
            vars_dict = self.json_data.get("Variables")
            if isinstance(vars_dict, dict):
                out: Dict[str, float] = {}
                for k, v in vars_dict.items():
                    try:
                        out[str(k)] = float(v)
                    except Exception:
                        pass
                if out:
                    return out

        # Fallback to row field
        out: Dict[str, float] = {}
        raw = self.variables or {}
        if isinstance(raw, dict):
            for k, v in raw.items():
                try:
                    out[str(k)] = float(v)
                except Exception:
                    pass
        else:
            for row in raw or []:
                if not isinstance(row, dict):
                    continue
                try:
                    n = str(row.get("VariableName"))
                    out[n] = float(row.get("VariableValue", 0.0) or 0.0)
                except Exception:
                    pass
        return out

    # ---------- geometry attach / read ----------
    def attach_json_payload(self, payload: Dict[str, Any], source: Optional[str] = None) -> None:
        """Attach a section JSON payload (already read)."""
        if not isinstance(payload, dict):
            return
        self.json_data = payload
        self.json_source = source
        logger.debug(
            "CrossSection.attach_json_payload: name=%r ncs=%r source=%r",
            self.name, self.ncs, source
        )

    def _raw_points_and_loops(self) -> Tuple[Dict[str, Dict[str, Any]], List[List[str]]]:
        """
        Find 'Points' and 'Loops' from either attached JSON or inline row fields.
        Returns (points_table, loops_ids)
        """
        root = self.json_data if isinstance(self.json_data, dict) else (self.points or {})
        pts_val, _ = _deep_find_first(root, ["Points", "points"])
        lps_val, _ = _deep_find_first(root, ["Loops", "Contours", "Polygons", "Rings", "Boundaries"])

        points_table = _points_to_table(pts_val or {})
        loops_ids = _normalize_loops_value(lps_val or [])

        return points_table, loops_ids

    # lightweight summary hooks (used by your loader/enricher)
    def has_geometry(self) -> bool:
        points_table, loops_ids = self._raw_points_and_loops()
        return bool(points_table) and bool(loops_ids)

    def geometry_counts(self) -> Tuple[int, int]:
        """(n_points, n_loops) for logging/diagnostics."""
        points_table, loops_ids = self._raw_points_and_loops()
        return len(points_table), len(loops_ids)

    # convenience constructor for safety
    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "CrossSection":
        """Very permissive; only pulls what exists."""
        kw = {
            "no": row.get("No"),
            "class_name": row.get("Class") or row.get("class_name"),
            "type": row.get("Type"),
            "description": row.get("Description"),
            "name": row.get("Name"),
            "inactive": row.get("InActive"),

            "ncs": row.get("NCS") or row.get("Ncs") or row.get("ncs"),
            "material1": row.get("Material1"),
            "material2": row.get("Material2"),
            "material_reinf": row.get("Material_Reinf"),

            "json_name": row.get("JSON_name") or row.get("JsonName") or row.get("json_name"),
            "sofi_code": row.get("SofiCode") or row.get("SOFiCode") or row.get("SOFiSTiKCustomCode"),

            "points": row.get("Points"),
            "variables": row.get("Variables"),

            "cross_section_types": row.get("cross_section_types") or row.get("CrossSection_Types"),
            "axis_variables": row.get("axis_variables") or row.get("AxisVariables"),
        }
        return cls(**{k: v for k, v in kw.items() if v is not None})


 
    # -------------------------------------------------------------------------
    # Defaults / variables
    # -------------------------------------------------------------------------

    # def get_defaults(self) -> Dict[str, float]:
    #     """
    #     Read variable defaults either from attached json_data['Variables'] (preferred)
    #     or from the dataclass 'variables' field (list/dict). Values are floats in mm.
    #     """
    #     # 1) Master JSON attached (preferred)
    #     jd = getattr(self, "json_data", None)
    #     if isinstance(jd, dict) and isinstance(jd.get("Variables"), dict):
    #         out = {}
    #         for k, v in (jd.get("Variables") or {}).items():
    #             try:
    #                 out[str(k)] = float(v)
    #             except Exception:
    #                 pass
    #         if out:
    #             return out

    #     # 2) Fallback to the dataclass field `variables`
    #     defaults: Dict[str, float] = {}
    #     raw = (self.variables or {}) or {}
    #     if isinstance(raw, dict):
    #         for k, v in raw.items():
    #             try:
    #                 defaults[str(k)] = float(v)
    #             except Exception:
    #                 pass
    #     else:
    #         # list-of-dicts [{VariableName, VariableValue}, ...]
    #         for row in raw or []:
    #             try:
    #                 n = str(row.get("VariableName"))
    #                 defaults[n] = float(row.get("VariableValue", 0.0) or 0.0)
    #             except Exception:
    #                 pass
    #     return defaults

    # -------------------------------------------------------------------------
    # DAG build (topological sort for point dependencies)
    # -------------------------------------------------------------------------

    @staticmethod
    def _pick_expr(expr_val, numeric_val):
        txt = str(expr_val) if expr_val is not None else ""
        if not txt or txt.strip().lower().startswith("error"):
            if numeric_val is None or str(numeric_val).strip() == "":
                return "0"
            return str(numeric_val)
        return txt


    def _normalize_point_row(self, p: dict) -> dict:
        # Already canonical?
        if "Id" in p and ("Coord" in p or "coord" in p):
            pid = str(p.get("Id"))
            coord = p.get("Coord") or p.get("coord") or [0, 0]
            ref   = p.get("Reference") or p.get("reference") or []
            return {"Id": pid, "Coord": [str(coord[0]), str(coord[1])], "Reference": list(ref)}

        # Legacy shape
        pid = str(p.get("PointName") or p.get("Id") or p.get("id") or "")
        y_expr = self._pick_expr(p.get("CoorY"), p.get("CoorYVal"))
        z_expr = self._pick_expr(p.get("CoorZ"), p.get("CoorZVal"))
        ref = p.get("Reference") or p.get("reference") or []
        return {"Id": pid, "Coord": [y_expr, z_expr], "Reference": list(ref)}

    # POINTS (you already had something similar)
    def _safe_points_list(self):
        pts = getattr(self, "points", None)
        if isinstance(pts, list) and pts:
            return pts
        jd = getattr(self, "json_data", None)
        if isinstance(jd, dict):
            pv = jd.get("Points") or jd.get("points")
            if isinstance(pv, list) and pv:
                return pv
        return []

    def _safe_loops_list(self):
        """
        Robustly find loops under common/legacy keys or nested containers.
        Returns a list in the 'canonical' JSON shape, i.e. a list of dicts
        with a 'Points' list, so the rest of your code can read it.
        """
        # 1) direct attribute (if some loader already mirrored it)
        loops_attr = getattr(self, "loops", None)
        if isinstance(loops_attr, list) and loops_attr:
            return loops_attr

        # 2) deep search in json_data
        jd = getattr(self, "json_data", None)
        if jd is None:
            return []

        candidates = ("Loops","Loop","Contours","Polygons","Rings","Boundaries","LoopPoints","LoopList")
        raw_loops, picked_path = _deep_find_first(jd, candidates)
        if not raw_loops:
            # helpful one-time debug
            keys = list(jd.keys()) if isinstance(jd, dict) else type(jd).__name__
            print(f"[loops] No loops found for {getattr(self,'name','?')} in json_data. Top-level keys: {keys}")
            return []

        loop_lists = _normalize_loops_value(raw_loops)
        if not loop_lists:
            print(f"[loops] Found '{picked_path}' for {getattr(self,'name','?')}, but could not normalize.")
            return []

        # Convert to canonical [{"Points":[{"Id":...}, ...]}, ...]
        out = []
        for seq in loop_lists:
            out.append({"Points": [{"Id": pid} for pid in seq]})
        print(f"[loops] {getattr(self,'name','?')} -> picked '{picked_path}', loops={len(out)}")
        return out

    def _collect_all_points(self) -> List[dict]:
        """
        Return a *single* flat list of normalized points, including:
        - top-level Points
        - all Loops[].Points
        This allows one DAG/eval pass and consistent indexing for loops_idx.
        """
        out: List[dict] = []

        # 1) Top-level Points
        for p in self._safe_points_list():
            try:
                norm = self._normalize_point_row(p)
                if norm.get("Id"):
                    out.append(norm)
            except Exception:
                pass

        # 2) Loop points (each loop has its own local points)
        for lp in self._safe_loops_list():
            for p in (lp.get("Points") or []):
                try:
                    norm = self._normalize_point_row(p)
                    if norm.get("Id"):
                        out.append(norm)
                except Exception:
                    pass

        # 3)
        print("Collected", len(out), "points in section", getattr(self, "name", "?"))
        print(" Points:", [p.get("Id") for p in out])
        print(" Loops:", len(self._safe_loops_list()), "in section", getattr(self, "name", "?"))

        # De-dup (keep first occurrence) in case a point appears in both places
        seen = set()
        uniq = []
        for p in out:
            pid = p.get("Id")
            if pid in seen:
                continue
            seen.add(pid)
            uniq.append(p)
        return uniq
    
    def _safe_points_list(self):
        """
        Resolve where points live on this CrossSection. We try a few common spots.
        Must return a list of dicts with at least {'Id': ..., 'Coord': [x_expr, y_expr]}.
        """
        pts = getattr(self, "points", None)
        if pts is None:
            # some codebases keep the raw JSON as self.json_data
            j = getattr(self, "json_data", None)
            if isinstance(j, dict):
                pts = j.get("Points")

            # # in CrossSection._safe_points_list/_safe_loops_list
            # j = getattr(self, "json_data", None)
            # if isinstance(j, dict):
            #     pts = j.get("Points")
            #     loops = j.get("Loops")

        return pts or []

    def _dag_identity_tuple(self):
        ncs = int(getattr(self, "ncs", -1))
        pts = self._collect_all_points()
        pts_tup = tuple(sorted(
            (
                str(p.get("Id")),
                str((p.get("Coord") or ["", ""])[0]),
                str((p.get("Coord") or ["", ""])[1]),
                tuple(str(r) for r in (p.get("Reference") or [])),
            )
            for p in pts
        ))
        loops = self._safe_loops_list()
        loops_tup = tuple(sorted(
            (tuple(str(pp.get("Id")) for pp in (lp.get("Points") or [])) for lp in (loops or [])),
            key=lambda ids: (ids[0] if ids else "", len(ids))
        ))
        return (ncs, pts_tup, loops_tup)

    
    def _dag_key_for_identity(self, identity) -> str:
        """
        Turn any identity object into a stable string key (even if parts are unhashable).
        """
        import hashlib, json
        try:
            # if it is already hashable & small, keep tuple; but we still stringify
            blob = json.dumps(identity, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:
            blob = repr(identity)
        return hashlib.blake2b(blob.encode("utf-8"), digest_size=16).hexdigest()


    # @lru_cache(maxsize=1024)
    # def _dag_key_for_identity(self, key: int) -> int:
    #     # thin wrapper to make lru_cache happy with identity int
    #     return key

    def build_dag(self) -> Tuple[List[str], Dict[str, dict]]:
        """
        Build topological order of points based on "Reference" dependencies.
        Caches by object identity (id(self.points)).
        """
        identity = self._dag_identity_tuple()
        key = self._dag_key_for_identity(identity)

        # instance cache slots
        cache_key = getattr(self, "_dag_cache_key", None)
        cache_val = getattr(self, "_dag_cache_val", None)
        if cache_key == key and cache_val is not None:
            return cache_val
        
        all_points = self._collect_all_points()
        by_id = {}
        deps: Dict[str, set] = {}
        for p in all_points:
            pid = str(p.get("Id") or p.get("id"))
            by_id[pid] = p
            r = p.get("Reference") or p.get("reference") or []
            if isinstance(r, (list, tuple)):
                deps[pid] = set(str(x) for x in r if x is not None)
            else:
                deps[pid] = set()

        # Kahn topological sort
        incoming = {k: set(v) for k, v in deps.items()}
        outgoing: Dict[str, set] = {k: set() for k in deps}
        for k, vs in deps.items():
            for v in vs:
                if v in outgoing:
                    outgoing[v].add(k)
                else:
                    outgoing[v] = {k}

        order: List[str] = []
        roots = [k for k, s in incoming.items() if not s]
        roots.sort()
        from collections import deque
        q = deque(roots)
        while q:
            u = q.popleft()
            order.append(u)
            for w in list(outgoing.get(u, ())):
                incoming[w].discard(u)
                if not incoming[w]:
                    q.append(w)

        # if cycles or missing deps -> append remaining in a stable order
        remaining = [k for k in deps.keys() if k not in order]
        if remaining:
            logger.warning("CrossSection DAG has unresolved dependencies or cycles: %s", remaining)
            order.extend(sorted(remaining))

        # keep for quick reuse
        self._dag_cache_key = key
        self._dag_cache_val = (order, by_id)
        return order, by_id

    # -------------------------------------------------------------------------
    # Variable array preparation & unit harmonization (vector env)
    # -------------------------------------------------------------------------

    @staticmethod
    def _results_signature(stations_m, axis_var_rows):
        import hashlib, json
        blob = json.dumps({
            "stations": [float(s) for s in (stations_m or [])],
            "vars": axis_var_rows or [],
        }, sort_keys=True, default=str)
        return hashlib.blake2b(blob.encode("utf-8"), digest_size=16).hexdigest()


    @staticmethod
    def _build_var_arrays_from_results(results: List[Dict[str, float]],
                                       defaults: Dict[str, float],
                                       keep: Optional[set] = None) -> Dict[str, np.ndarray]:
        """
        Make a name->array map (float64) for all stations.
        """
        names = keep or set()
        if not names:
            # if keep is empty, collect across all result dicts
            for d in results or []:
                names.update(d.keys())
        out: Dict[str, np.ndarray] = {}
        S = len(results or [])
        for name in names:
            arr = np.full(S, np.nan, dtype=float)
            for i, row in enumerate(results or []):
                try:
                    arr[i] = float(row.get(name, defaults.get(name, 0.0)))
                except Exception:
                    arr[i] = float(defaults.get(name, 0.0))
            out[name] = arr
        return out

    @staticmethod
    def _fix_var_units_inplace(var_arrays: Dict[str, np.ndarray], defaults: Dict[str, float]) -> None:
        """
        Heuristics identical to the engine:

        - Angles (W_/Q_/INCL_...) near 0.0 likely radians -> scale ×1000 to degrees (legacy convention).
        - Lengths (B_/T_/BEFF_/EX...) near <100 -> likely meters -> scale ×1000 to mm.
        If default is available in JSON, pick the scale that best matches the default.
        """
        def looks_angle(name: str) -> bool:
            n = name.upper()
            return n.startswith("W_") or n.startswith("Q_") or n.startswith("INCL_")

        def looks_length(name: str) -> bool:
            n = name.upper()
            return n.startswith(("B_", "T_", "BEFF_", "EX"))

        for name, arr in list(var_arrays.items()):
            a = np.asarray(arr, float)
            if a.size == 0:
                continue

            def_v = defaults.get(name)
            scale = 1.0

            if def_v is not None and np.isfinite(def_v) and def_v != 0:
                med = float(np.nanmedian(np.abs(a))) or 0.0
                candidates = (1.0, 1000.0, 0.001)
                costs = [abs(med * s - def_v) / max(1.0, abs(def_v)) for s in candidates]
                scale = candidates[int(np.argmin(costs))]
            else:
                if looks_angle(name) and np.nanmedian(np.abs(a)) < 0.5:
                    scale = 1000.0
                if looks_length(name) and 0 < np.nanmedian(np.abs(a)) < 100:
                    scale = 1000.0

            if scale != 1.0:
                var_arrays[name] = a * scale

    # -------------------------------------------------------------------------
    # Expression collection & evaluation
    # -------------------------------------------------------------------------

    @staticmethod
    def _collect_used_variable_names(section_json: dict) -> set:
        """
        Find variable names referenced by point Coord expressions.
        """
        used = set()
        pts = (section_json or {}).get("Points") or section_json.get("points") or []
        for p in pts or []:
            coord = p.get("Coord") or p.get("coord") or [0, 0]
            for expr in (coord[:2] or []):
                try:
                    txt = str(expr)
                except Exception:
                    continue
                # very simple parse: consider A..Z_0..9 tokens as potential names
                token = ""
                for ch in txt:
                    if ch.isalnum() or ch == "_":
                        token += ch
                    else:
                        if token and token not in _RESERVED_FUNC_NAMES and not token[0].isdigit():
                            used.add(token)
                        token = ""
                if token and token not in _RESERVED_FUNC_NAMES and not token[0].isdigit():
                    used.add(token)
        return used

    @staticmethod
    def _compile_pair(expr_x: str, expr_y: str):
        return _compile_expr(str(expr_x)), _compile_expr(str(expr_y))

    # -------------------------------------------------------------------------
    # Vectorized local-frame transforms
    # (XY here mean "local-Y" and "local-Z" in your convention)
    # -------------------------------------------------------------------------

    @staticmethod
    def _euclid_vectorized(X: np.ndarray, Y: np.ndarray,
                           ref_pts: List[dict] | None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Euclidean: if 0 ref -> (X,Y)
                   if 1 ref -> add (px, py)
                   if 2 ref -> add (p1.x, p2.y)  [historic behavior]
        """
        if not ref_pts:
            return X, Y
        r = [p for p in ref_pts if p is not None]
        if len(r) == 1:
            px = float(r[0].get("x", 0.0) or r[0].get("X", 0.0))
            py = float(r[0].get("y", 0.0) or r[0].get("Y", 0.0))
            return X + px, Y + py
        if len(r) >= 2:
            p1, p2 = r[0], r[1]
            px = float(p1.get("x", 0.0) or p1.get("X", 0.0))
            py = float(p2.get("y", 0.0) or p2.get("Y", 0.0))
            return X + px, Y + py
        return X, Y

    @staticmethod
    def _polar_vectorized(R: np.ndarray, A_deg: np.ndarray,
                          ref_pts: List[dict] | None) -> Tuple[np.ndarray, np.ndarray]:
        A = np.deg2rad(A_deg)
        x = R * _VECTOR_FUNCS["cos"](A)
        y = R * _VECTOR_FUNCS["sin"](A)
        return CrossSection._euclid_vectorized(x, y, ref_pts)

    @staticmethod
    def _cy_vectorized(ref_pts: List[dict] | None) -> Tuple[np.ndarray, np.ndarray]:
        # construction-al Y: origin at first reference point, X-axis along +X
        if not ref_pts:
            return np.zeros(1, float), np.zeros(1, float)
        p = ref_pts[0]
        px = float(p.get("x", 0.0) or p.get("X", 0.0))
        py = float(p.get("y", 0.0) or p.get("Y", 0.0))
        return np.asarray([px]), np.asarray([py])

    @staticmethod
    def _cz_vectorized(ref_pts: List[dict] | None) -> Tuple[np.ndarray, np.ndarray]:
        # construction-al Z: same as CY in this minimal port (can be specialized)
        return CrossSection._cy_vectorized(ref_pts)

    # -------------------------------------------------------------------------
    # Public: vectorized evaluation of section points
    # -------------------------------------------------------------------------

    def eval_expressions_vectorized(
        self,
        *,
        var_arrays: Dict[str, np.ndarray],
        order: List[str],
        by_id: Dict[str, dict],
        negate_x: bool = True,
        stations_count: Optional[int] = None,
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
      
        ids = list(order)
        # key change: S from stations_count (preferred), else from arrays (if any)
        if stations_count is not None:
            S = int(stations_count)
        else:
            S = len(next(iter(var_arrays.values()))) if var_arrays else 0

        N = len(ids)
        X = np.full((S, N), np.nan, float)
        Y = np.full((S, N), np.nan, float)

        env_base = {**_VECTOR_FUNCS}
        for k, arr in var_arrays.items():
            env_base[k] = np.asarray(arr, float)

        for j, pid in enumerate(ids):
            pj = by_id.get(pid) or {}
            coord = pj.get("Coord") or pj.get("coord") or [0, 0]
            x_expr = str(coord[0]) if len(coord) > 0 else "0"
            y_expr = str(coord[1]) if len(coord) > 1 else "0"

            cx, cy = self._compile_pair(x_expr, y_expr)

            env = env_base

            try:
                x_val = eval(cx, {"__builtins__": {}}, env)
            except NameError as e:
                logger.error("Unknown name in X for point '%s' in section '%s': %s",
                            pid, getattr(self, "name", "?"), e)
                x_val = np.full(S, np.nan, float)
            try:
                y_val = eval(cy, {"__builtins__": {}}, env)
            except Exception:
                logger.error("Unknown name in Y for point '%s' in section '%s': %s",
                            pid, getattr(self, "name", "?"), e)
                y_val = np.full(S, np.nan, float)

            X[:, j] = np.asarray(x_val, float)
            Y[:, j] = np.asarray(y_val, float)

        # apply Euclidean reference shifts in topological order
        id_to_col = {pid: idx for idx, pid in enumerate(ids)}
        for j, pid in enumerate(ids):
            refs = by_id.get(pid, {}).get("Reference") or []
            if not refs:
                continue
            if len(refs) == 1:
                k = id_to_col.get(str(refs[0]))
                if k is not None:
                    X[:, j] += X[:, k]
                    Y[:, j] += Y[:, k]
            else:
                kx = id_to_col.get(str(refs[0]))
                ky = id_to_col.get(str(refs[1]))
                if kx is not None: X[:, j] += X[:, kx]
                if ky is not None: Y[:, j] += Y[:, ky]

        if negate_x:
            X = -X
    
        # Interpret coordinates in the chosen local system (point-specific)
        # For the vectorized path we assume primary use is Euclidean coords already,
        # because reference-based mixes are uncommon across stations; users can
        # encode ref shifts in expressions. If needed, extend per-point ref frames here.

        return ids, X, Y

    def get_coordinates_vectorized(
        self,
        *,
        var_arrays: Dict[str, np.ndarray],
        order: List[str],
        by_id: Dict[str, dict],
        negate_x: bool = True,
        stations_count: Optional[int] = None,
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        return self.eval_expressions_vectorized(
            var_arrays=var_arrays, order=order, by_id=by_id,
            negate_x=negate_x, stations_count=stations_count
        )


    # -------------------------------------------------------------------------
    # Loops index (cached per ids order)
    # -------------------------------------------------------------------------

    def _loops_idx(self, ids):
        """
        Map loop point ids to column indices in your point matrix,
        tolerant to legacy point id keys.
        """
        loops_raw = self._safe_loops_list()
        if not loops_raw:
            return []

        id_to_col = {pid: j for j, pid in enumerate(ids)}

        def _pid_from_loop_point(p):
            if isinstance(p, dict):
                for k in ("Id","id","PointName","Name","PointId","PointID","Point"):
                    v = p.get(k)
                    if v is not None and str(v).strip():
                        return str(v)
            elif p is not None:
                return str(p)
            return None

        loops = []
        for lp in loops_raw:
            pts = lp.get("Points") or []
            idxs = []
            for p in pts:
                pid = _pid_from_loop_point(p)
                if pid is None:
                    continue
                j = id_to_col.get(pid)
                if j is not None:
                    idxs.append(j)
            if len(idxs) >= 2:
                loops.append(np.asarray(idxs, dtype=int))
        return loops


    # -------------------------------------------------------------------------
    # Public: compute local YZ points (vectorized)
    # -------------------------------------------------------------------------

    def compute_local_points(
        self, *, axis_var_results: List[Dict[str, float]], negate_x: bool = True
    ) -> Tuple[List[str], np.ndarray, np.ndarray, List[np.ndarray]]:
        # Build a minimal JSON "view" that includes both Points and Loops points
        pts_all = self._collect_all_points()
        section_json = {"Points": pts_all}
        used_names = self._collect_used_variable_names(section_json)
        defaults = self.get_defaults()

        var_arrays = self._build_var_arrays_from_results(axis_var_results, defaults, keep=used_names)
        self._fix_var_units_inplace(var_arrays, defaults)

        order, by_id = self.build_dag()

        ids, X_mm, Y_mm = self.get_coordinates_vectorized(
            var_arrays=var_arrays, order=order, by_id=by_id, negate_x=negate_x, stations_count=len(axis_var_results)
        )
        
        loops_idx = self._loops_idx(ids)

        return ids, X_mm, Y_mm, loops_idx


    # -------------------------------------------------------------------------
    # Public: embed local YZ as 3D using Axis (parallel-transport frames)
    # -------------------------------------------------------------------------

    def compute_embedded_points(
        self,
        *,
        axis,                          # models.axis.Axis (expects mm internal)
        axis_var_results: List[Dict[str, float]],
        stations_m: List[float],
        twist_deg: float = 0.0,
        negate_x: bool = True,
    ) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Full path: filter stations to axis domain, vector-eval local points, embed via Axis.
        Returns:
            ids, stations_mm, P_mm:(S,N,3), X_mm:(S,N), Y_mm:(S,N), loops_idx
        """
        if axis is None or not stations_m:
            return [], np.array([], float), np.zeros((0, 0, 3), float), np.zeros((0, 0), float), np.zeros((0, 0), float), []

        stations_mm_all = np.asarray(stations_m, float) * 1000.0
        smin = float(np.min(axis.stations))
        smax = float(np.max(axis.stations))
        keep = (stations_mm_all >= smin) & (stations_mm_all <= smax)
        if not np.any(keep):
            return [], np.array([], float), np.zeros((0, 0, 3), float), np.zeros((0, 0), float), np.zeros((0, 0), float), []

        stations_mm = stations_mm_all[keep]
        kept_results = [axis_var_results[i] for i, k in enumerate(keep) if k]

        ids, X_mm, Y_mm, loops_idx = self.compute_local_points(
            axis_var_results=kept_results, negate_x=negate_x
        )
        
        local_yz = np.dstack([X_mm, Y_mm])  # (S,N,2)   X==localY, Y==localZ (historic)
        P_mm = axis.embed_section_points_world(
            stations_mm, yz_points_mm=local_yz, x_offsets_mm=None, rotation_deg=float(twist_deg)
        )

        # after ids, X_mm, Y_mm are computed
        S, N = X_mm.shape if X_mm.size else (len(axis_var_results), 0)
        logger.info("Section %s (ncs=%s): stations=%d, points=%d",
                    getattr(self, "name", "?"), getattr(self, "ncs", "?"), S, N)

        # only log UL if it actually exists
        if "UL" in ids and S > 0:
            j = ids.index("UL")
            logger.info("UL @ first station (Y,Z mm): %.3f, %.3f", X_mm[0, j], Y_mm[0, j])

        # useful NaN check
        nan_xy = int(np.isnan(X_mm).sum() + np.isnan(Y_mm).sum())
        if nan_xy:
            logger.warning("NaNs in section %s: %d total", getattr(self, "name", "?"), nan_xy)

        
        return ids, stations_mm, P_mm, X_mm, Y_mm, loops_idx

    # -------------------------------------------------------------------------
    # Legacy scalar ReferenceFrame (kept for compatibility, marked deprecated)
    # -------------------------------------------------------------------------

    class ReferenceFrame:
        """Deprecated scalar path; kept for older preview tools."""
        def __init__(self, reference_type, reference=None, points=None, variables=None):
            self.reference_type = reference_type
            self.reference = reference or []
            self.points = points or []
            self.variables = variables or {}

        def eval_equation(self, string_equation):
            try:
                return float(string_equation)
            except (TypeError, ValueError):
                pass
            code = _compile_expr(string_equation)
            env = {**_SCALAR_FUNCS, **_sanitize_vars(self.variables)}
            try:
                val = eval(code, {"__builtins__": {}}, env)
                return float(val)
            except Exception as e:
                logger.debug("Scalar eval error %r: %s", string_equation, e)
                return float("nan")

        def get_coordinates(self, coords):
            rt = (self.reference_type or '').lower()
            if rt in ("c", "carthesian", "e", "euclidean"): return self._euclid(coords)
            if rt in ("p", "polar"):                         return self._polar(coords)
            if rt in ("constructionaly", "cy"):              return self._cy()
            if rt in ("constructionalz", "cz"):              return self._cz()
            return self._euclid(coords)

        def _euclid(self, coords):
            x = self.eval_equation(coords[0]); y = self.eval_equation(coords[1])
            return {'coords': {'x': x, 'y': y}, 'guides': None}

        def _polar(self, coords):
            r = self.eval_equation(coords[0])
            a = math.radians(self.eval_equation(coords[1]))
            return {'coords': {'x': r*math.cos(a), 'y': r*math.sin(a)}, 'guides': None}

        def _cy(self):
            return {'coords': {'x': 0.0, 'y': 0.0}, 'guides': None}

        def _cz(self):
            return {'coords': {'x': 0.0, 'y': 0.0}, 'guides': None}

    # Convenience: single-station scalar compute (legacy preview)
    def compute_local_points_scalar(self, env_vars: Dict[str, float]) -> List[Dict[str, float]]:
        pts = self._collect_all_points()
        out = []
        for p in pts:
            coord = p.get("Coord") or p.get("coord") or [0, 0]
            rf = CrossSection.ReferenceFrame(reference_type='euclidean', reference=p.get("Reference"), points=out, variables=env_vars)
            xy = rf.get_coordinates(coord)["coords"]
            out.append({"id": p.get("Id") or p.get("id"), "x": float(xy["x"]), "y": float(xy["y"])})
        return out
    
   

   

    @classmethod
    def from_dict(
        cls,
        row: Dict,
        *,
        mapping_cfg=None,
        axis_data=None,              # unused here but kept for signature parity
        json_loader: Optional[Callable[[str], Dict]] = None,
        default_json_by_type: Optional[Dict[str, List[str]]] = None,
        debug: bool | None = None
    ):
        """
        Create CrossSection, add sensible json_name fallback, and attach json_data.
        """
        if mapping_cfg is None:
            from .mapping import mapping as _mapping
            mapping_cfg = _mapping

        # 1) Let the generic loader map all basic fields (ncs, name, type, variables, points)
        from .base import from_dict as _generic_from_dict
        obj = _generic_from_dict(cls, row, mapping_cfg, axis_data, debug=debug)

        # 2) Ensure json_name exists: use fallback by 'type' if empty
        if not getattr(obj, "json_name", None):
            table = default_json_by_type or DEFAULT_JSON_BY_TYPE
            typ = (getattr(obj, "type", "") or "").strip()
            if typ in table:
                setattr(obj, "json_name", list(table[typ]))

        # normalize json_name to List[str]
        jn = getattr(obj, "json_name", None)
        if isinstance(jn, str) and jn.strip():
            obj.json_name = [jn.strip()]
        elif not isinstance(jn, list):
            obj.json_name = []

        # 3) Attach json_data (prefer the first file that parses and contains Points/Loops)
        if json_loader is None:
            # local minimal loader using your find_relative_file if available
            try:
                from .base import find_relative_file
            except Exception:
                find_relative_file = None

            def _default_loader(path: str) -> Dict:
                if not path:
                    return {}
                full = find_relative_file(path) if find_relative_file else path
                p = full if (full and os.path.exists(full)) else path
                try:
                    with open(p, "rb") as f:
                        return json.loads(f.read().decode("utf-8"))
                except Exception:
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            return json.load(f)
                    except Exception:
                        return {}
            json_loader = _default_loader

        obj.json_data = None
        for rel in (obj.json_name or []):
            js = json_loader(rel) or {}
            # accept typical schemas
            if isinstance(js, dict) and any(k in js for k in ("Points", "Loops", "Contours", "Polygons")):
                obj.json_data = js
                break

        # 4) If top-level 'points'/'variables' are missing but json_data exists, surface them
        if getattr(obj, "json_data", None):
            if not getattr(obj, "points", None):
                pts = obj.json_data.get("Points")
                if isinstance(pts, list):
                    obj.points = pts
            if not getattr(obj, "variables", None):
                vars_dict = obj.json_data.get("Variables")
                if isinstance(vars_dict, dict):
                    # keep dict shape; your get_defaults handles both dict and list
                    obj.variables = vars_dict

        return obj

DEFAULT_JSON_BY_TYPE = {
        "Deck":        ["MASTER_SECTION/SectionData.json"],
        "MainGirder":  ["MASTER_SECTION/MASTER_DeckMain-1Gird-Slab.json"],
        "CrossGirder": ["MASTER_SECTION/MASTER_DeckMain-1Gird-Slab.json"],
        "Pier":        ["MASTER_SECTION/MASTER_Pier.json"],
        "Foundation":  ["MASTER_SECTION/MASTER_Foundation.json"],
    }