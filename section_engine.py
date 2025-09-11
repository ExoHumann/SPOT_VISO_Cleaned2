from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import numpy as np

# Reuse your existing utilities
from Utils import (
    _RESERVED_FUNC_NAMES,
    _VECTOR_FUNCS,
    _sanitize_vars,
    _SCALAR_FUNCS,     # kept only because other parts import it; not used here
    _compile_expr,
)
from models import AxisVariable


@dataclass
class SectionGeometryEngine:
    """
    Vectorized section solver with caching.

    Responsibilities:
    - Parse section JSON once: build a point-dependency DAG (cached).
    - Evaluate point Coord expressions for all stations in one vectorized pass.
      => produces local section coordinates (X_mm, Y_mm) in the local 2D frame.
         NOTE: by convention here, X_mm == local Y, Y_mm == local Z.
    - Embed local (Y,Z) sections into 3D using Axis (parallel-transport frames).
      => Axis must provide `embed_section_points_world(...)`.

    Public API:
        compute(section_json, axis, axis_var_results, stations_m, twist_deg=0.0, negate_x=True)
            -> (ids, stations_mm, P_mm, X_mm, Y_mm, loops_idx)
    """

    # Small internal caches. Keys are object ids + rounded stations.
    _dag_cache: Dict[int, Tuple[List[str], Dict[str, dict]]] = field(default_factory=dict, repr=False)
    _loops_cache: Dict[Tuple[int, Tuple[str, ...]], List[np.ndarray]] = field(default_factory=dict, repr=False)
    _result_cache: Dict[Any, Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]] = field(default_factory=dict, repr=False)

    # ----------------------------- public API -----------------------------

    def clear_caches(self) -> None:
        """Manually clear all caches (useful during interactive debugging)."""
        self._dag_cache.clear()
        self._loops_cache.clear()
        self._result_cache.clear()

    def compute(
        self,
        *,
        section_json: dict,
        axis,                          # models.Axis instance
        axis_var_results: List[dict],  # one dict per requested station (already evaluated)
        stations_m: List[float],       # requested stations in meters
        twist_deg: float = 0.0,        # optional in-plane twist of the local section
        negate_x: bool = True,         # legacy local-X flip (kept for compatibility)
    ) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Returns:
          ids           : list[str] length N                      (point ids in topological order)
          stations_mm   : (S,)                                    (kept stations in mm)
          P_mm          : (S,N,3)                                 (global coordinates in mm)
          X_mm, Y_mm    : (S,N)                                   (local coordinates in mm; X==local Y, Y==local Z)
          loops_idx     : list[np.ndarray]                        (indices into `ids` for loop polylines)
        """
        if not section_json:
            return ([], np.array([], float), np.zeros((0, 0, 3), float),
                    np.zeros((0, 0), float), np.zeros((0, 0), float), [])

        # 1) Filter stations to the axis domain (and convert to mm)
        stations_mm_all = np.asarray(stations_m, float) * 1000.0
        smin = float(np.min(axis.stations))
        smax = float(np.max(axis.stations))
        keep_mask = (stations_mm_all >= smin) & (stations_mm_all <= smax)
        if not np.any(keep_mask):
            return ([], np.array([], float), np.zeros((0, 0, 3), float),
                    np.zeros((0, 0), float), np.zeros((0, 0), float), [])

        stations_mm = stations_mm_all[keep_mask]
        kept_results = [axis_var_results[i] for i, k in enumerate(keep_mask) if k]

        # 2) Identify variables used in Coord expressions & get defaults from JSON
        used_names = self._collect_used_variable_names(section_json)
        defaults   = self._extract_variable_defaults(section_json)

        # 3) Result cache key (section/axis/stations/twist/flip + tiny var signature)
        var_sig = self._results_signature(kept_results, used_names)
        key = (
            "sec", id(section_json),
            "ax",  id(axis),
            "st",  tuple(np.round(np.asarray(stations_m, float)[keep_mask], 6)),
            "tw",  round(float(twist_deg), 6),
            "neg", bool(negate_x),
            "varsig", var_sig,
        )
        hit = self._result_cache.get(key)
        if hit is not None:
            return hit

        # 4) Build variable arrays (vector env), then harmonize units heuristically
        var_arrays = self._build_var_arrays_from_results(kept_results, defaults, keep=used_names)
        self._fix_var_units_inplace(var_arrays, defaults)

        # 5) Topologically sorted point order + id->point mapping (cached)
        order, by_id = self._prepare_point_solver(section_json)

        # 6) Solve local XY for ALL stations (vectorized)
        ids, X_mm, Y_mm = self._get_point_coords_vectorized(var_arrays, order, by_id, negate_x=negate_x)

        # 7) Embed to global with Axis *once* (minimal twist frames inside Axis)
        # Pack local (Y,Z) in the order this engine uses:
        #   X_mm == local Y, Y_mm == local Z (historic naming kept)
        local_yz = np.dstack([X_mm, Y_mm])   # (S,N,2)
        P_mm = axis.embed_section_points_world(
            stations_mm,
            yz_points_mm=local_yz,
            x_offsets_mm=None,               # supply per-station X offsets if needed
            rotation_deg=float(twist_deg),
        )

        # 8) Loop indices (cached per section + ids order)
        loops_idx = self._loops_idx(section_json, ids)

        out = (ids, stations_mm, P_mm, X_mm, Y_mm, loops_idx)
        if len(self._result_cache) > 64:
            self._result_cache.clear()
        self._result_cache[key] = out
        return out
    
    def compute_slice_from_vars(
        self,
        *,
        section_json: dict,
        axis,                      # Axis
        axis_var_objs,             # List[AxisVariable]
        station_m: float,
        twist_deg: float = 0.0,
        negate_x: bool = True,
    ) -> dict | None:
        """
        Convenience: compute a single-station slice and return a compact dict:
          {"ids": [..], "P_row": (N,3) mm, "X_row": (N,), "Y_row": (N,),
           "station_m_used": float, "axis": axis, "loops_idx": [...]}
        """
        if not section_json or axis is None:
            return None
        s_eff = axis.clamp_station_m(float(station_m))
        var_rows = AxisVariable.evaluate_at_stations_cached(axis_var_objs or [], [s_eff])
        ids, stations_mm, P_mm, X_mm, Y_mm, loops_idx = self.compute(
            section_json=section_json,
            axis=axis,
            axis_var_results=var_rows,
            stations_m=[s_eff],
            twist_deg=float(twist_deg),
            negate_x=bool(negate_x),
        )
        P_mm = np.asarray(P_mm, float)
        X_mm = np.asarray(X_mm, float)
        Y_mm = np.asarray(Y_mm, float)
        if P_mm.ndim != 3 or P_mm.shape[0] == 0:
            return None
        return {
            "ids": ids,
            "P_row": P_mm[0],          # (N,3)
            "X_row": X_mm[0] if X_mm.size else np.array([], float),
            "Y_row": Y_mm[0] if Y_mm.size else np.array([], float),
            "station_m_used": float(s_eff),
            "axis": axis,
            "loops_idx": loops_idx,
        }

    # ----------------------------- cached helpers -----------------------------

    def _prepare_point_solver(self, section_json: dict) -> Tuple[List[str], Dict[str, dict]]:
        """
        Build the point dependency graph and return:
          - `order`: topological order of point Ids
          - `by_id`: dict(Id -> point_json)
        Cached per `id(section_json)`.
        """
        k = id(section_json)
        hit = self._dag_cache.get(k)
        if hit is not None:
            return hit

        all_points = self._collect_all_points(section_json)

        by_id = {p["Id"]: p for p in all_points}
        deps  = {pid: set(by_id[pid].get("Reference", []) or []) for pid in by_id}
        indeg = {pid: len(deps[pid]) for pid in deps}
        rev   = {pid: set() for pid in deps}
        for pid, ds in deps.items():
            for d in ds:
                if d in rev:
                    rev[d].add(pid)

        from collections import deque
        q = deque([pid for pid, d in indeg.items() if d == 0])
        order: List[str] = []
        while q:
            u = q.popleft()
            order.append(u)
            for v in rev[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        if len(order) != len(by_id):
            raise ValueError("Cyclic point references detected in section JSON.")

        if len(self._dag_cache) > 128:
            self._dag_cache.clear()
        self._dag_cache[k] = (order, by_id)
        return order, by_id

    def _loops_idx(self, section_json: dict, ids: List[str]) -> List[np.ndarray]:
        """
        Map each JSON 'Loop' to a numpy index array into `ids`.
        Cached per (id(section_json), tuple(ids)).
        """
        key = (id(section_json), tuple(ids))
        hit = self._loops_cache.get(key)
        if hit is not None:
            return hit

        loops = (section_json or {}).get("Loops", []) or []
        id_to_col = {pid: j for j, pid in enumerate(ids)}
        out: List[np.ndarray] = []
        for loop in loops:
            pts = (loop.get("Points", []) or [])
            idxs = [id_to_col.get((p or {}).get("Id")) for p in pts]
            idxs = [ix for ix in idxs if ix is not None]
            if idxs:
                out.append(np.asarray(idxs, dtype=int))

        if len(self._loops_cache) > 128:
            self._loops_cache.clear()
        self._loops_cache[key] = out
        return out

    # ----------------------------- vector solver -----------------------------

    def _get_point_coords_vectorized(
        self,
        var_arrays: Dict[str, np.ndarray],
        order: List[str],
        by_id: Dict[str, dict],
        *,
        negate_x: bool = True,
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Evaluate Coord expressions for all points and stations, vectorized.

        Returns:
          ids : the same `order` list
          X   : (S,N) local X values (== 'local Y' in the embedding convention)
          Y   : (S,N) local Y values (== 'local Z' in the embedding convention)
        """
        S = next((len(a) for a in var_arrays.values() if isinstance(a, np.ndarray)), 1)
        N = len(order)
        X = np.full((S, N), np.nan, float)
        Y = np.full((S, N), np.nan, float)

        ids = list(order)
        idx_of = {pid: j for j, pid in enumerate(ids)}

        # Safe eval env: vector functions override arrays; sanitize names
        env_arrays = _sanitize_vars(var_arrays)
        base_env = {**env_arrays, **_VECTOR_FUNCS}

        self._warned = getattr(self, "_warned", set())

        def eval_vec(expr: Any) -> np.ndarray:
            """Eval a scalar or vector expression across S stations; returns (S,) float array."""
            try:
                v = float(expr)  # numeric fast-path
                return np.full(S, v, float)
            except Exception:
                pass
            try:
                code = _compile_expr(expr)
                out = eval(code, {"__builtins__": {}}, base_env)
                out = np.asarray(out, float)
                return np.full(S, float(out), float) if out.ndim == 0 else out
            except Exception as e:
                key = (str(expr), type(e).__name__)
                if key not in self._warned:
                    print(f"[Section] eval error: {expr!r} -> {e}")
                    self._warned.add(key)
                return np.full(S, np.nan, float)

        for j, pid in enumerate(ids):
            p = by_id[pid]
            rtype = (p.get("ReferenceType") or p.get("Type", "Euclidean")).lower()
            refs  = p.get("Reference", []) or []

            xr = eval_vec(p["Coord"][0])
            yr = eval_vec(p["Coord"][1])
            if negate_x:
                xr = -xr

            # Euclidean (C/E): origin or relative to 1-2 references
            if rtype in ("c", "carthesian", "e", "euclidean"):
                if not refs:
                    X[:, j], Y[:, j] = xr, yr
                elif len(refs) == 1:
                    j0 = idx_of[refs[0]]
                    X[:, j] = xr + X[:, j0]
                    Y[:, j] = yr + Y[:, j0]
                elif len(refs) >= 2:
                    jx, jy = idx_of[refs[0]], idx_of[refs[1]]
                    X[:, j] = xr + X[:, jx]
                    Y[:, j] = yr + Y[:, jy]
                else:
                    X[:, j], Y[:, j] = xr, yr

            # Polar (P): (r, t) relative to segment (ref0 -> ref1)
            elif rtype in ("p", "polar"):
                if len(refs) < 2:
                    X[:, j], Y[:, j] = xr, yr
                else:
                    j0, j1 = idx_of[refs[0]], idx_of[refs[1]]
                    dx = X[:, j1] - X[:, j0]
                    dy = Y[:, j1] - Y[:, j0]
                    L  = np.hypot(dx, dy)
                    Ls = np.where(L == 0.0, 1.0, L)
                    ux, uy = dx / Ls, dy / Ls
                    vx, vy = -uy, ux
                    X[:, j] = X[:, j0] + xr * ux + yr * vx
                    Y[:, j] = Y[:, j0] + xr * uy + yr * vy

            # Construction A_Y (CY): vertical from line (ref0-ref1) through ref2.x
            elif rtype in ("constructionaly", "cy"):
                if len(refs) == 3:
                    j1, j2, j3 = idx_of[refs[0]], idx_of[refs[1]], idx_of[refs[2]]
                    dx = X[:, j2] - X[:, j1]
                    dy = Y[:, j2] - Y[:, j1]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        m = np.where(dx != 0.0, dy / np.where(dx == 0.0, 1.0, dx), 0.0)
                    c = Y[:, j1] - m * X[:, j1]
                    Y[:, j] = m * X[:, j3] + c
                    X[:, j] = X[:, j3]
                else:
                    X[:, j], Y[:, j] = xr, yr

            # Construction A_Z (CZ): horizontal from line (ref0-ref1) through ref3.y
            elif rtype in ("constructionalz", "cz"):
                if len(refs) == 3:
                    j1, j2, j3 = idx_of[refs[0]], idx_of[refs[1]], idx_of[refs[2]]
                    dx = X[:, j2] - X[:, j1]
                    dy = Y[:, j2] - Y[:, j1]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        m = np.where(dx != 0.0, dy / np.where(dx == 0.0, 1.0, dx), 0.0)
                    c = Y[:, j1] - m * X[:, j1]
                    X[:, j] = np.where(m != 0.0, (Y[:, j3] - c) / m, X[:, j3])
                    Y[:, j] = Y[:, j3]
                else:
                    X[:, j], Y[:, j] = xr, yr

            # Fallback: treat as Euclidean at origin
            else:
                X[:, j], Y[:, j] = xr, yr

        return ids, X, Y

    # ----------------------------- var & unit helpers -----------------------------

    @staticmethod
    def _collect_all_points(data: dict) -> List[dict]:
        """Collect all point definitions from top-level, loops, reinforcements, and zones (deduplicated by Id)."""
        pts, seen = [], set()
        for item in (data.get("Points") or []):
            if item["Id"] not in seen:
                seen.add(item["Id"]); pts.append(item)
        for loop in (data.get("Loops") or []):
            for item in (loop.get("Points") or []):
                if item["Id"] not in seen:
                    seen.add(item["Id"]); pts.append(item)
        for pr in (data.get("PointReinforcements") or []):
            item = pr["Point"]
            if item["Id"] not in seen:
                seen.add(item["Id"]); pts.append(item)
        for lr in (data.get("LineReinforcements") or []):
            for item in (lr["PointStart"], lr["PointEnd"]):
                if item["Id"] not in seen:
                    seen.add(item["Id"]); pts.append(item)
        for nez in (data.get("NonEffectiveZones") or []):
            for item in (nez["PointStart"], nez["PointEnd"]):
                if item["Id"] not in seen:
                    seen.add(item["Id"]); pts.append(item)
        return pts

    @staticmethod
    def _collect_used_variable_names(section_json: dict) -> set:
        """Find variable names used in Coord expressions (minus reserved function names)."""
        used = set()
        for p in (section_json.get("Points") or []):
            for expr in (p.get("Coord") or [])[:2]:
                s = str(expr)
                try:
                    node = ast.parse(s, mode="eval")
                    for n in ast.walk(node):
                        if isinstance(n, ast.Name):
                            used.add(n.id)
                except Exception:
                    import re
                    used.update(re.findall(r"[A-Za-z_]\w*", s))
        used -= _RESERVED_FUNC_NAMES
        return used

    @staticmethod
    def _extract_variable_defaults(section_json: dict) -> Dict[str, float]:
        """
        Read default variable values from JSON.
        Supports:
          - dict: {"H_QS": 2750, "W_ST": 75, ...}
          - list: [{"VariableName": "H_QS", "VariableValue": 2750}, ...]
        """
        defaults: Dict[str, float] = {}
        raw = section_json.get("Variables") or {}
        if isinstance(raw, dict):
            for k, v in raw.items():
                try:
                    defaults[str(k)] = float(v)
                except Exception:
                    pass
        else:
            for v in (raw or []):
                try:
                    name = str(v.get("VariableName"))
                    defaults[name] = float(v.get("VariableValue", 0) or 0.0)
                except Exception:
                    pass
        return defaults

    @staticmethod
    def _build_var_arrays_from_results(results: List[dict], defaults: Dict[str, float], keep: set | None = None) -> Dict[str, np.ndarray]:
        """Build a vector environment: one float array per variable name over S stations."""
        S = len(results)
        names = set(defaults.keys())
        for r in results:
            names.update(r.keys())
        if keep:
            names.update(keep)
        names -= _RESERVED_FUNC_NAMES

        out: Dict[str, np.ndarray] = {}
        for name in sorted(names):
            default = defaults.get(name, 0.0)
            it = (float(r.get(name, default) or 0.0) for r in results)
            out[name] = np.fromiter(it, dtype=float, count=S)
        return out

    def _fix_var_units_inplace(self, var_arrays: Dict[str, np.ndarray], defaults: Dict[str, float]) -> None:
        """
        Harmonize units (heuristics).
        - Angles: names starting with W_, Q_, INCL_ that look too small -> assume degrees * 1000.
        - Lengths: names starting with B_, T_, BEFF_, EX that look like meters -> convert to mm.
        If a default value is available in JSON, prefer the scale that best matches the default.
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
                candidates = (1.0, 1000.0, 0.001)  # identity, *1000, /1000
                costs = [abs(med*s - def_v) / max(1.0, abs(def_v)) for s in candidates]
                scale = candidates[int(np.argmin(costs))]
            else:
                if looks_angle(name) and np.nanmedian(np.abs(a)) < 0.5:
                    scale = 1000.0       # e.g., 0.075 -> 75 deg
                if looks_length(name) and 0 < np.nanmedian(np.abs(a)) < 100:
                    scale = 1000.0       # e.g., 8.575 -> 8575 mm

            if scale != 1.0:
                var_arrays[name] = a * scale

    @staticmethod
    def _results_signature(results: List[Dict[str, float]], used: set) -> Tuple:
        """
        Compact signature so cached geometry reflects meaningful var changes.
        We use up to 3 samples (first/middle/last) of each *used* variable.
        """
        if not results:
            return ()
        idxs = [0, len(results)//2, len(results)-1] if len(results) > 2 else [0, len(results)-1]
        sig = []
        for name in sorted(used or []):
            vals = []
            for i in idxs:
                v = results[i].get(name, 0.0)
                try:
                    vals.append(round(float(v), 6))
                except Exception:
                    vals.append(0.0)
            sig.append((name, tuple(vals)))
        return tuple(sig)
