# # geometry/section_engine.py
# from __future__ import annotations
# from dataclasses import dataclass, field
# from typing import Any, Dict, List, Tuple, Optional
# import numpy as np

# # Reuse your utilities/constants
# from Utils import _compile_expr, _sanitize_vars, _VECTOR_FUNCS
# # If Utils hosts _SCALAR_FUNCS too, fine; engine only needs vector funcs.

# # ---------- Small typed bundle for precomputed section ----------
# @dataclass
# class SectionPrep:
#     by_id: Dict[str, dict]            # point id -> point json
#     order: List[str]                  # topological order of point ids
#     used_vars: set                    # variable names used in Coord exprs (filtered)
#     loops_idx_by_ids: Dict[Tuple[str,...], List[np.ndarray]] = field(default_factory=dict)

# # ---------- Main engine ----------
# class SectionGeometryEngine:
#     """
#     Stateless API over stateful caches.
#     - Precompute and cache DAG/loops per section (json identity).
#     - Vectorized solve + embed, cached per (section, axis, stations, twist[, var-signature]).
#     """

#     def __init__(self, *, max_prep_cache=128, max_geom_cache=128, round_stations=3, round_twist=3):
#         self._prep_cache: Dict[int, SectionPrep] = {}
#         self._geom_cache: Dict[Tuple, Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]] = {}
#         self._max_prep = max_prep_cache
#         self._max_geom = max_geom_cache
#         self._r_sta = int(round_stations)
#         self._r_tw  = int(round_twist)

#     # -------- Public entrypoint --------
#     def compute(
#         self,
#         *,
#         section_json: dict,
#         axis,
#         axis_var_results: List[Dict[str, float]],   # one dict per kept station
#         stations_m: List[float],
#         twist_deg: float = 0.0,
#         negate_x: bool = True,
#     ):
#         """
#         Returns: (ids, stations_mm, P_mm[S,N,3], X_mm[S,N], Y_mm[S,N], loops_idx[list[np.ndarray]])
#         """
#         if not section_json:
#             return [], np.array([]), np.zeros((0,0,3)), np.zeros((0,0)), np.zeros((0,0)), []

#         prep = self._prepare(section_json)
#         ids_key = tuple(prep.order)

#         # Build station mask to axis range & keep results aligned
#         stations_m = np.asarray(stations_m, float)
#         stations_mm_all = stations_m * 1000.0
#         smin, smax = float(np.min(axis.stations)), float(np.max(axis.stations))
#         keep = (stations_mm_all >= smin) & (stations_mm_all <= smax)
#         if not np.any(keep):
#             return [], np.array([]), np.zeros((0,0,3)), np.zeros((0,0)), np.zeros((0,0)), []

#         stations_mm = stations_mm_all[keep]
#         kept_results = [axis_var_results[i] for i, k in enumerate(keep) if k]

#         # Cache key
#         key = (
#             id(section_json), id(axis),
#             tuple(np.round(stations_m[keep], self._r_sta)),
#             round(float(twist_deg), self._r_tw),
#             ids_key,
#             # Optional: a tiny signature of variables that actually matter:
#             self._results_signature(kept_results, prep.used_vars),
#         )
#         hit = self._geom_cache.get(key)
#         if hit is not None:
#             return hit

#         # Solve 2D vectorized
#         variables   = section_json.get('Variables', {}) or {}
#         var_arrays  = self._build_var_arrays(kept_results, variables, prep.used_vars)
#         self._fix_var_units_inplace(var_arrays)
#         X_mm, Y_mm  = self._solve_points_vec(section_json, prep, var_arrays, negate_x=negate_x)

#         # Embed to 3D
#         P_mm = axis.embed_points_to_global_mm(axis, stations_mm, X_mm, Y_mm, twist_deg=twist_deg)

#         # Loops index arrays (cached per ids)
#         loops_idx = self._loops_idx_for_ids(section_json, prep, ids_key)

#         out = (prep.order, stations_mm, P_mm, X_mm, Y_mm, loops_idx)
#         if len(self._geom_cache) > self._max_geom:
#             self._geom_cache.clear()
#         self._geom_cache[key] = out
#         return out

#     # -------- Prep: DAG + used vars + loops --------
#     def _prepare(self, section_json: dict) -> SectionPrep:
#         jk = id(section_json)
#         hit = self._prep_cache.get(jk)
#         if hit is not None:
#             return hit

#         points = self._collect_all_points(section_json)
#         order, by_id = self._build_point_graph(points)
#         used_vars = self._collect_used_variable_names(section_json)

#         prep = SectionPrep(by_id=by_id, order=order, used_vars=used_vars)
#         if len(self._prep_cache) > self._max_prep:
#             self._prep_cache.clear()
#         self._prep_cache[jk] = prep
#         return prep

#     # -------- Vectorized solver (your ReferenceFrame-equivalent) --------
#     def _solve_points_vec(self, section_json: dict, prep: SectionPrep, env_arrays: Dict[str, np.ndarray], *, negate_x: bool) -> Tuple[np.ndarray, np.ndarray]:
#         ids = prep.order
#         by_id = prep.by_id
#         S = next((len(a) for a in env_arrays.values() if isinstance(a, np.ndarray)), 1)
#         X = np.full((S, len(ids)), np.nan, float)
#         Y = np.full((S, len(ids)), np.nan, float)
#         idx_of = {pid: j for j, pid in enumerate(ids)}

#         for j, pid in enumerate(ids):
#             p = by_id[pid]
#             rtype = (p.get('ReferenceType') or p.get('Type', 'Euclidean')).lower()
#             refs  = p.get('Reference', []) or []
#             # eval
#             xr = self._eval_expr_vec(p['Coord'][0], env_arrays, S)
#             yr = self._eval_expr_vec(p['Coord'][1], env_arrays, S)
#             if negate_x:
#                 xr = -xr

#             if rtype in ('c', 'carthesian', 'e', 'euclidean'):
#                 if not refs:
#                     X[:, j], Y[:, j] = xr, yr
#                 elif len(refs) == 1:
#                     j0 = idx_of[refs[0]]; X[:, j] = xr + X[:, j0]; Y[:, j] = yr + Y[:, j0]
#                 elif len(refs) == 2:
#                     jx = idx_of[refs[0]]; jy = idx_of[refs[1]]
#                     X[:, j] = xr + X[:, jx]; Y[:, j] = yr + Y[:, jy]
#                 else:
#                     X[:, j], Y[:, j] = xr, yr

#             elif rtype in ('p', 'polar'):
#                 if len(refs) < 2:
#                     X[:, j], Y[:, j] = xr, yr
#                 else:
#                     j0, j1 = idx_of[refs[0]], idx_of[refs[1]]
#                     dx = X[:, j1] - X[:, j0]; dy = Y[:, j1] - Y[:, j0]
#                     L  = np.hypot(dx, dy); Ls = np.where(L == 0.0, 1.0, L)
#                     ux, uy = dx/Ls, dy/Ls
#                     vx, vy = -uy, ux
#                     X[:, j] = X[:, j0] + xr*ux + yr*vx
#                     Y[:, j] = Y[:, j0] + xr*uy + yr*vy

#             elif rtype in ('constructionaly', 'cy'):
#                 if len(refs) == 3:
#                     j1, j2, j3 = idx_of[refs[0]], idx_of[refs[1]], idx_of[refs[2]]
#                     dx = X[:, j2]-X[:, j1]; dy = Y[:, j2]-Y[:, j1]
#                     with np.errstate(divide='ignore', invalid='ignore'):
#                         m = np.where(dx != 0.0, dy/np.where(dx==0.0, 1.0, dx), 0.0)
#                     c = Y[:, j1] - m*X[:, j1]
#                     Y[:, j] = m*X[:, j3] + c
#                     X[:, j] = X[:, j3]
#                 else:
#                     X[:, j], Y[:, j] = xr, yr

#             elif rtype in ('constructionalz', 'cz'):
#                 if len(refs) == 3:
#                     j1, j2, j3 = idx_of[refs[0]], idx_of[refs[1]], idx_of[refs[2]]
#                     dx = X[:, j2]-X[:, j1]; dy = Y[:, j2]-Y[:, j1]
#                     with np.errstate(divide='ignore', invalid='ignore'):
#                         m = np.where(dx != 0.0, dy/np.where(dx==0.0, 1.0, dx), 0.0)
#                     c = Y[:, j1] - m*X[:, j1]
#                     X[:, j] = np.where(m != 0.0, (Y[:, j3]-c)/m, X[:, j3])
#                     Y[:, j] = Y[:, j3]
#                 else:
#                     X[:, j], Y[:, j] = xr, yr
#             else:
#                 X[:, j], Y[:, j] = xr, yr

#         return X, Y

#     # -------- Loops index for a given ids order --------
#     def _loops_idx_for_ids(self, section_json: dict, prep: SectionPrep, ids_key: Tuple[str, ...]) -> List[np.ndarray]:
#         hit = prep.loops_idx_by_ids.get(ids_key)
#         if hit is not None:
#             return hit
#         loops = (section_json or {}).get('Loops', []) or []
#         id_to_col = {pid: i for i, pid in enumerate(ids_key)}
#         out: List[np.ndarray] = []
#         for loop in loops:
#             idxs = [id_to_col.get((p or {}).get('Id')) for p in loop.get('Points', []) or []]
#             idxs = [ix for ix in idxs if ix is not None]
#             if idxs:
#                 out.append(np.asarray(idxs, int))
#         prep.loops_idx_by_ids[ids_key] = out
#         return out

#     # -------- Helpers (mostly your existing logic, trimmed) --------
#     @staticmethod
#     def _collect_all_points(data: dict) -> List[dict]:
#         pts, seen = [], set()
#         for item in (data.get('Points', []) or []):
#             if item['Id'] not in seen: seen.add(item['Id']); pts.append(item)
#         for loop in (data.get('Loops', []) or []):
#             for item in (loop.get('Points', []) or []):
#                 if item['Id'] not in seen: seen.add(item['Id']); pts.append(item)
#         for pr in (data.get('PointReinforcements', []) or []):
#             item = pr['Point']; 
#             if item['Id'] not in seen: seen.add(item['Id']); pts.append(item)
#         for lr in (data.get('LineReinforcements', []) or []):
#             for item in [lr['PointStart'], lr['PointEnd']]:
#                 if item['Id'] not in seen: seen.add(item['Id']); pts.append(item)
#         for nez in (data.get('NonEffectiveZones', []) or []):
#             for item in [nez['PointStart'], nez['PointEnd']]:
#                 if item['Id'] not in seen: seen.add(item['Id']); pts.append(item)
#         return pts

#     @staticmethod
#     def _build_point_graph(all_points: List[dict]) -> Tuple[List[str], Dict[str, dict]]:
#         by_id = {p['Id']: p for p in all_points}
#         deps  = {pid: set(by_id[pid].get('Reference', []) or []) for pid in by_id}
#         indeg = {pid: len(deps[pid]) for pid in deps}
#         rev   = {pid: set() for pid in deps}
#         for pid, ds in deps.items():
#             for d in ds:
#                 if d in rev: rev[d].add(pid)
#         from collections import deque
#         q = deque([pid for pid, d in indeg.items() if d == 0])
#         order = []
#         while q:
#             u = q.popleft(); order.append(u)
#             for v in rev[u]:
#                 indeg[v] -= 1
#                 if indeg[v] == 0: q.append(v)
#         if len(order) != len(by_id):
#             raise ValueError("Cyclic point references detected.")
#         return order, by_id

#     @staticmethod
#     def _collect_used_variable_names(section_json: dict) -> set:
#         import ast, re
#         used = set()
#         for p in (section_json.get('Points', []) or []):
#             for expr in (p.get('Coord', []) or [])[:2]:
#                 s = str(expr)
#                 try:
#                     node = ast.parse(s, mode='eval')
#                     for n in ast.walk(node):
#                         if isinstance(n, ast.Name):
#                             used.add(n.id)
#                 except Exception:
#                     used.update(re.findall(r'[A-Za-z_]\w*', s))
#         # remove reserved math names (done via _sanitize_vars later)
#         return used

#     @staticmethod
#     def _build_var_arrays(results: List[Dict[str, float]], variables: Dict[str, float], keep: set) -> Dict[str, np.ndarray]:
#         names = set(variables.keys()) | set().union(*(r.keys() for r in results)) | set(keep or set())
#         env = {}
#         S = len(results)
#         for name in sorted(names):
#             default = variables.get(name, 0.0)
#             env[name] = np.fromiter((float(r.get(name, default) or 0.0) for r in results), dtype=float, count=S)
#         # ensure we don’t shadow ufuncs/constants at eval:
#         return _sanitize_vars(env)

#     @staticmethod
#     def _fix_var_units_inplace(env: Dict[str, np.ndarray]) -> None:
#         for k, a in list(env.items()):
#             aa = np.asarray(a, float)
#             if k.startswith('SLOP_') and np.nanmax(np.abs(aa)) > 10:
#                 aa = aa / 1000.0
#             env[k] = aa

#     @staticmethod
#     def _eval_expr_vec(expr_text: Any, env_arrays: Dict[str, np.ndarray], S_fallback: int) -> np.ndarray:
#         # numeric fast-path
#         try:
#             v = float(expr_text)
#             return np.full(next((len(vv) for vv in env_arrays.values()), S_fallback), v, float)
#         except Exception:
#             pass
#         code = _compile_expr(str(expr_text))
#         safe_env = {**env_arrays, **_VECTOR_FUNCS}   # functions override arrays
#         try:
#             out = eval(code, {"__builtins__": {}}, safe_env)
#             out = np.asarray(out, float)
#             if out.ndim == 0:
#                 S = next((len(vv) for vv in env_arrays.values()), S_fallback)
#                 return np.full(S, float(out), float)
#             return out
#         except Exception:
#             return np.zeros(S_fallback, float)

#     @staticmethod
#     def _results_signature(results: List[Dict[str, float]], used: set) -> Tuple:
#         """
#         Compact signature so cached geometry reflects meaningful var changes.
#         We use first/last (or up to 3 samples) of each used variable.
#         """
#         if not results:
#             return ()
#         idxs = [0, len(results)//2, len(results)-1] if len(results) > 2 else [0, len(results)-1]
#         sig = []
#         for name in sorted(used):
#             vals = []
#             for i in idxs:
#                 v = results[i].get(name, 0.0)
#                 try: vals.append(round(float(v), 6))
#                 except Exception: vals.append(0.0)
#             sig.append((name, tuple(vals)))
#         return tuple(sig)


# section_engine.py
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import numpy as np

# ----------------------------- expr utils -----------------------------

_EXPR_CACHE: Dict[str, Any] = {}

def _compile_expr(s: str):
    """Small cached compiler for point Coord expressions."""
    code = _EXPR_CACHE.get(s)
    if code is not None:
        return code
    code = compile(s, "<coord-expr>", "eval")
    if len(_EXPR_CACHE) > 256:
        _EXPR_CACHE.clear()
    _EXPR_CACHE[s] = code
    return code

# numpy vector funcs + constants, both lower/upper-case & common aliases
_VECTOR_FUNCS: Dict[str, Any] = {
    # trig (radians)
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan, "atan2": np.arctan2,
    "SIN": np.sin, "COS": np.cos, "TAN": np.tan,
    "ASIN": np.arcsin, "ACOS": np.arccos, "ATAN": np.arctan, "ATAN2": np.arctan2,
    # hyperbolic / misc
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "abs": np.abs, "fabs": np.fabs, "sqrt": np.sqrt,
    "round": np.round, "floor": np.floor, "ceil": np.ceil,
    "min": np.minimum, "max": np.maximum,
    # constants
    "pi": np.pi, "PI": np.pi, "Pi": np.pi,
}

_RESERVED_FUNC_NAMES = set(_VECTOR_FUNCS.keys())

def _sanitize_vars(env: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Drop any variable that would shadow functions/constants."""
    out: Dict[str, np.ndarray] = {}
    for k, v in env.items():
        if k in _RESERVED_FUNC_NAMES:
            continue
        out[k] = v
    return out


# ----------------------------- engine -----------------------------

@dataclass
class SectionGeometryEngine:
    """
    Centralized + cached section solver:
      - Prepares point DAG once per section (cached).
      - Vector-evaluates local X/Y for all stations in one pass.
      - Builds loop indices (cached per section+ids).
      - Embeds local section into global using Axis (twist supported).
      - Cache key is sensitive to used-variable signatures -> avoids stale results.
    """
    # caches
    _dag_cache: Dict[int, Tuple[List[str], Dict[str, dict]]] = field(default_factory=dict, repr=False)
    _loops_cache: Dict[Tuple[int, Tuple[str, ...]], List[np.ndarray]] = field(default_factory=dict, repr=False)
    _result_cache: Dict[Any, Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]] = field(default_factory=dict, repr=False)

    # ----------------------------- public API -----------------------------

    def clear_caches(self) -> None:
        self._dag_cache.clear()
        self._loops_cache.clear()
        self._result_cache.clear()

    def compute(
        self,
        *,
        section_json: dict,
        axis,                          # models.Axis (must have .stations and embed_points_to_global_mm())
        axis_var_results: List[dict],  # one dict per requested station (already evaluated)
        stations_m: List[float],       # meters
        twist_deg: float = 0.0,
        negate_x: bool = True,
    ) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Returns:
          ids           : list[str] length N
          stations_mm   : (S,)
          P_mm          : (S,N,3) global mm
          X_mm, Y_mm    : (S,N) local mm
          loops_idx     : list[np.ndarray]  (indices into ids)
        """
        if not section_json:
            return ([], np.array([], float), np.zeros((0, 0, 3), float),
                    np.zeros((0, 0), float), np.zeros((0, 0), float), [])

        # 1) keep stations within axis range, convert to mm
        stations_mm_all = np.asarray(stations_m, float) * 1000.0
        smin = float(np.min(axis.stations))
        smax = float(np.max(axis.stations))
        keep_mask = (stations_mm_all >= smin) & (stations_mm_all <= smax)
        if not np.any(keep_mask):
            return ([], np.array([], float), np.zeros((0, 0, 3), float),
                    np.zeros((0, 0), float), np.zeros((0, 0), float), [])

        stations_mm = stations_mm_all[keep_mask]
        kept_results = [axis_var_results[i] for i, k in enumerate(keep_mask) if k]

        # 2) used var names (from Coord exprs) + defaults from JSON
        used_names = self._collect_used_variable_names(section_json)
        defaults   = self._extract_variable_defaults(section_json)

        # 3) cache key (section/axis/stations/twist/neg + tiny var signature)
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

        # 4) build variable arrays, harmonize units
        var_arrays = self._build_var_arrays_from_results(kept_results, defaults, keep=used_names)
        self._fix_var_units_inplace(var_arrays, defaults)

        # 5) DAG once (order & by_id)
        order, by_id = self._prepare_point_solver(section_json)

        # 6) local XY for ALL stations (vectorized)
        ids, X_mm, Y_mm = self._get_point_coords_vectorized(var_arrays, order, by_id, negate_x=negate_x)

        # 7) embed to global
        # Axis.embed_points_to_global_mm is assumed available (staticmethod or function on Axis)
        from models.axis import Axis  # safe import; adjust if your path differs
        P_mm = Axis.embed_points_to_global_mm(axis, stations_mm, X_mm, Y_mm, twist_deg=twist_deg)
        

        # 8) loop indices
        loops_idx = self._loops_idx(section_json, ids)

        out = (ids, stations_mm, P_mm, X_mm, Y_mm, loops_idx)
        if len(self._result_cache) > 64:
            self._result_cache.clear()
        self._result_cache[key] = out
        return out

    # ----------------------------- cached helpers -----------------------------

    def _prepare_point_solver(self, section_json: dict) -> Tuple[List[str], Dict[str, dict]]:
        """Topologically sort point dependency graph once per section."""
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
        """Map loop point Ids to column indices once per (section, ids)."""
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
        """Solve local XY for all stations (vectorized)."""
        S = next((len(a) for a in var_arrays.values() if isinstance(a, np.ndarray)), 1)
        N = len(order)
        X = np.full((S, N), np.nan, float)
        Y = np.full((S, N), np.nan, float)

        ids = list(order)
        idx_of = {pid: j for j, pid in enumerate(ids)}

        # build safe eval env (functions/constants override arrays)
        env_arrays = _sanitize_vars(var_arrays)
        base_env = {**env_arrays, **_VECTOR_FUNCS}

        def eval_vec(expr: Any) -> np.ndarray:
            # numeric fast-path
            try:
                v = float(expr)
                return np.full(S, v, float)
            except Exception:
                pass
            try:
                code = _compile_expr(str(expr))
                out = eval(code, {"__builtins__": {}}, base_env)
                out = np.asarray(out, float)
                if out.ndim == 0:
                    return np.full(S, float(out), float)
                return out
            except Exception:
                # on bad expr (e.g. "Error 2015") return NaNs so it won't pollute bbox
                return np.full(S, np.nan, float)

        for j, pid in enumerate(ids):
            p = by_id[pid]
            rtype = (p.get("ReferenceType") or p.get("Type", "Euclidean")).lower()
            refs  = p.get("Reference", []) or []

            xr = eval_vec(p["Coord"][0])
            yr = eval_vec(p["Coord"][1])
            if negate_x:
                xr = -xr

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

            elif rtype in ("p", "polar"):
                if len(refs) < 2:
                    X[:, j], Y[:, j] = xr, yr
                else:
                    j0, j1 = idx_of[refs[0]], idx_of[refs[1]]
                    dx = X[:, j1] - X[:, j0]
                    dy = Y[:, j1] - Y[:, j0]
                    L  = np.hypot(dx, dy)
                    Ls = np.where(L == 0.0, 1.0, L)
                    ux, uy = dx/Ls, dy/Ls
                    vx, vy = -uy, ux
                    X[:, j] = X[:, j0] + xr*ux + yr*vx
                    Y[:, j] = Y[:, j0] + xr*uy + yr*vy

            elif rtype in ("constructionaly", "cy"):
                if len(refs) == 3:
                    j1, j2, j3 = idx_of[refs[0]], idx_of[refs[1]], idx_of[refs[2]]
                    dx = X[:, j2] - X[:, j1]
                    dy = Y[:, j2] - Y[:, j1]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        m = np.where(dx != 0.0, dy/np.where(dx == 0.0, 1.0, dx), 0.0)
                    c = Y[:, j1] - m*X[:, j1]
                    Y[:, j] = m*X[:, j3] + c
                    X[:, j] = X[:, j3]
                else:
                    X[:, j], Y[:, j] = xr, yr

            elif rtype in ("constructionalz", "cz"):
                if len(refs) == 3:
                    j1, j2, j3 = idx_of[refs[0]], idx_of[refs[1]], idx_of[refs[2]]
                    dx = X[:, j2] - X[:, j1]
                    dy = Y[:, j2] - Y[:, j1]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        m = np.where(dx != 0.0, dy/np.where(dx == 0.0, 1.0, dx), 0.0)
                    c = Y[:, j1] - m*X[:, j1]
                    X[:, j] = np.where(m != 0.0, (Y[:, j3] - c)/m, X[:, j3])
                    Y[:, j] = Y[:, j3]
                else:
                    X[:, j], Y[:, j] = xr, yr

            else:
                X[:, j], Y[:, j] = xr, yr

        return ids, X, Y

    # ----------------------------- var & unit helpers -----------------------------

    @staticmethod
    def _collect_all_points(data: dict) -> List[dict]:
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
        Supports both:
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
        Harmonize units for variables coming from mixed sources (mm vs m, deg-like as 1/1000, slopes in mm/m).
        Uses JSON defaults to infer the most plausible scale among {1, ×1000, ÷1000}.
        """
        def looks_angular(name: str) -> bool:
            n = name.upper()
            return n.startswith("W_") or n.startswith("Q_") or n.startswith("INCL_")

        for name, arr in list(var_arrays.items()):
            a = np.asarray(arr, float)
            if a.size == 0:
                continue

            # If we have a JSON default, pick scale that best matches the default magnitude.
            def_v = defaults.get(name)
            if def_v is not None and np.isfinite(def_v):
                med = float(np.nanmedian(np.abs(a))) or 0.0
                scales = (1.0, 1000.0, 0.001)
                errs = [abs(med*s - def_v) / max(1.0, abs(def_v)) for s in scales]
                best = scales[int(np.argmin(errs))]
                if best != 1.0:
                    a = a * best

            # SLOP_* often arrive in mm/m -> convert to m/m if suspiciously large.
            if name.upper().startswith("SLOP_") and np.nanmax(np.abs(a)) > 10.0:
                a = a / 1000.0

            # Angular guardrail: tiny vs clearly-angular default (>5 deg) -> likely milli-deg, scale up.
            if looks_angular(name):
                if def_v is not None and def_v > 5.0 and np.nanmedian(np.abs(a)) < 0.5:
                    a = a * 1000.0

            var_arrays[name] = a

    @staticmethod
    def _results_signature(results: List[Dict[str, float]], used: set) -> Tuple:
        """
        Compact signature so cached geometry reflects meaningful var changes.
        Use up to 3 samples (first/middle/last) of each used variable.
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
