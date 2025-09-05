## Step by step: 
## 1. Once you have venv activared: in cdm -> C:\venv\Scripts\activate.bat
## 2. get to folder location where main.py is located -> cd C:\RCZ\23_E45\09_NEW_SPOT\FOR_JUHA\SPOT_VISO
## 3. run in cmd code below:

# pyinstaller --onedir ^
#   --add-data "Axis.py;." ^
#   --add-data "AxisVariables.py;." ^
#   --add-data "AddClasses.py;." ^
#   --add-data "resources/plotly-latest.min.js;resources" ^
#   --hidden-import=numpy ^
#   --hidden-import=scipy ^
#   --hidden-import=scipy.interpolate ^
#   --hidden-import=plotly ^
#   --hidden-import=plotly.graph_objects ^
#   --hidden-import=jinja2 ^
#   --upx-dir="C:\upx" ^
#   main_dash.py

import math
import json
import re
import os
import sys
import webbrowser
import ast
from functools import lru_cache

import numpy as np
import plotly.graph_objects as go
from jinja2 import Template

from AddClasses import *           # CrossSection, DeckObject, FoundationObject, PierObject, mapping, find_git_folder, load_from_json
from AxisVariables import AxisVariable
from Axis import Axis

verbose = False
import sys; print(sys.executable)
try:
    import orjson as _fastjson
except Exception as e:
    print(f"[orjson] Unavailable: {e}")   # shows the actual reason
    _fastjson = None

# ---------- Axis instance cache ----------
_AXIS_OBJ_CACHE = {}

def get_axis_cached_by_tuple(stations, x, y, z):
    import numpy as _np
    key = (
        tuple(_np.asarray(stations, float)),
        tuple(_np.asarray(x, float)),
        tuple(_np.asarray(y, float)),
        tuple(_np.asarray(z, float)),
    )
    ax = _AXIS_OBJ_CACHE.get(key)
    if ax is None:
        from Axis import Axis
        ax = Axis(stations, x, y, z, units='mm')
        if len(_AXIS_OBJ_CACHE) > 64:
            _AXIS_OBJ_CACHE.clear()
        _AXIS_OBJ_CACHE[key] = ax
    return ax


# scalar (math) and vector (numpy) function maps
_SCALAR_FUNCS = {
    'COS': math.cos, 'SIN': math.sin, 'TAN': math.tan,
    'ACOS': math.acos, 'ASIN': math.asin, 'ATAN': math.atan,
    'cos': math.cos, 'sin': math.sin, 'tan': math.tan,
    'acos': math.acos, 'asin': math.asin, 'atan': math.atan,
    'LOG': math.log, 'EXP': math.exp, 'SQRT': math.sqrt, 'ABS': abs,
    'log': math.log, 'exp': math.exp, 'sqrt': math.sqrt, 'abs': abs,
    'PI': math.pi, 'Pi': math.pi, 'pi': math.pi,
}
_VECTOR_FUNCS = {
    'COS': np.cos, 'SIN': np.sin, 'TAN': np.tan,
    'ACOS': np.arccos, 'ASIN': np.arcsin, 'ATAN': np.arctan,
    'cos': np.cos, 'sin': np.sin, 'tan': np.tan,
    'acos': np.arccos, 'asin': np.arcsin, 'atan': np.arctan,
    'LOG': np.log, 'EXP': np.exp, 'SQRT': np.sqrt, 'ABS': np.abs,
    'log': np.log, 'exp': np.exp, 'sqrt': np.sqrt, 'abs': np.abs,
    'PI': math.pi, 'Pi': math.pi, 'pi': math.pi,
}

# prevent variable names from shadowing function names
_RESERVED_FUNC_NAMES = set(_SCALAR_FUNCS.keys()) | set(_VECTOR_FUNCS.keys())

_ALLOWED_AST = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant, ast.Name, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv, ast.USub, ast.UAdd, ast.Call
)

@lru_cache(maxsize=4096)
def _compile_expr(expr_text: str):
    node = ast.parse(str(expr_text), mode='eval')
    for n in ast.walk(node):
        if not isinstance(n, _ALLOWED_AST):
            raise ValueError(f"Disallowed expression: {expr_text}")
        if isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name) or n.func.id not in _SCALAR_FUNCS:
                raise ValueError(f"Disallowed function: {getattr(n.func, 'id', '?')}")
    return compile(node, "<expr>", "eval")

def _sanitize_vars(variables: dict) -> dict:
    # drop keys that would shadow functions/constants
    return {k: v for k, v in (variables or {}).items() if k not in _RESERVED_FUNC_NAMES}


# =========================
# Vectorized geometry core
# =========================

# ---------- Vectorized axis frames + 180° rotation ----------
# _FRAMES_CACHE = {}

# ---------- Point graph compiler + per-station solver ----------
_solver_cache = {}

def solver_for(json_path_or_dict):
    key = id(json_path_or_dict) if isinstance(json_path_or_dict, dict) else json_path_or_dict
    hit = _solver_cache.get(key)
    if hit is None:
        data = json_path_or_dict if isinstance(json_path_or_dict, dict) else json.load(open(json_path_or_dict, 'r', encoding='utf-8'))
        hit = PointGraphSolver(data)
        if len(_solver_cache) > 16:
            _solver_cache.clear()
        _solver_cache[key] = hit
    return hit

class PointGraphSolver:
    def __init__(self, json_data: dict):
        self.json = json_data
        self.all_points = self._collect_all_points(json_data)
        self.order, self.by_id = self._build_graph(self.all_points)

    @staticmethod
    def _collect_all_points(data):
        pts, seen = [], set()
        for item in data.get('Points', []) or []:
            if item['Id'] not in seen: seen.add(item['Id']); pts.append(item)
        for loop in data.get('Loops', []) or []:
            for item in loop.get('Points', []) or []:
                if item['Id'] not in seen: seen.add(item['Id']); pts.append(item)
        for pr in data.get('PointReinforcements', []) or []:
            item = pr['Point']
            if item['Id'] not in seen: seen.add(item['Id']); pts.append(item)
        for lr in data.get('LineReinforcements', []) or []:
            for item in [lr['PointStart'], lr['PointEnd']]:
                if item['Id'] not in seen: seen.add(item['Id']); pts.append(item)
        for nez in data.get('NonEffectiveZones', []) or []:
            for item in [nez['PointStart'], nez['PointEnd']]:
                if item['Id'] not in seen: seen.add(item['Id']); pts.append(item)
        return pts

    @staticmethod
    def _build_graph(all_points):
        by_id = {p['Id']: p for p in all_points}
        deps = {pid: set(by_id[pid].get('Reference', [])) for pid in by_id}
        indeg = {pid: len(deps[pid]) for pid in deps}
        rev = {pid: set() for pid in deps}
        for pid, ds in deps.items():
            for d in ds:
                if d in rev: rev[d].add(pid)
        from collections import deque
        q = deque([pid for pid, d in indeg.items() if d == 0])
        order = []
        while q:
            u = q.popleft(); order.append(u)
            for v in rev[u]:
                indeg[v] -= 1
                if indeg[v] == 0: q.append(v)
        if len(order) != len(by_id):
            raise ValueError("Cyclic point references detected.")
        return order, by_id

    def solve_one(self, variables: dict):
        """Scalar (per-station) solve with full ReferenceFrame semantics."""
        solved = {}
        out = []
        for pid in self.order:
            ref_point = self.by_id[pid]
            frame = CrossSection.ReferenceFrame(
                ref_point.get('ReferenceType') or ref_point.get('Type', 'Euclidean'),
                ref_point.get('Reference', []),
                [{'id': k, **v} for k, v in solved.items()],
                variables
            )
            lx = frame.eval_equation(ref_point['Coord'][0])
            ly = frame.eval_equation(ref_point['Coord'][1])

            if ref_point.get('Reference'):
                data_xy = frame.get_coordinates(ref_point['Coord'])
                px = data_xy['coords']['x']; py = data_xy['coords']['y']
                guides = data_xy['guides']
            else:
                px, py = lx, ly
                m = max(abs(lx), abs(ly))*1.5
                guides = {'isPlane': True, 'origin': {'x':0,'y':0}, 'dirX': {'x':m,'y':0}, 'dirY': {'x':0,'y':m}}

            solved[pid] = {'x': px, 'y': py}
            out.append({'id': pid, 'x': px, 'y': py, 'localX': lx, 'localY': ly, 'ref': ref_point, 'guides': guides})
        return out

_LOOPS_INDEX_CACHE = {}

def loops_idx_cached(json_data, ids):
    key = (id(json_data), tuple(ids))
    hit = _LOOPS_INDEX_CACHE.get(key)
    if hit is not None:
        return hit
    loops = (json_data or {}).get('Loops', []) or []
    id_to_col = {pid: j for j, pid in enumerate(ids)}
    out = []
    for loop in loops:
        idxs = [id_to_col.get(p.get('Id')) for p in loop.get('Points', []) or []]
        idxs = [ix for ix in idxs if ix is not None]
        if idxs:
            out.append(np.asarray(idxs, dtype=int))
    if len(_LOOPS_INDEX_CACHE) > 64:
        _LOOPS_INDEX_CACHE.clear()
    _LOOPS_INDEX_CACHE[key] = out
    return out


# =========================
# ReferenceFrame (scalar)
# =========================
#TODO Crosssection class move

# =========================
# Utility (scalar + vector)
# =========================
# --------- Point/DAG preparation cache (drop-in) ----------
_POINT_DAG_CACHE = {}

def prepare_point_solver_cached(json_data, json_file_path=None):
    """
    Cached wrapper for Utility.prepare_point_solver(json_data).
    Keyed by the identity of json_data or the json file path if provided.
    """
    # Prefer path key when available (stable across runs)
    key = f"PPS:{json_file_path}" if json_file_path else f"PPS_ID:{id(json_data)}"
    pre = _POINT_DAG_CACHE.get(key)
    if pre is not None:
        return pre
    pre = Utility.prepare_point_solver(json_data)  # your existing function
    # keep cache small
    if len(_POINT_DAG_CACHE) > 64:
        _POINT_DAG_CACHE.clear()
    _POINT_DAG_CACHE[key] = pre
    return pre

class Utility:
    # ----- Points DAG -----
    @staticmethod
    def get_all_points(data):
        points = []
        seen = set()

        for item in data.get('Points', []):
            if item['Id'] not in seen:
                seen.add(item['Id']); points.append(item)

        for loop in data.get('Loops', []):
            for item in loop.get('Points', []):
                if item['Id'] not in seen:
                    seen.add(item['Id']); points.append(item)

        for pr in data.get('PointReinforcements', []):
            item = pr['Point']
            if item['Id'] not in seen:
                seen.add(item['Id']); points.append(item)

        for lr in data.get('LineReinforcements', []):
            for item in [lr['PointStart'], lr['PointEnd']]:
                if item['Id'] not in seen:
                    seen.add(item['Id']); points.append(item)

        for nez in data.get('NonEffectiveZones', []):
            for item in [nez['PointStart'], nez['PointEnd']]:
                if item['Id'] not in seen:
                    seen.add(item['Id']); points.append(item)
        return points

    @staticmethod
    def _build_point_graph(all_points):
        by_id = {p['Id']: p for p in all_points}
        deps = {pid: set(by_id[pid].get('Reference', [])) for pid in by_id}
        indeg = {pid: len(deps[pid]) for pid in deps}
        rev = {pid: set() for pid in deps}
        for pid, ds in deps.items():
            for d in ds:
                if d in rev:
                    rev[d].add(pid)
        order = []
        from collections import deque
        q = deque([pid for pid, d in indeg.items() if d == 0])
        while q:
            u = q.popleft()
            order.append(u)
            for v in rev[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        if len(order) != len(by_id):
            raise ValueError("Cyclic point references detected.")
        return order, by_id

    @staticmethod
    def prepare_point_solver(data):
        all_points = Utility.get_all_points(data)
        return Utility._build_point_graph(all_points)

    # ----- Scalar per-station solver (kept for compatibility) -----
    @staticmethod
    def get_point_coords(data, variables, precomputed=None):
        if precomputed is None:
            all_points = Utility.get_all_points(data)
            order, by_id = Utility._build_point_graph(all_points)
        else:
            order, by_id = precomputed

        points = []
        solved = {}
        for pid in order:
            ref_point = by_id[pid]
            frame = CrossSection.ReferenceFrame(
                ref_point.get('ReferenceType') or ref_point.get('Type', 'Euclidean'),
                ref_point.get('Reference', []),
                [{'id': k, **v} for k, v in solved.items()],
                variables
            )
            x = frame.eval_equation(ref_point['Coord'][0])
            y = frame.eval_equation(ref_point['Coord'][1])

            if ref_point.get('Reference'):
                data_xy = frame.get_coordinates(ref_point['Coord'])
                px = data_xy['coords']['x']; py = data_xy['coords']['y']
                guides = data_xy['guides']
            else:
                px, py = x, y
                guides = {
                    'isPlane': True, 'origin': {'x': 0, 'y': 0},
                    'dirX': {'x': max(abs(x), abs(y))*1.5, 'y': 0},
                    'dirY': {'x': 0, 'y': max(abs(x), abs(y))*1.5}
                }

            solved[pid] = {'x': px, 'y': py}
            points.append({
                'id': pid, 'x': px, 'y': py,
                'localX': px, 'localY': py,
                'ref': ref_point,
                'guides': guides
            })
        return points

    # ----- Vectorized evaluation over all stations -----
    @staticmethod
    def build_var_arrays_from_results(results, variables, keep=None):
        """
        Faster: minimize Python overhead, avoid per-element try/except,
        and build arrays with numpy.fromiter.
        """
        import numpy as np
        S = len(results)
        # collect names once
        names = set(variables.keys())
        for r in results:
            names.update(r.keys())
        if keep:
            names.update(keep)
        names -= _RESERVED_FUNC_NAMES  # don't shadow math funcs

        out = {}
        for name in sorted(names):
            default = variables.get(name, 0.0)
            # avoid exceptions in the hot loop
            it = (float(r.get(name, default) or 0.0) for r in results)
            out[name] = np.fromiter(it, dtype=float, count=S)
        return out


    @staticmethod
    def eval_equation_vec(expr_text, env_arrays, S_fallback):
        """
        Vectorized evaluator. Returns np.ndarray shape (S,).
        - expr_text: string or number
        - env_arrays: dict name -> np.ndarray(S,)
        - S_fallback: used when expr is a plain number and we can't infer S
        Uses NumPy ufuncs; ensures functions cannot be shadowed by variables.
        """
        import numpy as np
        # numeric fast path
        try:
            val = float(expr_text)
            S = next((len(v) for v in env_arrays.values() if isinstance(v, np.ndarray)), None)
            if S is None:
                S = S_fallback
            return np.full(S, val, dtype=float)
        except Exception:
            pass

        code = _compile_expr(str(expr_text))
        # Functions must override any same-named arrays -> put funcs last
        safe_env = {**_sanitize_vars(env_arrays), **_VECTOR_FUNCS}
        try:
            out = eval(code, {"__builtins__": {}}, safe_env)
            out = np.asarray(out, dtype=float)
            if out.ndim == 0:
                S = next((len(v) for v in env_arrays.values() if isinstance(v, np.ndarray)), S_fallback)
                return np.full(S, float(out), dtype=float)
            return out
        except Exception as e:
            print(f"Vector eval error for '{expr_text}': {e}")
            return np.zeros(S_fallback, dtype=float)
        
    @staticmethod
    def get_point_coords_vectorized(data, var_arrays, precomputed=None):
        """
        Solve 2D section points for ALL stations in one pass (vectorized).
        Handles Euclidean / POLAR / CY / CZ exactly like the scalar ReferenceFrame.
        Returns:
           ids (list[str]),
           X (S,N) in mm,
           Y (S,N) in mm
        """
        import numpy as np

        # Build DAG once
        if precomputed is None:
            all_points = Utility.get_all_points(data)
            order, by_id = Utility._build_point_graph(all_points)
        else:
            order, by_id = precomputed

        # ONE vector env (NumPy funcs + arrays)
        # env = {**_ALLOWED_FUNCS, **var_arrays}
        env = dict(var_arrays)
        
        # S (stations count)
        S = next((len(a) for a in var_arrays.values() if isinstance(a, np.ndarray)), 1)

        N = len(order)
        X = np.full((S, N), np.nan, dtype=float)
        Y = np.full((S, N), np.nan, dtype=float)
        ids = list(order)
        idx_of = {pid: j for j, pid in enumerate(ids)}

        for j, pid in enumerate(ids):
            p = by_id[pid]
            rtype = (p.get('ReferenceType') or p.get('Type', 'Euclidean')).lower()
            refs  = p.get('Reference', [])
            # evaluate JSON expressions station-wise
            x_raw = -Utility.eval_equation_vec(p['Coord'][0], env, S)  # NEGATED
            y_raw =  Utility.eval_equation_vec(p['Coord'][1], env, S)   

            if rtype in ('c','carthesian','e','euclidean'):
                if not refs:
                    X[:, j] = x_raw; Y[:, j] = y_raw
                elif len(refs) == 1:
                    j0 = idx_of[refs[0]]
                    X[:, j] = x_raw + X[:, j0]
                    Y[:, j] = y_raw + Y[:, j0]
                elif len(refs) == 2:
                    jx = idx_of[refs[0]]; jy = idx_of[refs[1]]
                    X[:, j] = x_raw + X[:, jx]
                    Y[:, j] = y_raw + Y[:, jy]
                else:
                    X[:, j] = x_raw; Y[:, j] = y_raw

            elif rtype in ('p','polar'):
                if len(refs) < 2:
                    X[:, j] = x_raw; Y[:, j] = y_raw
                else:
                    j0, j1 = idx_of[refs[0]], idx_of[refs[1]]
                    dx = X[:, j1] - X[:, j0]
                    dy = Y[:, j1] - Y[:, j0]
                    L  = np.hypot(dx, dy)
                    Ls = np.where(L == 0.0, 1.0, L)
                    ux, uy = dx / Ls, dy / Ls
                    vx, vy = -uy, ux
                    X[:, j] = X[:, j0] + x_raw*ux + y_raw*vx
                    Y[:, j] = Y[:, j0] + x_raw*uy + y_raw*vy

            elif rtype in ('constructionaly','cy'):
                if len(refs) != 3:
                    X[:, j] = x_raw; Y[:, j] = y_raw
                else:
                    j1, j2, j3 = idx_of[refs[0]], idx_of[refs[1]], idx_of[refs[2]]
                    dx = X[:, j2] - X[:, j1]
                    dy = Y[:, j2] - Y[:, j1]
                    # y = m*x + c through p1,p2 at x = X[:, j3]
                    with np.errstate(divide='ignore', invalid='ignore'):
                        m = np.where(dx != 0.0, dy/np.where(dx==0.0, 1.0, dx), 0.0)
                    c = Y[:, j1] - m*X[:, j1]
                    Y[:, j] = m*X[:, j3] + c
                    X[:, j] = X[:, j3]

            elif rtype in ('constructionalz','cz'):
                if len(refs) != 3:
                    X[:, j] = x_raw; Y[:, j] = y_raw
                else:
                    j1, j2, j3 = idx_of[refs[0]], idx_of[refs[1]], idx_of[refs[2]]
                    dx = X[:, j2] - X[:, j1]
                    dy = Y[:, j2] - Y[:, j1]
                    with np.errstate(divide='ignore', invalid='ignore'):
                        m = np.where(dx != 0.0, dy/np.where(dx==0.0, 1.0, dx), 0.0)
                    c = Y[:, j1] - m*X[:, j1]
                    # x = (y - c)/m at y = Y[:, j3]
                    X[:, j] = np.where(m != 0.0, (Y[:, j3] - c)/m, X[:, j3])
                    Y[:, j] = Y[:, j3]

            else:
                X[:, j] = x_raw; Y[:, j] = y_raw

        return ids, X, Y


    # ----- Unit & misc -----
    @staticmethod
    def collect_used_variable_names(section_json):
        """
        Collect variable names used anywhere in Points[*].Coord, excluding
        reserved math/ufunc names to avoid shadowing (COS/SIN/TAN/PI/etc.).
        """
        import ast, re
        used = set()
        for p in section_json.get('Points', []):
            for expr in p.get('Coord', [])[:2]:
                s = str(expr)
                try:
                    node = ast.parse(s, mode='eval')
                    for n in ast.walk(node):
                        if isinstance(n, ast.Name):
                            used.add(n.id)
                except Exception:
                    for tok in re.findall(r'[A-Za-z_]\w*', s):
                        used.add(tok)
        # Drop reserved function/constant names (case-sensitive, matches your map)
        used -= _RESERVED_FUNC_NAMES
        return used


    @staticmethod
    def switch_unit_to_mm(unit):
        return {'m':1000,'cm':10,'mm':1,'ft':304.8,'in':25.4}.get(unit, 1)

    @staticmethod
    def extract_variables(equation, variables):
        variable_keys = list(variables.keys())
        pattern = r'\b(' + '|'.join(map(re.escape, variable_keys)) + r')\b'
        matches = re.findall(pattern, str(equation))
        return matches or []

    @staticmethod
    def save_as_website(fig, file_path):
        """
        Save the Plotly figure to HTML and open it. The right panel updates on hover
        (magnetic behavior) and click.
        """
        if getattr(sys, 'frozen', False):
            base_path = os.path.dirname(sys.executable)
        else:
            base_path = os.path.dirname(__file__)
        html_file = os.path.join(base_path, os.path.splitext(os.path.basename(file_path))[0] + '_3d.html')

        template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>3D Plot</title>
<script src="resources/plotly-latest.min.js"></script>
<script src="_internal/resources/plotly-latest.min.js"></script>
<style>
  html,body{margin:0;height:100%}
  body{display:flex;flex-direction:row;font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,sans-serif}
  #plot-container{flex:2;min-width:320px;height:100vh;background:#f5f6f7}
  #plot{width:100%;height:100%}
  #json-panel{flex:1.4;min-width:240px;max-width:560px;padding:14px;background:#fff;border-left:1px solid #ddd;overflow-y:auto}
  h3{margin:0 0 10px 0;font-size:15px;font-weight:600}
  pre{margin:0;white-space:pre-wrap;word-break:break-word;font-size:13px;line-height:1.35}
  .hint{color:#666;font-size:12px;margin-top:6px}
  @media (max-width: 900px){
    body{flex-direction:column}
    #plot-container{height:60vh}
    #json-panel{height:40vh;border-left:none;border-top:1px solid #ddd}
  }
</style>
</head>
<body>
  <div id="plot-container"><div id="plot"></div></div>
  <div id="json-panel">
    <h3>Hover to view metadata</h3>
    <pre id="json-output">No item selected.</pre>
    <div class="hint">Click also sets the panel to that item.</div>
  </div>

<script>
function isObject(x){ return x && typeof x === 'object' && !Array.isArray(x); }
function pretty(x){ try { return JSON.stringify(x, null, 2); } catch(e){ return String(x); } }

function toPanelPayload(pt){
  if (!pt) return null;
  if (pt.customdata !== undefined) return pt.customdata;
  return { trace: (pt.fullData && pt.fullData.name) || '(unknown)', x: pt.x, y: pt.y, z: pt.z };
}

function bootPlot(){
  var plotDiv = document.getElementById('plot');
  var figure  = {{ fig | tojson }};
  Plotly.newPlot(plotDiv, figure.data, figure.layout, {responsive:true}).then(function(){
    var panel = document.getElementById('json-output');
    var last = null;

    function updateFrom(evt){
      if (!evt || !evt.points || !evt.points.length) return;
      var payload = toPanelPayload(evt.points[0]);
      if (payload === null || payload === undefined) return;
      last = payload;
      panel.textContent = isObject(payload) ? pretty(payload) : String(payload);
    }
    plotDiv.on('plotly_hover',  updateFrom);
    plotDiv.on('plotly_click',  updateFrom);
    plotDiv.on('plotly_unhover', function(){
      // keep last selection; clear if you prefer:
      // panel.textContent = "No item selected.";
    });
  }).catch(function(err){
    console.error(err);
    document.getElementById('json-output').textContent = "Plotly error. See console.";
  });
}

// Try local -> internal -> CDN
(function ensurePlotly(){
  if (window.Plotly) return bootPlot();

  function load(src, onload, onerror){
    var s = document.createElement('script');
    s.src = src; s.async = false;
    s.onload = onload; s.onerror = onerror;
    document.head.appendChild(s);
  }

  load('resources/plotly-latest.min.js', bootPlot, function(){
    load('_internal/resources/plotly-latest.min.js', bootPlot, function(){
      load('https://cdn.plot.ly/plotly-2.27.0.min.js', bootPlot, function(){
        document.getElementById('json-output').textContent =
          "Could not load Plotly (local or CDN).";
      });
    });
  });
})();
</script>

</body>
</html>
        """

        try:
            template = Template(template_str)
            fig_dict = fig.to_dict()
            final_html = template.render(fig=fig_dict)
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(final_html)
            print(f"Saved 3D plot with metadata panel to {html_file}")
            url = 'file:\\' + os.path.abspath(html_file)
            print(f"Opening {url}")
            webbrowser.open(url)             # <— make sure this line is complete
        except Exception as e:
            print(f"Error saving {html_file}: {e}")


# =========================
# Plot trace builder
# =========================
def _fix_var_units_inplace(var_arrays: dict):
    """
    Normalize troublesome variables to the units the point solver expects:
      • SLOP_* must be dimensionless (m/m). If we see >10, it's almost certainly mm/m → divide by 1000.
      • INCL_* are angles in degrees; don't scale.
      • Q_NG, W_ST are gradients/ratios; don't scale.
    """
    import numpy as np

    for name, arr in list(var_arrays.items()):
        a = np.asarray(arr, dtype=float)

        if name.startswith('SLOP_'):
            # if someone scaled slope to mm/m, bring it back to m/m
            if np.nanmax(np.abs(a)) > 10:   # heuristically detect the mm/m case
                a = a / 1000.0

        # angles/ratios: keep as-is
        elif name.startswith('INCL_'):
            pass
        elif name in ('Q_NG', 'W_ST'):
            pass

        var_arrays[name] = a

import plotly.graph_objects as go

# ---------- Build S×N×3 matrices of global section points ----------
def build_point_matrices(axis, json_data, results, stations_m, twist_deg=0.0):
    """
    Vectorized path:
      - Apply the SAME keep-mask to stations and results (so S matches everywhere).
      - Build var arrays once (avoid function/constant shadowing).
      - Solve 2D section points for ALL kept stations at once.
      - Embed to global 3D and apply the 180° flip (inside embed_points_to_global_mm).
      - Build loop index lists.

    Returns:
      ids:        list of N point ids (union of loop order + all stations)
      stations_mm: (S,) stations in mm, filtered to axis range
      P_mm:       (S,N,3) global coords in mm (already flipped 180° about axis)
      X_mm:       (S,N) local X in mm  (your "negate x" convention preserved)
      Y_mm:       (S,N) local Y in mm
      loops_idx:  list[np.ndarray] index arrays for loops over ids
    """
    # ----- 1) Keep-mask stations to axis range (and apply to results) -----
    stations_mm_all = np.asarray(stations_m, float) * 1000.0          # input stations in meters -> mm
    smin, smax = float(np.min(axis.stations)), float(np.max(axis.stations))
    keep_mask = (stations_mm_all >= smin) & (stations_mm_all <= smax)
    if not np.any(keep_mask):
        return [], np.array([], dtype=float), np.zeros((0, 0, 3), float), np.zeros((0, 0), float), np.zeros((0, 0), float), []

    stations_mm = stations_mm_all[keep_mask]
    kept_results = [results[i] for i, k in enumerate(keep_mask) if k]  # <<< CRITICAL: keep results in sync with stations

    # ----- 2) Vectorized 2D section solve (build arrays, then solve once) -----
    variables   = json_data.get('Variables', {}) or {}
    used_names  = Utility.collect_used_variable_names(json_data)
    var_arrays  = Utility.build_var_arrays_from_results(kept_results, variables, keep=used_names)
    _fix_var_units_inplace(var_arrays)
    pre_graph = prepare_point_solver_cached(json_data)
    ids, X_mm, Y_mm = Utility.get_point_coords_vectorized(json_data, var_arrays, precomputed=pre_graph)
    # X_mm,Y_mm are shape (S,N) for S==len(stations_mm).  get_point_coords_vectorized already applies your "negate x".

    # ----- 3) Embed to global & rotate (vectorized) -----
    P_mm = Axis.embed_points_to_global_mm(axis, stations_mm, X_mm, Y_mm, twist_deg=twist_deg)
    # ----- 4) Build loop index arrays once (based on ids) -----
    loops_idx = loops_idx_cached(json_data, ids)

    return ids, stations_mm, P_mm, X_mm, Y_mm, loops_idx

def _clean_numbers(seq):
    out = []
    for v in seq:
        if v is None: continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fv): out.append(fv)
    return out

def _extend_numeric(dst, src):
    if not src: return
    for v in src:
        if v is None: dst.append(None)
        else:
            try:
                fv = float(v)
                if np.isfinite(fv): dst.append(fv)
            except (TypeError, ValueError):
                continue

def get_plot_traces(axis, json_data, stations_mm, ids, P_mm, X_mm, Y_mm,
                    obj_name, colors=None, loops_idx=None, show_labels=False):
    import plotly.graph_objects as go
    colors = {**{'axis':'black','first_station_points':'blue','cross_section_loops':'red','longitudinal_lines':'gray'}, **(colors or {})}
    traces = []

    # axis polyline (m)
    axis_x_m = (np.asarray(axis.x_coords)/1000.0).tolist()
    axis_y_m = (np.asarray(axis.y_coords)/1000.0).tolist()
    axis_z_m = (np.asarray(axis.z_coords)/1000.0).tolist()
    axis_meta = [{"type":"axis","obj":obj_name,"station_m":s} for s in (np.asarray(axis.stations)/1000.0).tolist()]
    traces.append(go.Scatter3d(x=axis_x_m,y=axis_y_m,z=axis_z_m,mode='lines',
        line=dict(color=colors['axis'],width=3,dash='dash'),name=f'{obj_name} Axis',
        customdata=axis_meta, hovertemplate="<b>%{customdata.obj}</b><br>Axis @ %{customdata.station_m:.3f} m<extra></extra>"
    ))

    if P_mm.size == 0:
        return traces, axis_x_m, axis_y_m, axis_z_m

    S, N, _ = P_mm.shape
    P_m = P_mm/1000.0

    # first station points with rich metadata
    P0 = P_m[0]; mask0 = ~np.isnan(P0).any(axis=1)
    x0 = P0[mask0,0].tolist(); y0 = P0[mask0,1].tolist(); z0 = P0[mask0,2].tolist()
    ids0 = [ids[j] for j,ok in enumerate(mask0) if ok]
    station0_m = float(stations_mm[0]/1000.0)
    meta0 = []
    for j, pid in enumerate(ids):
        if not mask0[j]: continue
        meta0.append({"type":"point","obj":obj_name,"id":pid,"station_m":station0_m,
                      "localX": X_mm[0,j] if not np.isnan(X_mm[0,j]) else None,
                      "localY": Y_mm[0,j] if not np.isnan(Y_mm[0,j]) else None,
                      "X_m": P0[j,0], "Y_m": P0[j,1], "Z_m": P0[j,2]})
    traces.append(go.Scatter3d(
        x=x0,y=y0,z=z0, mode='markers+text' if show_labels else 'markers',
        marker=dict(size=4,color=colors['first_station_points']),
        text=ids0 if show_labels else None, textposition='top center',
        name=f'{obj_name} Points @ {station0_m:.3f} m',
        customdata=meta0,
        hovertemplate="<b>%{customdata.obj}</b><br>Point: %{customdata.id}<br>Station: %{customdata.station_m:.3f} m<br>LocalX: %{customdata.localX}<br>LocalY: %{customdata.localY}<br>X: %{customdata.X_m:.3f} m<br>Y: %{customdata.Y_m:.3f} m<br>Z: %{customdata.Z_m:.3f} m<extra></extra>"
    ))

    # cross-section loops
    if loops_idx is None:
        loops = (json_data or {}).get('Loops', [])
        loops_idx = []
        id_to_col = {pid:j for j,pid in enumerate(ids)}
        for loop in loops:
            idxs = [id_to_col.get(p['Id']) for p in loop.get('Points', [])]
            idxs = [ix for ix in idxs if ix is not None]
            if idxs: loops_idx.append(np.asarray(idxs, int))

    loop_x, loop_y, loop_z, loop_meta = [], [], [], []
    for s in range(S):
        st_m = float(stations_mm[s]/1000.0)
        for li, idxs in enumerate(loops_idx):
            seg = P_m[s, idxs, :]
            valid = ~np.isnan(seg).any(axis=1)
            if not valid.any(): continue
            # draw contiguous runs; close if full loop valid
            run = []
            for k, ok in enumerate(valid):
                if ok:
                    run.append(k)
                else:
                    if len(run)>=2:
                        pts = seg[run,:]; close = (len(run)==len(idxs))
                        xs = pts[:,0].tolist(); ys = pts[:,1].tolist(); zs = pts[:,2].tolist()
                        if close: xs.append(xs[0]); ys.append(ys[0]); zs.append(zs[0])
                        loop_x.extend(xs+[None]); loop_y.extend(ys+[None]); loop_z.extend(zs+[None])
                        for kk in range(len(xs)): loop_meta.append({"type":"loop","obj":obj_name,"loop_index":li,"station_m":st_m})
                        loop_meta.append(None)
                    run = []
            if len(run)>=2:
                pts = seg[run,:]; close = (len(run)==len(idxs))
                xs = pts[:,0].tolist(); ys = pts[:,1].tolist(); zs = pts[:,2].tolist()
                if close: xs.append(xs[0]); ys.append(ys[0]); zs.append(zs[0])
                loop_x.extend(xs+[None]); loop_y.extend(ys+[None]); loop_z.extend(zs+[None])
                for kk in range(len(xs)): loop_meta.append({"type":"loop","obj":obj_name,"loop_index":li,"station_m":st_m})
                loop_meta.append(None)

    if loop_x:
        traces.append(go.Scatter3d(
            x=loop_x,y=loop_y,z=loop_z, mode='lines',
            line=dict(color=colors['cross_section_loops'], width=2),
            name=f'{obj_name} Cross Sections',
            customdata=loop_meta,
            hovertemplate="<b>%{customdata.obj}</b><br>Loop #%{customdata.loop_index}<br>Station: %{customdata.station_m:.3f} m<extra></extra>"
        ))

    # longitudinal lines
    long_x,long_y,long_z,long_meta = [],[],[],[]
    for j,_ in enumerate(ids):
        col = P_m[:,j,:]; valid = ~np.isnan(col).any(axis=1)
        if not valid.any(): continue
        for s in range(S):
            if valid[s]:
                st_m = float(stations_mm[s]/1000.0)
                long_x.append(col[s,0]); long_y.append(col[s,1]); long_z.append(col[s,2])
                long_meta.append({"type":"longitudinal","obj":obj_name,"station_m":st_m})
            else:
                if long_x and long_x[-1] is not None:
                    long_x.append(None); long_y.append(None); long_z.append(None); long_meta.append(None)
        if long_x and long_x[-1] is not None:
            long_x.append(None); long_y.append(None); long_z.append(None); long_meta.append(None)

    if long_x:
        traces.append(go.Scatter3d(
            x=long_x,y=long_y,z=long_z, mode='lines',
            line=dict(color=colors['longitudinal_lines'], width=1),
            name=f'{obj_name} Longitudinal',
            customdata=long_meta,
            hovertemplate="<b>%{customdata.obj}</b><br>Longitudinal<br>Station: %{customdata.station_m:.3f} m<extra></extra>"
        ))

    # return everything we plotted (meters); safe for range calc after filtering None
    all_x = axis_x_m + x0 + [v for v in long_x if v is not None] + [v for v in loop_x if v is not None]
    all_y = axis_y_m + y0 + [v for v in long_y if v is not None] + [v for v in loop_y if v is not None]
    all_z = axis_z_m + z0 + [v for v in long_z if v is not None] + [v for v in loop_z if v is not None]
    return traces, all_x, all_y, all_z


import plotly.graph_objects as go
import numpy as np

def _clean_json_lines(meta_obj):
    """Pretty-print object metadata as a list of lines for the right panel."""
    try:
        import json
        txt = json.dumps(meta_obj, indent=2, ensure_ascii=False)
    except Exception:
        return [str(meta_obj)]
    lines = []
    for raw in txt.splitlines():
        s = raw.strip()
        if s.endswith(','):
            s = s[:-1]
        if s.startswith('"') and '":' in s:
            k, rest = s.split('":', 1)
            k = k.strip('"')
            s = f"{k}:{rest}"
        if s:
            lines.append(s)
    return lines or ["(no metadata)"]

def get_plot_traces_matrix(
    axis,
    json_data,
    stations_mm,      # (S,) in mm (already filtered)
    ids,              # length N
    P_mm,             # (S,N,3) global coords in mm (already flipped)
    X_mm=None,        # (S,N) local Y in mm (for metadata)
    Y_mm=None,        # (S,N) local Z in mm (for metadata)
    obj_name="Object",
    colors=None,
    cls_obj=None,
    show_labels=False,
):
    """
    Fast, vectorized plotting with full metadata restored:
      • Axis = solid line + small markers (like original)
      • First-station points: rich metadata incl. globalX_m/globalY_m/globalZ_m
      • Cross-section loops: vertex metadata incl. localY_m/localZ_m + globalX/Y/Z
      • Longitudinal lines: FILTERED to loop points; customdata shows X/Y/Z + object meta lines
    """
    default_colors = {
        'axis': 'black',
        'first_station_points': 'blue',
        'cross_section_loops': 'red',
        'longitudinal_lines': 'gray',
    }
    colors = {**default_colors, **(colors or {})}

    traces = []
    S = int(np.asarray(stations_mm).size)
    if P_mm.ndim != 3:
        raise ValueError("P_mm must be (S,N,3)")
    N = P_mm.shape[1]

    # meters
    P_m = P_mm / 1000.0
    X_m = (X_mm / 1000.0) if X_mm is not None else None
    Y_m = (Y_mm / 1000.0) if Y_mm is not None else None
    stations_m = np.asarray(stations_mm, dtype=float) / 1000.0

    # ---------- Axis: solid + markers (no dashes) ----------
    axis_x_m = (np.asarray(axis.x_coords, float) / 1000.0).tolist()
    axis_y_m = (np.asarray(axis.y_coords, float) / 1000.0).tolist()
    axis_z_m = (np.asarray(axis.z_coords, float) / 1000.0).tolist()
    axis_stations_m = (np.asarray(axis.stations, float) / 1000.0).tolist()

    axis_meta = [{"type": "axis", "obj": obj_name, "station_m": s} for s in axis_stations_m]
    traces.append(
        go.Scatter3d(
            x=axis_x_m, y=axis_y_m, z=axis_z_m,
            mode='lines+markers',
            line=dict(color=colors['axis'], width=3),         # solid
            marker=dict(size=3, color=colors['axis'], opacity=0.9),
            name=f'{obj_name} Axis',
            customdata=axis_meta,
            hovertemplate="<b>%{customdata.obj}</b><br>Axis @ %{customdata.station_m:.3f} m<extra></extra>"
        )
    )

    if S == 0 or N == 0:
        return traces, axis_x_m, axis_y_m, axis_z_m

    # ---------- First-station points ----------
    P0 = P_m[0]                       # (N,3)
    valid0 = ~np.isnan(P0).any(axis=1)
    x0 = P0[valid0, 0].tolist()
    y0 = P0[valid0, 1].tolist()
    z0 = P0[valid0, 2].tolist()
    ids0 = [ids[j] for j, ok in enumerate(valid0) if ok]

    local_y0 = (X_m[0, valid0].tolist() if X_m is not None else [None]*len(ids0))
    local_z0 = (Y_m[0, valid0].tolist() if Y_m is not None else [None]*len(ids0))
    st0 = float(stations_m[0])

    first_meta = []
    for pid, ly, lz, gx, gy, gz in zip(ids0, local_y0, local_z0, x0, y0, z0):
        first_meta.append({
            "type": "point",
            "obj": obj_name,
            "id": pid,
            "station_m": st0,
            "localY_m": ly,
            "localZ_m": lz,
            "globalX_m": gx,   # camelCase keys (as original)
            "globalY_m": gy,
            "globalZ_m": gz,
        })

    traces.append(
        go.Scatter3d(
            x=x0, y=y0, z=z0,
            mode='markers+text' if show_labels else 'markers',
            marker=dict(size=4, opacity=0.9, color=colors['first_station_points']),
            text=ids0 if show_labels else None,
            textposition='top center',
            name=f'{obj_name} Points @ {st0:.3f} m',
            customdata=first_meta,
            hovertemplate=(
                "<b>%{customdata.obj}</b><br>"
                "Point: %{customdata.id}<br>"
                "Station: %{customdata.station_m:.3f} m<br>"
                "Local Y: %{customdata.localY_m:.3f} m<br>"
                "Local Z: %{customdata.localZ_m:.3f} m<br>"
                "X: %{customdata.globalX_m:.3f} m<br>"
                "Y: %{customdata.globalY_m:.3f} m<br>"
                "Z: %{customdata.globalZ_m:.3f} m<extra></extra>"
            )
        )
    )

    # ---------- Cross-section loops ----------
    loops = (json_data or {}).get('Loops', [])
    id_to_col = {pid: j for j, pid in enumerate(ids)}

    loops_idx = []
    for loop in loops:
        idxs = [id_to_col.get(p.get('Id')) for p in loop.get('Points', [])]
        idxs = [ix for ix in idxs if ix is not None]
        if idxs:
            loops_idx.append(np.asarray(idxs, dtype=int))

    all_loop_x, all_loop_y, all_loop_z, all_loop_meta = [], [], [], []

    for s in range(S):
        st_m = float(stations_m[s])
        for idxs in loops_idx:
            seg = P_m[s, idxs, :]                 # (L,3)
            valid = ~np.isnan(seg).any(axis=1)
            if not valid.any():
                continue

            run = []
            for k, ok in enumerate(valid):
                if ok:
                    run.append(k)
                else:
                    if len(run) >= 2:
                        run_pts = seg[run, :]
                        close = (len(run) == len(idxs))
                        xs = run_pts[:, 0].tolist()
                        ys = run_pts[:, 1].tolist()
                        zs = run_pts[:, 2].tolist()
                        if close:
                            xs.append(xs[0]); ys.append(ys[0]); zs.append(zs[0])

                        all_loop_x.extend(xs + [None])
                        all_loop_y.extend(ys + [None])
                        all_loop_z.extend(zs + [None])

                        # metadata per vertex (include globalX/Y/Z)
                        for kk in range(len(xs)):
                            base_idx = run[kk if kk < len(run) else 0]
                            pid = ids[idxs[base_idx]] if kk < len(run) else ids[idxs[run[0]]]
                            ly = float(X_m[s, idxs[base_idx]]) if X_m is not None else None
                            lz = float(Y_m[s, idxs[base_idx]]) if Y_m is not None else None
                            all_loop_meta.append({
                                "type": "loop",
                                "obj": obj_name,
                                "id": pid,
                                "station_m": st_m,
                                "localY_m": ly,
                                "localZ_m": lz,
                                "globalX_m": xs[kk],
                                "globalY_m": ys[kk],
                                "globalZ_m": zs[kk],
                            })
                        all_loop_meta.append(None)
                    run = []
            # tail
            if len(run) >= 2:
                run_pts = seg[run, :]
                close = (len(run) == len(idxs))
                xs = run_pts[:, 0].tolist()
                ys = run_pts[:, 1].tolist()
                zs = run_pts[:, 2].tolist()
                if close:
                    xs.append(xs[0]); ys.append(ys[0]); zs.append(zs[0])
                all_loop_x.extend(xs + [None])
                all_loop_y.extend(ys + [None])
                all_loop_z.extend(zs + [None])
                for kk in range(len(xs)):
                    base_idx = run[kk if kk < len(run) else 0]
                    pid = ids[idxs[base_idx]] if kk < len(run) else ids[idxs[run[0]]]
                    ly = float(X_m[s, idxs[base_idx]]) if X_m is not None else None
                    lz = float(Y_m[s, idxs[base_idx]]) if Y_m is not None else None
                    all_loop_meta.append({
                        "type": "loop",
                        "obj": obj_name,
                        "id": pid,
                        "station_m": st_m,
                        "localY_m": ly,
                        "localZ_m": lz,
                        "globalX_m": xs[kk],
                        "globalY_m": ys[kk],
                        "globalZ_m": zs[kk],
                    })
                all_loop_meta.append(None)

    if all_loop_x:
        traces.append(
            go.Scatter3d(
                x=all_loop_x, y=all_loop_y, z=all_loop_z,
                mode='lines',
                line=dict(color=colors['cross_section_loops'], width=2),
                name=f'{obj_name} Cross Sections',
                customdata=all_loop_meta,
                hovertemplate=(
                    "<b>%{customdata.obj}</b><br>"
                    "Point: %{customdata.id}<br>"
                    "Station: %{customdata.station_m:.3f} m<br>"
                    "Local Y: %{customdata.localY_m:.3f} m<br>"
                    "Local Z: %{customdata.localZ_m:.3f} m<br>"
                    "Global X: %{customdata.globalX_m:.3f} m<br>"
                    "Global Y: %{customdata.globalY_m:.3f} m<br>"
                    "Global Z: %{customdata.globalZ_m:.3f} m<extra></extra>"
                )
            )
        )

    # ---------- Longitudinal lines (filtered to loop points; include X/Y/Z in customdata lines) ----------
    loop_point_ids = set()
    for loop in loops:
        for p in loop.get('Points', []):
            pid = p.get('Id')
            if pid is not None:
                loop_point_ids.add(pid)

    obj_lines = _clean_json_lines(getattr(cls_obj, "get_object_metada", lambda: {})())

    long_x, long_y, long_z, long_meta = [], [], [], []
    for j, pid in enumerate(ids):
        if pid not in loop_point_ids:
            continue
        col = P_m[:, j, :]                       # (S,3)
        valid = ~np.isnan(col).any(axis=1)
        if not valid.any():
            continue
        for k in range(S - 1):
            if valid[k] and valid[k + 1]:
                p1, p2 = col[k], col[k + 1]
                long_x.extend([p1[0], p2[0], None])
                long_y.extend([p1[1], p2[1], None])
                long_z.extend([p1[2], p2[2], None])

                # Put per-vertex X/Y/Z *first* so your right-panel shows them,
                # then append object metadata lines.
                cmn1 = [f"X: {p1[0]:.3f} m", f"Y: {p1[1]:.3f} m", f"Z: {p1[2]:.3f} m", "—"] + obj_lines
                cmn2 = [f"X: {p2[0]:.3f} m", f"Y: {p2[1]:.3f} m", f"Z: {p2[2]:.3f} m", "—"] + obj_lines
                long_meta.extend([cmn1, cmn2, None])

    if long_x:
        traces.append(
            go.Scatter3d(
                x=long_x, y=long_y, z=long_z,
                mode='lines',
                line=dict(color=colors['longitudinal_lines'], width=1),
                name=f'{obj_name} Longitudinal',
                customdata=long_meta,
                 hovertemplate=(
                obj_name + " Longitudinal<br>" "X: %{x:.3f} m<br>" "Y: %{y:.3f} m<br>" "Z: %{z:.3f} m<extra></extra>"),
            )
        )

    # ---------- Return all coords (meters; may include None) ----------
    all_x_m = axis_x_m + x0 + all_loop_x + long_x
    all_y_m = axis_y_m + y0 + all_loop_y + long_y
    all_z_m = axis_z_m + z0 + all_loop_z + long_z
    return traces, all_x_m, all_y_m, all_z_m

# =========================
# Main
# =========================

if __name__ == "__main__":
    import os, sys, json
    import numpy as np
    import plotly.graph_objects as go

    from AddClasses import load_from_json, mapping
    from AxisVariables import AxisVariable
    from Axis import Axis
    from spot_loader import SpotLoader
    from Visualization import VisRenderer  

    MASTER_GIT = r"C:\Git\SPOT_VISO_krzys\SPOT_VISO\GIT"
    BRANCH     = "MAIN"
    COND_PIER_FOUNDATION = False

    loader = SpotLoader(MASTER_GIT, BRANCH, cond_pier_foundation=COND_PIER_FOUNDATION, verbose=True)
    vis_data_all, vis_objs_all = (
        loader.load_raw()
              .group_by_class()
              .build_vis_components(attach_section_json=True)
              .vis_data, loader.vis_objs
    )

    base_path = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(__file__)

    fig = go.Figure()
    renderer = VisRenderer()  # NEW

    all_x, all_y, all_z = [], [], []

    # Pass compute helpers to the renderer (no circular imports)
    helpers = dict(
        collect_used_variable_names=Utility.collect_used_variable_names,
        build_var_arrays_from_results=Utility.build_var_arrays_from_results,
        fix_var_units_inplace=_fix_var_units_inplace,                     # or Utility.fix_var_units_inplace if that's your name
        prepare_point_solver_cached=prepare_point_solver_cached,
        get_point_coords_vectorized=Utility.get_point_coords_vectorized,
    )

    for obj, cls_obj in zip(vis_data_all, vis_objs_all):
        # Prefer JSON already attached by SpotLoader
        json_data = obj.get('json_data')
        if json_data is None:
            jf = obj.get('json_file')
            if not jf:
                print(f"Skip {obj.get('name','Unknown')}: no json_file / json_data")
                continue
            try:
                with open(jf, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
            except Exception as e:
                print(f"Skip {obj.get('name')}: {e}")
                continue

        # Axis (convert meters -> mm inside Axis)
        axis = get_axis_cached_by_tuple(obj['stations_axis'], obj['x_coords'], obj['y_coords'], obj['z_coords'])

        axis_vars = AxisVariable.create_axis_variables(obj['AxisVariables'])
        stations_to_plot_m = obj['stations_to_plot']   # meters
        colors = obj.get('colors', {})
        twist_deg = obj.get('AxisRotation', 0.0)

        # Evaluate axis-variable curves at stations (list of dicts)
        results = AxisVariable.evaluate_at_stations_cached(axis_vars, stations_to_plot_m)

        # Build matrices via renderer (moved from main)
        ids, stations_mm, P_mm, X_mm, Y_mm, loops_idx = renderer.build_point_matrices(
            axis, json_data, results, stations_to_plot_m,
            twist_deg=twist_deg, helpers=helpers
        )

        # Plot with metadata via renderer
        traces, xs, ys, zs = renderer.get_plot_traces_matrix(
            axis, json_data, stations_mm, ids, P_mm,
            X_mm=X_mm, Y_mm=Y_mm,
            obj_name=obj['name'], colors=colors, cls_obj=cls_obj,
        )
        for t in traces:
            fig.add_trace(t)

        renderer._extend_numeric(all_x, xs)
        renderer._extend_numeric(all_y, ys)
        renderer._extend_numeric(all_z, zs)

    # Layout + export via renderer
    renderer.apply_equal_ranges(fig, all_x, all_y, all_z)
    renderer.save_as_website(fig, os.path.join(base_path, 'combined_objects'))
