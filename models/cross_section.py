from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
import numpy as np

# uses your utils.py (already in your repo)
from models.utils import _compile_expr, _VECTOR_FUNCS, _RESERVED_FUNC_NAMES


def _normalize_point_row(p: dict) -> dict:
    # Canonicalize to: {'Id': str, 'Coord': [str,str], 'Reference': [str]}
    if "Id" in p and ("Coord" in p or "coord" in p):
        pid = str(p["Id"])
        coord = p.get("Coord") or p.get("coord") or [0, 0]
        ref = p.get("Reference") or []
        return {"Id": pid, "Coord": [str(coord[0]), str(coord[1])], "Reference": list(ref or [])}

    pid = str(p.get("PointName") or p.get("Id") or p.get("id") or "")
    def pick(expr, val):
        txt = str(expr) if expr is not None else ""
        if not txt or txt.strip().lower().startswith("error"):
            return "0" if (val is None or str(val).strip() == "") else str(val)
        return txt
    y = pick(p.get("CoorY"), p.get("CoorYVal"))
    z = pick(p.get("CoorZ"), p.get("CoorZVal"))
    ref = p.get("Reference") or []
    return {"Id": pid, "Coord": [y, z], "Reference": list(ref or [])}


def _collect_points(section_json: Dict) -> Dict[str, Dict]:
    pts: Dict[str, Dict] = {}
    def add(p):
        if not isinstance(p, dict): return
        q = _normalize_point_row(p)
        pid = q.get("Id")
        if pid: pts.setdefault(pid, q)
    jd = section_json or {}
    for p in (jd.get("Points") or []): add(p)
    for lp in (jd.get("Loops") or []):
        for p in (lp.get("Points") or []): add(p)
    for pr in (jd.get("PointReinforcements") or []): add(pr.get("Point", {}))
    for lr in (jd.get("LineReinforcements") or []):
        add(lr.get("PointStart", {})); add(lr.get("PointEnd", {}))
    for nez in (jd.get("NonEffectiveZones") or []):
        add(nez.get("PointStart", {})); add(nez.get("PointEnd", {}))
    return pts


def _topo_order(points: Dict[str, Dict]) -> List[str]:
    incoming = {pid: set(points[pid].get("Reference", [])) for pid in points}
    outgoing: Dict[str, set] = {pid: set() for pid in points}
    for pid, refs in incoming.items():
        for r in refs:
            if r in outgoing: outgoing[r].add(pid)
    order: List[str] = []
    roots = sorted([pid for pid, refs in incoming.items() if not refs])
    queue = list(roots)
    while queue:
        u = queue.pop(0)
        order.append(u)
        for w in list(outgoing.get(u, ())):
            incoming[w].discard(u)
            if not incoming[w]:
                queue.append(w)
    for pid in points:
        if pid not in order:
            order.append(pid)
    return order


def _used_vars(points: Dict[str, Dict]) -> set[str]:
    used: set[str] = set()
    for p in points.values():
        for expr in (p.get("Coord") or ["0", "0"])[:2]:
            token = ""
            for ch in str(expr):
                if ch.isalnum() or ch == "_":
                    token += ch
                else:
                    if token and token not in _RESERVED_FUNC_NAMES and not token[0].isdigit():
                        used.add(token)
                    token = ""
            if token and token not in _RESERVED_FUNC_NAMES and not token[0].isdigit():
                used.add(token)
    return used


def _loops_idx(section_json: Dict, id_to_col: Dict[str,int]) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for lp in (section_json.get("Loops") or []):
        idxs = [id_to_col.get(p.get("Id")) for p in (lp.get("Points") or [])]
        idxs = [i for i in idxs if i is not None]
        if len(idxs) >= 2:
            out.append(np.asarray(idxs, dtype=int))
    return out


def _fix_var_units_inplace(arrs: Dict[str, np.ndarray], defaults_mm: Dict[str, float]) -> None:
    # Heuristic: deck axis vars are mostly meters → scale ×1000 unless default suggests otherwise
    for name, a in list(arrs.items()):
        a = np.asarray(a, float)
        if a.size == 0: continue
        def_v = defaults_mm.get(name)
        scale = 1000.0
        if def_v is not None and np.isfinite(def_v):
            med = float(np.nanmedian(np.abs(a))) or 0.0
            cands = (1.0, 1000.0)
            costs = [abs(med*s - def_v)/max(1.0, abs(def_v)) for s in cands]
            scale = cands[int(np.argmin(costs))]
        arrs[name] = a * scale


@dataclass
class CrossSection:
    """Single responsibility: evaluate local (Y,Z) from a section JSON."""
    name: str
    data: Dict

    @classmethod
    def from_file(cls, path: str, name: Optional[str]=None) -> "CrossSection":
        with open(path, "r", encoding="utf-8") as f:
            jd = json.load(f)
        return cls(name or jd.get("Name") or "Section", jd)

    def defaults_mm(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, v in (self.data.get("Variables") or {}).items():
            try: out[str(k)] = float(v)
            except: pass
        return out

    def evaluate(
        self,
        var_arrays_all: Dict[str, np.ndarray],
        *,
        negate_x: bool = True
    ) -> Tuple[List[str], np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Vectorized evaluation for all stations.
        Returns: ids, X_mm:(S,N), Y_mm:(S,N), loops_idx:[np.ndarray]
        """
        points = _collect_points(self.data)
        order = _topo_order(points)
        id_to_idx = {pid: j for j, pid in enumerate(order)}
        S = len(next(iter(var_arrays_all.values()))) if var_arrays_all else 0
        N = len(order)
        X = np.full((S, N), np.nan, float)
        Y = np.full((S, N), np.nan, float)

        used = _used_vars(points)
        defaults_mm = self.defaults_mm()

        # keep only needed vars; fill missing with defaults (or 0)
        vars_raw = {k: np.asarray(v, float) for k, v in var_arrays_all.items() if k in used or k in defaults_mm}
        for k in used:
            if k not in vars_raw:
                vars_raw[k] = np.full(S, float(defaults_mm.get(k, 0.0)), float)

        _fix_var_units_inplace(vars_raw, defaults_mm)

        env = {**_VECTOR_FUNCS, **vars_raw}
        compiled = {pid: (_compile_expr((points[pid].get("Coord") or ["0","0"])[0]),
                          _compile_expr((points[pid].get("Coord") or ["0","0"])[1]))
                    for pid in order}

        # base coords
        for j, pid in enumerate(order):
            cx, cy = compiled[pid]
            try:
                xv = np.asarray(eval(cx, {"__builtins__": {}}, env), float)
            except Exception:
                xv = np.full(S, np.nan, float)
            try:
                yv = np.asarray(eval(cy, {"__builtins__": {}}, env), float)
            except Exception:
                yv = np.full(S, np.nan, float)
            X[:, j] = xv; Y[:, j] = yv

        # Euclidean references
        for j, pid in enumerate(order):
            refs = points[pid].get("Reference") or []
            if len(refs) == 1:
                k = id_to_idx.get(refs[0])
                if k is not None: X[:, j] += X[:, k]; Y[:, j] += Y[:, k]
            elif len(refs) >= 2:
                kx = id_to_idx.get(refs[0]); ky = id_to_idx.get(refs[1])
                if kx is not None: X[:, j] += X[:, kx]
                if ky is not None: Y[:, j] += Y[:, ky]

        if negate_x: X = -X
        loops = _loops_idx(self.data, id_to_idx)
        return order, X, Y, loops


# # ----------------------------------------------------------------------------- #
# # Minimal test (runs if you execute this file directly)
# # ----------------------------------------------------------------------------- #
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)

#     # tiny axis stub: stations in mm, embed copies Y/Z into world at x=0 (no twist)
#     class _AxisStub:
#         def __init__(self, stations_mm):
#             self.stations = np.asarray(stations_mm, float)

#         def embed_section_points_world(self, stations_mm, yz_points_mm, x_offsets_mm=None, rotation_deg=0.0):
#             # expected shape: (S, N, 2) -> return (S, N, 3)
#             if yz_points_mm.ndim != 3 or yz_points_mm.shape[2] != 2:
#                 raise ValueError("yz_points_mm must be (S, N, 2).")
#             S, N, _ = yz_points_mm.shape
#             P = np.zeros((S, N, 3), float)
#             # simple: world X=0, Y=local_Y, Z=local_Z
#             P[:, :, 1] = yz_points_mm[:, :, 0]
#             P[:, :, 2] = yz_points_mm[:, :, 1]
#             return P

#     # sample JSON (short)
#     json_data = {
#         "Points": [
#             {"Id": "O",  "Coord": ["0", "0"]},
#             {"Id": "UM", "Coord": ["0", "H_QS"]},
#             {"Id": "UL", "Coord": ["1000", "H_QS"]},
#         ],
#         "Loops": [
#             {"Points": [{"Id": "O"}, {"Id": "UM"}, {"Id": "UL"}]}
#         ],
#         "Variables": {"H_QS": 2750.0}
#     }

#     json_data ={
#     "Name":"section_0100.cdb",
#     "Id":100,
#     "Unit":"mm",
#     "MaterialId":1,
#     "Points":
#     [
#         {"Id":"O", "Coord":["0", "0"], "PointType":"StressPoint"},
#         {"Id":"UM", "Coord":["0", "H_QS"], "ReferenceType":"Euclidean", "PointType":"ConstructionPoint"},
#         {"Id":"UL", "Coord":["-37+B_TR/2-(1/TAN((W_ST)*(Pi/180))*(H_QS-T_FBA/COS((Q_NG/100)*(Pi/180))-B_TR/2*Q_NG/100))", "H_QS"], "ReferenceType":"Euclidean", "PointType":"StressPoint"},
#         {"Id":"UR", "Coord":["-37-B_TR/2+(1/TAN((W_ST)*(Pi/180))*(H_QS-T_FBA/COS((Q_NG/100)*(Pi/180))+B_TR/2*Q_NG/100))", "H_QS"], "ReferenceType":"Euclidean", "PointType":"StressPoint"},
#         {"Id":"AKL", "Coord":["B_TR/2-37", "+T_FBA/COS((Q_NG/100)*(Pi/180))+B_TR/2*Q_NG/100"], "ReferenceType":"Euclidean", "PointType":"StressPoint"},
#         {"Id":"AKR", "Coord":["-B_TR/2-37", "+T_FBA/COS((Q_NG/100)*(Pi/180))-B_TR/2*Q_NG/100"], "ReferenceType":"Euclidean", "PointType":"StressPoint"},
#         {"Id":"OL", "Coord":["(B_FB/2)+EX", "((B_FB/2+EX-1900)*Q_NG/100)-1900*0.025"], "ReferenceType":"Euclidean", "PointType":"ConstructionPoint"},
#         {"Id":"OR", "Coord":["(-B_FB/2)+EX-75", "(-B_FB/2+EX)*Q_NG/100"], "ReferenceType":"Euclidean", "PointType":"ConstructionPoint"},
#         {"Id":"1L", "Coord":["-299.9999523162842", "100.00002384185791"], "Reference":["AKL"], "ReferenceType":"Euclidean", "PointType":"StressPoint"},
#         {"Id":"2L", "Coord":["-400.00009536743164", "400.00003576278687"], "Reference":["AKL"], "ReferenceType":"Euclidean", "PointType":"StressPoint"},
#         {"Id":"3L", "Coord":["-500", "699.999988079071"], "Reference":["AKL"], "ReferenceType":"Euclidean", "PointType":"StressPoint"},
#         {"Id":"4L", "Coord":["-600.0001430511475", "1100.0000834465027"], "Reference":["AKL"], "ReferenceType":"Euclidean", "PointType":"StressPoint"},
#         {"Id":"5L", "Coord":["-700.0000476837158", "1400.0000357627869"], "Reference":["AKL"], "ReferenceType":"Euclidean", "PointType":"StressPoint"}
#     ],
#     "Loops":
#     [
#         {
#             "Points":
#             [
#                 {"Id":"100", "Coord":["0", "0"], "Reference":["UM"], "ReferenceType":"Euclidean"},
#                 {"Id":"115", "Coord":["0", "0"], "Reference":["UR"], "ReferenceType":"Euclidean"},
#                 {"Id":"114", "Coord":["0", "0"], "Reference":["AKR"], "ReferenceType":"Euclidean"},
#                 {"Id":"113", "Coord":["T_ST/COS((90-W_ST)*(Pi/180))-(T_FBA-T_FBI)/TAN((W_ST)*(Pi/180))", "(T_ST/COS((90-W_ST)*(Pi/180)))*Q_NG/100+(T_FBI-T_FBA)"], "Reference":["AKR"], "ReferenceType":"Euclidean"},
#                 {"Id":"112", "Coord":["T_ST/COS((90-W_ST)*(Pi/180))-T_BA*TAN((90-W_ST)*(Pi/180))", "-T_BA"], "Reference":["UR"], "ReferenceType":"Euclidean"},
#                 {"Id":"111", "Coord":["T_ST/COS((90-W_ST)*(Pi/180))-T_BA*TAN((90-W_ST)*(Pi/180))+B_BV", "-T_BM"], "Reference":["UR"], "ReferenceType":"Euclidean"},
#                 {"Id":"106", "Coord":["-T_ST/COS((90-W_ST)*(Pi/180))+T_BA*TAN((90-W_ST)*(Pi/180))-B_BV", "-T_BM"], "Reference":["UL"], "ReferenceType":"Euclidean"},
#                 {"Id":"105", "Coord":["-T_ST/COS((90-W_ST)*(Pi/180))+T_BA*TAN((90-W_ST)*(Pi/180))", "-T_BA"], "Reference":["UL"], "ReferenceType":"Euclidean"},
#                 {"Id":"104", "Coord":["-T_ST/COS((90-W_ST)*(Pi/180))+(T_FBA-T_FBI)/TAN((W_ST)*(Pi/180))", "(-T_ST/COS((90-W_ST)*(Pi/180)))*Q_NG/100+(T_FBI-T_FBA)"], "Reference":["AKL"], "ReferenceType":"Euclidean"},
#                 {"Id":"103", "Coord":["0", "0"], "Reference":["AKL"], "ReferenceType":"Euclidean"},
#                 {"Id":"102", "Coord":["0", "0"], "Reference":["UL"], "ReferenceType":"Euclidean"}
#             ],
#             "Id":"L1",
#             "MaterialId":"1"
#         },
#         {
#             "Points":
#             [
#                 {"Id":"120", "Coord":["0", "0"], "Reference":["AKL"], "ReferenceType":"Euclidean"},
#                 {"Id":"136", "Coord":["-T_ST/COS((90-W_ST)*(Pi/180))+(T_FBA-T_FBI)/TAN((W_ST)*(Pi/180))", "(-T_ST/COS((90-W_ST)*(Pi/180)))*Q_NG/100+(T_FBI-T_FBA)"], "Reference":["AKL"], "ReferenceType":"Euclidean"},
#                 {"Id":"135", "Coord":["B_FBV/2", "(T_FBM/COS((Q_NG/100)*(Pi/180)))+B_FBV/2*Q_NG/100"], "ReferenceType":"Euclidean"},
#                 {"Id":"134", "Coord":["-B_FBV/2", "(T_FBM/COS((Q_NG/100)*(Pi/180)))-B_FBV/2*Q_NG/100"], "ReferenceType":"Euclidean"},
#                 {"Id":"132", "Coord":["T_ST/COS((90-W_ST)*(Pi/180))-(T_FBA-T_FBI)/TAN((W_ST)*(Pi/180))", "(T_ST/COS((90-W_ST)*(Pi/180)))*Q_NG/100+(T_FBI-T_FBA)"], "Reference":["AKR"], "ReferenceType":"Euclidean"},
#                 {"Id":"131", "Coord":["0", "0"], "Reference":["AKR"], "ReferenceType":"Euclidean"},
#                 {"Id":"130", "Coord":["B_KK+1", "T_KP+B_KK*Q_NG/100"], "Reference":["OR"], "ReferenceType":"Euclidean"},
#                 {"Id":"128", "Coord":["0", "T_KP"], "Reference":["OR"], "ReferenceType":"Euclidean"},
#                 {"Id":"127", "Coord":["0", "0"], "Reference":["OR"], "ReferenceType":"Euclidean"},
#                 {"Id":"126", "Coord":["0", "0"]},
#                 {"Id":"125", "Coord":["-1900.0000953674316", "47.49999940395355"], "Reference":["OL"], "ReferenceType":"Euclidean"},
#                 {"Id":"124", "Coord":["0", "0"], "Reference":["OL"], "ReferenceType":"Euclidean"},
#                 {"Id":"123", "Coord":["0", "T_KP"], "Reference":["OL"], "ReferenceType":"Euclidean"},
#                 {"Id":"122", "Coord":["-B_KK-1", "T_KP+B_KK*0.025"], "Reference":["OL"], "ReferenceType":"Euclidean"}
#             ],
#             "Id":"L2",
#             "MaterialId":"2"
#         }
#     ],
#     "PointReinforcements":
#     [
#         {
#             "Point": {"Id":"PRF150", "Coord":["-82.49998092651367", "-82.49998092651367"], "Reference":["UL"], "ReferenceType":"Euclidean"},
#             "Id":"150",
#             "Layer":"7",
#             "MaterialId":"15",
#             "TorsionalContribution":"Acti",
#             "Diameter":"12"
#         },
#         {
#             "Point": {"Id":"PRF151", "Coord":["82.49998092651367", "-82.49998092651367"], "Reference":["UR"], "ReferenceType":"Euclidean"},
#             "Id":"151",
#             "Layer":"7",
#             "MaterialId":"15",
#             "TorsionalContribution":"Acti",
#             "Diameter":"12"
#         },
#         {
#             "Point": {"Id":"PRF152", "Coord":["51.49984359741211", "-517.500028014183"], "Reference":["AKL"], "ReferenceType":"Euclidean"},
#             "Id":"152",
#             "Layer":"7",
#             "MaterialId":"15",
#             "TorsionalContribution":"Acti",
#             "Diameter":"12"
#         },
#         {
#             "Point": {"Id":"PRF153", "Coord":["-51.49984359741211", "-517.5000242888927"], "Reference":["AKR"], "ReferenceType":"Euclidean"},
#             "Id":"153",
#             "Layer":"7",
#             "MaterialId":"15",
#             "TorsionalContribution":"Acti",
#             "Diameter":"12"
#         }
#     ],
#     "LineReinforcements":
#     [
#         {
#             "BarDistribution":"Even",
#             "Spacing":"150",
#             "BarCount":"0",
#             "PointStart": {"Id":"LRF100Start", "Coord":["-174.99995231628418", "-174.99995231628418"], "Reference":["UL"], "ReferenceType":"Euclidean"},
#             "PointEnd": {"Id":"LRF100End", "Coord":["174.99995231628418", "-174.99995231628418"], "Reference":["UR"], "ReferenceType":"Euclidean"},
#             "Id":"100",
#             "Layer":"1",
#             "MaterialId":"10",
#             "TorsionalContribution":"Pass",
#             "Diameter":"25"
#         },
#         {
#             "BarDistribution":"Even",
#             "Spacing":"150",
#             "BarCount":"0",
#             "PointStart": {"Id":"LRF300Start", "Coord":["-250", "-174.99995231628418"], "Reference":["UL"], "ReferenceType":"Euclidean"},
#             "PointEnd": {"Id":"LRF300End", "Coord":["-250", "19.999980926513672"], "Reference":["AKL"], "ReferenceType":"Euclidean"},
#             "Id":"300",
#             "Layer":"3",
#             "MaterialId":"10",
#             "TorsionalContribution":"Pass",
#             "Diameter":"20"
#         },
#         {
#             "BarDistribution":"Even",
#             "Spacing":"150",
#             "BarCount":"0",
#             "PointStart": {"Id":"LRF310Start", "Coord":["250", "-174.99995231628418"], "Reference":["UR"], "ReferenceType":"Euclidean"},
#             "PointEnd": {"Id":"LRF310End", "Coord":["250", "20.00001072883606"], "Reference":["AKR"], "ReferenceType":"Euclidean"},
#             "Id":"310",
#             "Layer":"4",
#             "MaterialId":"10",
#             "TorsionalContribution":"Pass",
#             "Diameter":"20"
#         },
#         {
#             "BarDistribution":"Even",
#             "Spacing":"150",
#             "BarCount":"0",
#             "PointStart": {"Id":"LRF200Start", "Coord":["-82.49998092651367", "125"], "Reference":["OL"], "ReferenceType":"Euclidean"},
#             "PointEnd": {"Id":"LRF200End", "Coord":["-1724.9999046325684", "225.0000238418579"], "Reference":["OL"], "ReferenceType":"Euclidean"},
#             "Id":"200",
#             "Layer":"2",
#             "MaterialId":"11",
#             "TorsionalContribution":"Pass",
#             "Diameter":"20"
#         },
#         {
#             "BarDistribution":"Even",
#             "Spacing":"150",
#             "BarCount":"0",
#             "PointStart": {"Id":"LRF250Start", "Coord":["-1724.9999046325684", "225.0000238418579"], "Reference":["OL"], "ReferenceType":"Euclidean"},
#             "PointEnd": {"Id":"LRF250End", "Coord":["82.49998092651367", "125"], "Reference":["OR"], "ReferenceType":"Euclidean"},
#             "Id":"250",
#             "Layer":"2",
#             "MaterialId":"11",
#             "TorsionalContribution":"Pass",
#             "Diameter":"20"
#         }
#     ],
#     "NonEffectiveZones":[{
#         "Id":"BP1",
#         "Type":"ZV",
#         "PointStart": {"Id":"NEFFBP1Min", "Coord":["-BEFF_BP/2", "-1000"], "Reference":["UM"], "ReferenceType":"Euclidean"},
#         "PointEnd": {"Id":"NEFFBP1Max", "Coord":["BEFF_BP/2", "1000"], "Reference":["UM"], "ReferenceType":"Euclidean"}
#     },
#     {
#         "Id":"FPL",
#         "Type":"ZV",
#         "PointStart": {"Id":"NEFFFPLMin", "Coord":["-BEFF_KR", "-1000.0000596046448"], "Reference":["OL"], "ReferenceType":"Euclidean"},
#         "PointEnd": {"Id":"NEFFFPLMax", "Coord":["BEFF_KR", "1000.0000596046448"], "Reference":["OL"], "ReferenceType":"Euclidean"}
#     },
#     {
#         "Id":"FPR",
#         "Type":"ZV",
#         "PointStart": {"Id":"NEFFFPRMin", "Coord":["-BEFF_KR", "1000.000074505806"], "Reference":["OR"], "ReferenceType":"Euclidean"},
#         "PointEnd": {"Id":"NEFFFPRMax", "Coord":["BEFF_KR", "-1000.0001043081284"], "Reference":["OR"], "ReferenceType":"Euclidean"}
#     },
#     {
#         "Id":"FPM",
#         "Type":"ZV",
#         "PointStart": {"Id":"NEFFFPMMin", "Coord":["-BEFF_FB/2", "-700.0000476837158"], "Reference":["O"], "ReferenceType":"Euclidean"},
#         "PointEnd": {"Id":"NEFFFPMMax", "Coord":["BEFF_FB/2", "700.0000476837158"], "Reference":["O"], "ReferenceType":"Euclidean"}
#     }],
#     "Variables": {"H_QS":2750.000238418579, "B_TR":8575.000762939453, "W_ST":75.0, "T_FBA":600.0000238418579, "Q_NG":3.0, "B_FB":15900.00057220459, "EX":0.0, "T_ST":500.0, "T_FBI":600.0000238418579, "T_BA":600.0000238418579, "B_BV":2000.0, "T_BM":350.0000238418579, "B_KK":500.0, "T_KP":250.0, "B_FBV":3000.000238418579, "T_FBM":350.0000238418579, "BEFF_BP":100.00000149011612, "BEFF_KR":2500.0, "BEFF_FB":100.00000149011612},
#     "VariableDescriptions":
#     {
#         "H_QS":"H_QS:Axis Variable",
#         "B_TR":"B_TR:Axis Variable",
#         "W_ST":"W_ST:Axis Variable",
#         "T_FBA":"T_FBA:Axis Variable",
#         "Q_NG":"Q_NG:Axis Variable",
#         "B_FB":"B_FB:Axis Variable",
#         "EX":"EX:Axis Variable",
#         "T_ST":"T_ST:Axis Variable",
#         "T_FBI":"T_FBI:Axis Variable",
#         "T_BA":"T_BA:Axis Variable",
#         "B_BV":"B_BV:Axis Variable",
#         "T_BM":"T_BM:Axis Variable",
#         "B_KK":"B_KK:Axis Variable",
#         "T_KP":"T_KP:Axis Variable",
#         "B_FBV":"B_FBV:Axis Variable",
#         "T_FBM":"T_FBM:Axis Variable",
#         "BEFF_BP":"BEFF_BP:Axis Variable",
#         "BEFF_KR":"BEFF_KR:Axis Variable",
#         "BEFF_FB":"BEFF_FB:Axis Variable"
#     }
#     }

#     cs = CrossSection.from_dict({
#         "No": "1", "Class": "CrossSection", "Type": "Deck", "Name": "MASTER_Deck",
#         "NCS": 111, "JSON_name": ["MASTER_SECTION/MASTER_Pier.json"], "Variables": [{"VariableName": "H_QS", "VariableValue": 2750.0}],
#     })
#     # cs.attach_json_payload(json_data)
#     # cs.inflate_points_from_json()
#     # cs.prime_caches()

#     # two stations; axis variables in meters here to exercise ×1000
#     axis_vars = [{"H_QS": 6.75}, {"H_QS": 2.75}]
#     ids_, X, Y, loops = cs.compute_local_points(axis_var_results=axis_vars, negate_x=True)
#     print("ids:", ids_)
#     print("X[0,:]:", X[0])
#     print("Y[0,:]:", Y[0])
#     print("loops:", loops)

#     axis = _AxisStub(stations_mm=[0.0, 1000.0, 2000.0])
#     ids, S_mm, P_mm, X_mm, Y_mm, loops_idx = cs.compute_embedded_points(
#         axis=axis, axis_var_results=axis_vars, stations_m=[0.0, 1.0], twist_deg=0.0, negate_x=True
#     )
#     print("embedded shape:", P_mm.shape)

#     from viso_ploter import plot_local_points_with_axes_table
   
#     plot_local_points_with_axes_table(ids_, X, Y, loops)

    
