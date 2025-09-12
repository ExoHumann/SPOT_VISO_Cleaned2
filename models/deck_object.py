from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional
import numpy as np
from .base import BaseObject
from .axis import Axis
from .cross_section import CrossSection
from .axis_variable import AxisVariable
def _resolve_section_key(ctx, key):
    if key is None: return None
    if isinstance(key, str):
        return getattr(ctx, "crosssec_by_name", {}).get(key)
    try:
        return getattr(ctx, "crosssec_by_ncs", {}).get(int(key))
    except Exception:
        return None

def _sections_for_stations(ctx, stations_m, *, schedule=None, names=None, ncs_list=None):
    S = len(stations_m)
    if not S: return []
    # piecewise-constant schedule [{'station_m':..., 'name'/'ncs':...}, ...]
    if schedule:
        pairs = []
        for it in schedule:
            sm = it.get("station_m", it.get("station_mm", None))
            if sm is None: continue
            sm = float(sm) if "station_m" in it else float(sm)/1000.0
            key = it.get("name", it.get("ncs", None))
            sec = _resolve_section_key(ctx, key)
            if sec: pairs.append((sm, sec))
        pairs.sort(key=lambda x: x[0])
        out, j = [], 0
        for s in stations_m:
            while j+1 < len(pairs) and pairs[j+1][0] <= s: j += 1
            out.append(pairs[j][1] if pairs else None)
        return out

    # per-station names
    if names and len(names) == S:
        return [_resolve_section_key(ctx, nm) for nm in names]

    # constant fallback
    for seq in (names or []), (ncs_list or []):
        for key in (seq or []):
            sec = _resolve_section_key(ctx, key)
            if sec: return [sec]*S
    return [None]*S


@dataclass(kw_only=True)
class DeckObject(BaseObject):
    no: str = ""
    class_name: str = ""
    type: str = ""
    description: str = ""
    name: str = ""         # keeps your current override; fine
    inactive: str = ""

    cross_section_types: List[str] = field(default_factory=list)
    cross_section_names: List[str] = field(default_factory=list)
    grp_offset: List[float] = field(default_factory=list)
    placement_id: List[str] = field(default_factory=list)
    placement_description: List[str] = field(default_factory=list)
    ref_placement_id: List[str] = field(default_factory=list)
    ref_station_offset: List[float] = field(default_factory=list)
    station_value: List[float] = field(default_factory=list)
    cross_section_points_name: List[str] = field(default_factory=list)
    grp: List[str] = field(default_factory=list)
    cross_section_ncs: List[int] = field(default_factory=list)


    def get_object_metada(self):
        data = asdict(self)
        # Replace or remove verbose fields
        data['axis_variables'] = f"<{len(self.axis_variables_obj)} axis variables>"
        data['axis_variables_obj'] = f"<{len(self.axis_variables_obj)} axis variable objects>"
        data['axis_obj'] = f"<Axis object>" if self.axis_obj is not None else None


        # Remove 'colors' key from output
        data.pop('colors', None)  # 'None' avoids KeyError if it's missing
        data.pop('user_stations', None)  # 'None' avoids KeyError if it's missing
        data.pop('axis_obj', None)  # 'None' avoids KeyError if it's missing
        data.pop('axis_variables_obj', None)  # 'None' avoids KeyError if it's missing
        data.pop('axis_rotation', None)  # 'None' avoids KeyError if it's missing


        return data
    
    # Optional pretty metadata (preserves your existing style)
    def get_object_metadata(self) -> Dict:
        data = asdict(self)
        data['axis_variables'] = f"<{len(self.axis_variables_obj)} axis variables>"
        data['axis_variables_obj'] = f"<{len(self.axis_variables_obj)} axis variable objects>"
        data['axis_obj'] = "<Axis object>" if getattr(self, "axis_obj", None) is not None else None
        # remove UI-only noise (match your current trimming)
        data.pop('colors', None)
        data.pop('user_stations', None)
        data.pop('axis_obj', None)
        data.pop('axis_variables_obj', None)
        data.pop('axis_rotation', None)
        return data

    # ------------------------------------------------------------
    # New: Deck geometry (2D local -> 3D embedded)
    # ------------------------------------------------------------
    def compute_geometry(
    self,
    *,
    ctx,                                # VisoContext
    stations_m: Optional[List[float]] = None,
    twist_deg: float = 0.0,
    negate_x: bool = True,
) -> Dict[str, object]:
        """
        Compute 3D geometry for this object:
        - picks stations (or uses given),
        - evaluates axis variables at those stations,
        - picks which CrossSection applies at each station (schedule/names/NCS),
        - runs CrossSection.compute_embedded_points in contiguous batches,
        - concatenates the results along stations.

        Returns:
        dict(ids, stations_mm, points_mm, local_Y_mm, local_Z_mm, loops_idx)
        """
        axis: Optional[Axis] = getattr(self, "axis_obj", None)
        EMPTY = {
            "ids": [],
            "stations_mm": np.array([], float),
            "points_mm": np.zeros((0, 0, 3), float),
            "local_Y_mm": np.zeros((0, 0), float),
            "local_Z_mm": np.zeros((0, 0), float),
            "loops_idx": [],
        }
        if axis is None:
            return EMPTY

        # ---- Stations (meters) -------------------------------------------------
        if stations_m is None:
            S = np.asarray(getattr(axis, "stations", []), dtype=float)
            stations_m = (S / 1000.0).tolist()
        stations_m = [float(s) for s in (stations_m or [])]
        if not stations_m:
            return EMPTY

        # ---- Axis variables: per-station dict rows -----------------------------
        axis_results = AxisVariable.evaluate_at_stations_cached(
            getattr(self, "axis_variables_obj", []) or [], stations_m
        )

        # ---- Decide which section to use per station ---------------------------
        # You can expose a schedule like:
        #   self.section_schedule = [{"station_m": 0.0, "ncs": 111}, {"station_m": 45.0, "name": "MASTER_Pier"}, ...]
        sec_per_station: List[Optional[CrossSection]] = _sections_for_stations(
            ctx,
            stations_m,
            schedule=getattr(self, "section_schedule", None),
            names=getattr(self, "cross_section_names", None),
            ncs_list=getattr(self, "cross_section_ncs", None),
        )

        # Fallback: if nothing resolved, try any pre-resolved _cross_sections list or a single ctx fallback
        if not any(sec_per_station):
            # a) pre-resolved on the object
            pre = next((cs for cs in (getattr(self, "_cross_sections", None) or []) if isinstance(cs, CrossSection)), None)
            # b) by name/type
            if pre is None and hasattr(ctx, "crosssec_by_name"):
                for nm in (getattr(self, "cross_section_names", None) or []) + (getattr(self, "cross_section_types", None) or []):
                    pre = ctx.crosssec_by_name.get(nm)
                    if pre:
                        break
            # c) by NCS
            if pre is None and hasattr(ctx, "crosssec_by_ncs"):
                for ncs in (getattr(self, "cross_section_ncs", None) or []):
                    try:
                        pre = ctx.crosssec_by_ncs.get(int(ncs))
                    except Exception:
                        pre = None
                    if pre:
                        break
            if pre:
                sec_per_station = [pre] * len(stations_m)

        # If we still have no section at all, return empty
        if not any(sec_per_station):
            return EMPTY

        # ---- Effective twist (object's axis_rotation + caller) -----------------
        twist_eff = float(getattr(self, "axis_rotation", 0.0) or 0.0) + float(twist_deg or 0.0)

        # ---- Batch by contiguous section to minimize calls ---------------------
        def _flush_batch(cs: CrossSection, i0: int, i1: int):
            # slice of stations and axis_results for [i0, i1)
            sm_slice = stations_m[i0:i1]
            ar_slice = axis_results[i0:i1]
            if not sm_slice:
                return None
            ids, S_mm, P_mm, X_mm, Y_mm, loops_idx = cs.compute_embedded_points(
                axis=axis,
                axis_var_results=ar_slice,
                stations_m=sm_slice,
                twist_deg=90,
                negate_x=negate_x,
            )
            return (ids, S_mm, P_mm, X_mm, Y_mm, loops_idx)

        out_ids = None
        out_loops = None
        stations_all = []
        P_all = []
        X_all = []
        Y_all = []

        i = 0
        N = len(stations_m)
        while i < N:
            cs_i = sec_per_station[i]
            if cs_i is None:
                i += 1
                continue
            j = i + 1
            # extend while the section object remains the same
            while j < N and sec_per_station[j] is cs_i:
                j += 1

            res = _flush_batch(cs_i, i, j)
            if res is not None:
                ids, S_mm, P_mm, X_mm, Y_mm, loops_idx = res
                # Keep ids/loops from the first non-empty batch; if later batches differ, we keep the first
                if out_ids is None and ids:
                    out_ids = ids
                    out_loops = loops_idx
                # Accumulate along station dimension
                stations_all.append(np.asarray(S_mm, dtype=float))
                P_all.append(np.asarray(P_mm, dtype=float))
                X_all.append(np.asarray(X_mm, dtype=float))
                Y_all.append(np.asarray(Y_mm, dtype=float))

            i = j

        if not P_all:
            return EMPTY

        stations_mm = np.concatenate(stations_all, axis=0) if len(stations_all) > 1 else stations_all[0]
        points_mm   = np.concatenate(P_all,       axis=0) if len(P_all)       > 1 else P_all[0]
        local_X     = np.concatenate(X_all,       axis=0) if len(X_all)       > 1 else X_all[0]
        local_Y     = np.concatenate(Y_all,       axis=0) if len(Y_all)       > 1 else Y_all[0]

        return {
            "ids":        out_ids or [],
            "stations_mm": stations_mm,
            "points_mm":   points_mm,
            # NOTE: CrossSection returns (X_mm==local Y, Y_mm==local Z) by convention.
            "local_Y_mm":  local_X,
            "local_Z_mm":  local_Y,
            "loops_idx":   out_loops or [],
        }
