from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from .base import BaseObject
from .axis import Axis
from .cross_section import CrossSection
from .main_station import MainStationRef  # your type

# Generic helper (previously deck-specific) kept for potential external callers.
def interp_axis_variables(axis_var_rows: List[Dict],
                          section_defaults_mm: Dict[str, float],
                          stations_m: List[float]) -> Dict[str, np.ndarray]:
    S = len(stations_m)
    out: Dict[str, np.ndarray] = {}
    for k, v in (section_defaults_mm or {}).items():
        out[str(k)] = np.full(S, float(v or 0.0), dtype=float)
    for row in axis_var_rows or []:
        name = str(row.get("VariableName") or "").strip()
        if not name:
            continue
        xs = [float(x) for x in (row.get("StationValue") or [])]
        ys = [float(y) for y in (row.get("VariableValues") or [])]
        if not xs or not ys:
            continue
        out[name] = np.interp(stations_m, xs, ys, left=ys[0], right=ys[-1])
    return out

@dataclass
class LinearObject(BaseObject):
    """
    Base class for linear structural objects (Deck, Pier, Foundation).
    Combines common object functionality with linear geometry computation.
    """

    # Cross-section management
    base_section: Optional[CrossSection] = None
    ncs_steps: Optional[List[Tuple[float, int]]] = None        # [(station_m, ncs), ...] sorted by station
    sections_by_ncs: Optional[Dict[int, CrossSection]] = None  # ncs -> CrossSection

    # Main stations for rotations
    mainstations: Optional[List[MainStationRef]] = None

    def configure(self, 
                  available_axes: Dict[str, Axis],
                  available_cross_sections: Dict[int, CrossSection], 
                  available_mainstations: Dict[str, List[MainStationRef]],
                  axis_name: Optional[str] = None,
                  cross_section_ncs: Optional[List[int]] = None,
                  mainstation_name: Optional[str] = None) -> None:
        """
        Configure this LinearObject with components from available parsed data.
        Subclasses can override this for specific configuration logic.
        
        Args:
            available_axes: Dict of axis_name -> Axis object
            available_cross_sections: Dict of ncs -> CrossSection object  
            available_mainstations: Dict of axis_name -> List[MainStationRef]
            axis_name: Name of axis to use (defaults to self.axis_name)
            cross_section_ncs: List of NCS values for cross sections
            mainstation_name: Name of mainstations to use (defaults to axis_name)
        """
        axis_name = axis_name or self.axis_name
        mainstation_name = mainstation_name or axis_name
        
        # Set axis
        if axis_name in available_axes:
            self.axis_obj = available_axes[axis_name]
        
        # Set main stations
        if mainstation_name in available_mainstations:
            self.mainstations = available_mainstations[mainstation_name]
        
        # Set cross sections (generic implementation - subclasses may override)
        if cross_section_ncs:
            self.sections_by_ncs = {}
            for ncs in cross_section_ncs:
                if ncs in available_cross_sections:
                    self.sections_by_ncs[ncs] = available_cross_sections[ncs]
            
            # Set base section to first available
            if self.sections_by_ncs:
                first_ncs = min(self.sections_by_ncs.keys())
                self.base_section = self.sections_by_ncs[first_ncs]

    # ---------- helpers ----------
    @staticmethod
    def _round_key(x: float, nd: int = 9) -> float:
        return round(float(x), nd)

    @staticmethod
    def _unique_sorted_union(*seqs: List[float]) -> np.ndarray:
        seen = set()
        vals: List[float] = []
        for seq in seqs:
            if not seq:
                continue
            for v in seq:
                k = LinearObject._round_key(v)
                if k not in seen:
                    seen.add(k)
                    vals.append(float(v))
        vals.sort()
        return np.asarray(vals, float)

    @staticmethod
    def _downsample_preserving(S_m: np.ndarray, must_keep: set, cap: int) -> np.ndarray:
        if not cap or len(S_m) <= cap:
            return S_m
        keep_idx = [i for i, s in enumerate(S_m) if LinearObject._round_key(s) in must_keep]
        keep_idx = sorted(set(keep_idx))
        remaining = cap - len(keep_idx)
        if remaining <= 0:
            picks = np.linspace(0, len(keep_idx) - 1, num=max(cap, 1)).round().astype(int)
            idx = sorted({keep_idx[i] for i in picks})
            return S_m[idx]
        avail = [i for i in range(len(S_m)) if i not in keep_idx]
        if not avail:
            return S_m[keep_idx]
        picks = np.linspace(0, len(avail) - 1, num=remaining).round().astype(int)
        idx = sorted(keep_idx + [avail[i] for i in picks])
        return S_m[idx]

    def _collect_critical_stations(self) -> List[float]:
        crit: List[float] = []
        # AxisVariables knots
        for row in (self.axis_variables or []):
            for s in (row.get("StationValue") or []):
                try:
                    crit.append(float(s))
                except Exception:
                    pass
        # NCS switch stations
        for tup in (self.ncs_steps or []):
            try:
                crit.append(float(tup[0]))
            except Exception:
                pass
        # MainStations
        for ms in (self.mainstations or []):
            try:
                crit.append(float(ms.station_m))
            except Exception:
                pass
        # Axis endpoints (meters)
        if self.axis_obj:
            S_axis_m = (
                np.asarray(getattr(self.axis_obj, "stations", []), float) / 1000.0
                if getattr(self.axis_obj, "stations", None) is not None
                else np.array([])
            )
            if S_axis_m.size:
                crit.append(float(S_axis_m[0]))
                crit.append(float(S_axis_m[-1]))
        return self._unique_sorted_union(crit).tolist()
    
    def _axis_variable_knots(self) -> list[float]:
        seen = set(); out = []
        for row in (self.axis_variables or []):
            for s in (row.get("StationValue") or []):
                try:
                    k = self._round_key(float(s), 6)
                    if k not in seen:
                        seen.add(k); out.append(float(s))
                except Exception:
                    pass
        return sorted(out)


    # ---------- existing utilities ----------
    def _station_array(self, stations_m: Optional[List[float]], cap: Optional[int]) -> np.ndarray:
        if stations_m is None:
            if self.axis_obj:
                stations_m = list(np.asarray(self.axis_obj.stations, float) / 1000.0)
            else:
                stations_m = []
        if cap and len(stations_m) > cap > 0:
            idx = np.linspace(0, len(stations_m) - 1, num=cap).round().astype(int)
            stations_m = [stations_m[i] for i in idx]
        return np.asarray(stations_m, float)

    def _rotations_from_mainstations(self, S_m: np.ndarray, base_twist: float, extra_rot: float) -> np.ndarray:
        rot = np.full(S_m.shape, float(base_twist + extra_rot), float)
        if not self.mainstations:
            return rot
        ms_by_s = {round(float(ms.station_m), 9): float(ms.rotation_deg or 0.0) for ms in self.mainstations}
        for i, s in enumerate(S_m):
            key = round(float(s), 9)
            if key in ms_by_s:
                rot[i] = ms_by_s[key] + extra_rot
        return rot

    def _active_ncs_per_station(self, S_m: np.ndarray) -> List[Optional[int]]:
        if not self.ncs_steps:
            return [None] * len(S_m)
        starts = np.array([float(t[0]) for t in self.ncs_steps], float)
        ncss   = [int(t[1]) for t in self.ncs_steps]
        out: List[Optional[int]] = []
        for s in S_m:
            idx = int(np.searchsorted(starts, s, side='right') - 1)
            out.append(ncss[idx] if idx >= 0 else None)
        return out

    def _build_canonical_ids(self, var_arrays_all: Dict[str, np.ndarray]) -> List[str]:
        if not self.base_section:
            return []
        # probe with a single-station slice (cheap, consistent)
        take1 = {k: np.asarray(v, float)[:1] for k, v in var_arrays_all.items()}
        base_ids, *_ = self.base_section.evaluate(take1, negate_x=True)
        canon: List[str] = list(base_ids or [])
        if self.sections_by_ncs:
            for sec in self.sections_by_ncs.values():
                ids, *_ = sec.evaluate(take1, negate_x=True)
                for pid in ids or []:
                    if pid not in canon:
                        canon.append(pid)
        return canon

    @staticmethod
    def _align_to_ids(ids: List[str], X: np.ndarray, Y: np.ndarray, canonical_ids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        col = {pid: j for j, pid in enumerate(ids or [])}
        S = X.shape[0]
        N = len(canonical_ids)
        X2 = np.full((S, N), np.nan, float)
        Y2 = np.full((S, N), np.nan, float)
        for j, pid in enumerate(canonical_ids):
            k = col.get(pid)
            if k is not None:
                X2[:, j] = X[:, k]
                Y2[:, j] = Y[:, k]
        return X2, Y2

    def _ncs_at(self, s: float) -> Optional[int]:
        if not self.ncs_steps:
            return None
        starts = np.array([float(t[0]) for t in self.ncs_steps], float)
        ncss   = [int(t[1]) for t in self.ncs_steps]
        idx = int(np.searchsorted(starts, s, side='right') - 1)
        return ncss[idx] if idx >= 0 else None




    # ---------- main ----------
    def build(
        self,
        stations_m: Optional[List[float]] = None,
        twist_deg: float = 0.0,
        station_cap: Optional[int] = None,
        plan_rotation_deg: float = 0.0,
        frame_mode: str = "symmetric",  # "symmetric" or "pt"
        rotation_mode: str = "additive",  # "additive" or "absolute"
    ) -> Dict:
        # 1) unified station set (axis samples + object critical points)
        if not self.axis_obj:
            return {"name": self.name, "stations_mm": np.array([], float), "ids": [], "local_Y_mm": np.zeros((0, 0), float), "local_Z_mm": np.zeros((0, 0), float), "points_world_mm": np.zeros((0, 0, 3), float), "loops_idx": [], "axis": None, "section_json": None, "overlays": []}
        
        S_axis_all_m = list(np.asarray(self.axis_obj.stations, float) / 1000.0)
        S_axis_m = np.asarray(self.axis_obj.stations, float) / 1000.0
        ax_min = float(S_axis_m.min()) if S_axis_m.size else -np.inf
        ax_max = float(S_axis_m.max()) if S_axis_m.size else  np.inf

        base_samples_m = [float(s) for s in (stations_m or S_axis_all_m)]
        crit_m = self._collect_critical_stations()
        S0_m = self._unique_sorted_union(base_samples_m, crit_m)

        must_keep_keys = {self._round_key(s) for s in crit_m}
        S_m_all = self._downsample_preserving(S0_m, must_keep_keys, station_cap or 0)

        # Clamp to deck extents (prefer MainStations if present, else NCS span, else axis range)
        s_min = float(S_m_all[0]) if len(S_m_all) else 0.0
        s_max = float(S_m_all[-1]) if len(S_m_all) else 0.0
        s_min = max(s_min, ax_min)
        s_max = min(s_max, ax_max)

        if self.mainstations:
            starts = [ms.station_m for ms in self.mainstations if str(ms.placement_id or "").lower() in ("str","start","a")]
            ends   = [ms.station_m for ms in self.mainstations if str(ms.placement_id or "").lower() in ("end","e")]
            if starts: s_min = float(min(starts))
            else:      s_min = float(min(ms.station_m for ms in self.mainstations))
            if ends:   s_max = float(max(ends))
            else:      s_max = float(max(ms.station_m for ms in self.mainstations))
        elif self.ncs_steps:
            s_min = float(self.ncs_steps[0][0])
            s_max = max(float(self.ncs_steps[-1][0]), float(self.axis_obj.stations[-1] / 1000.0))
            # s_min = float(self.ncs_steps[0][0])
            # s_max = max(float(self.ncs_steps[-1][0]), float(self.axis.stations[-1] / 1000.0))

        s_min = max(s_min, ax_min)
        s_max = min(s_max, ax_max)
        if s_max < s_min:   # safety
            s_min, s_max = s_max, s_min

        mask = (S_m_all >= s_min - 1e-9) & (S_m_all <= s_max + 1e-9)
        S_m  = S_m_all[mask]
        S_mm = S_m * 1000.0

        theta0_deg  = float(twist_deg)

        # 2) evaluate AxisVariables at S_m (with robust fallback)
        var_arrays_all: Dict[str, np.ndarray] = {}
        try:
            from models.axis_variable import AxisVariable
            av_objs = AxisVariable.create_axis_variables(self.axis_variables or [])
            rows = AxisVariable.evaluate_at_stations_cached(av_objs, S_m.tolist())
            if rows:
                names = set().union(*[r.keys() for r in rows])
                for nm in names:
                    var_arrays_all[nm] = np.asarray(
                        [float(rows[i].get(nm, np.nan)) for i in range(len(S_m))], float
                    )
        except Exception:
            # Fallback: use base section defaults + simple linear interpolation
            # over raw axis_variables (generic for any LinearObject subclass).
            defaults_mm = self.base_section.defaults_mm() if self.base_section else {}
            for k, v in (defaults_mm or {}).items():
                var_arrays_all[str(k)] = np.full(len(S_m), float(v or 0.0), dtype=float)
            raw_rows = getattr(self, 'axis_variables', None) or []
            for row in raw_rows:
                name = str(row.get('VariableName') or '').strip()
                if not name:
                    continue
                xs = [float(x) for x in (row.get('StationValue') or [])]
                ys = [float(y) for y in (row.get('VariableValues') or [])]
                if not xs or not ys:
                    continue
                var_arrays_all[name] = np.interp(S_m, xs, ys, left=ys[0], right=ys[-1])

        # 3) choose NCS per station and split into runs
        ncs_for_station = [self._ncs_at(s) for s in S_m]
        runs: List[Tuple[int,int,Optional[int]]] = []
        if len(S_m):
            a = 0; cur = ncs_for_station[0]
            for i in range(1, len(S_m)):
                if ncs_for_station[i] != cur:
                    runs.append((a, i, cur))
                    a, cur = i, ncs_for_station[i]
            runs.append((a, len(S_m), cur))

        # 4) canonical IDs across all sections
        canonical_ids = self._build_canonical_ids(var_arrays_all)

        # base rotation helper
        def _apply_base_rot(X_mm, Y_mm, deg):
            if abs(deg) < 1e-12:
                return X_mm, Y_mm
            th = np.deg2rad(deg)
            c, s = np.cos(th), np.sin(th)
            Xr = X_mm * c - Y_mm * s
            Yr = X_mm * s + Y_mm * c
            return Xr, Yr

        # 5) build geometry per run
        chunks_P: List[np.ndarray] = []; chunks_X: List[np.ndarray] = []; chunks_Y: List[np.ndarray] = []
        used_sections: List[CrossSection] = []

        # Build per-station rotation arrays (twist around tangent, plan yaw about global Z)
        twist_array = np.full(len(S_m), theta0_deg, float)
        plan_array  = np.full(len(S_m), float(plan_rotation_deg), float)
        if self.mainstations:
            for i, s in enumerate(S_m):
                key = round(float(s), 9)
                for ms in self.mainstations:
                    if round(float(ms.station_m), 9) == key:
                        ms_tw = float(getattr(ms, 'station_rotation_x_deg', getattr(ms, 'rotation_deg', 0.0)) or 0.0)
                        ms_yaw = float(getattr(ms, 'station_rotation_z_deg', 0.0) or 0.0)
                        if rotation_mode.lower() == 'absolute':
                            twist_array[i] = theta0_deg + ms_tw  # base + absolute override
                            plan_array[i]  = float(plan_rotation_deg) + ms_yaw
                        else:  # additive
                            twist_array[i] += ms_tw
                            plan_array[i]  += ms_yaw
                        break

        for a, b, ncs in runs:
            section = self.base_section
            if ncs is not None and self.sections_by_ncs and ncs in self.sections_by_ncs:
                section = self.sections_by_ncs[ncs]
            if section is None:
                continue
            used_sections.append(section)

            var_arrays = {k: np.asarray(v, float)[a:b] for k, v in var_arrays_all.items()}
            # If there are NO axis variables at all (common for simplified PierObject),
            # section.evaluate() would otherwise fall back to S=1 (single station) which
            # breaks embedding when we actually have len(run) stations. Provide a dummy
            # placeholder array so evaluate sees the correct station count and then
            # replaces needed variables with defaults at that length.
            if not var_arrays:
                var_arrays = {"__dummy__": np.zeros(b - a, float)}
            ids, X_mm, Y_mm, loops_idx = section.evaluate(var_arrays, negate_x=True)
            X_mm, Y_mm = self._align_to_ids(ids, X_mm, Y_mm, canonical_ids)
            # Remove pre-rotation - let embed_section_points_world handle rotation
            # X_mm, Y_mm = _apply_base_rot(X_mm, Y_mm, theta0_deg)

            yz = np.dstack([X_mm, Y_mm])  # (len(run), N, 2)
            S_chunk_mm = S_mm[a:b]

            if self.axis_obj:
                # slice rotation arrays for this chunk
                rot_slice  = twist_array[a:b]
                plan_slice = plan_array[a:b]
                if frame_mode.lower() == 'symmetric' and hasattr(self.axis_obj, 'embed_section_points_world_symmetric'):
                    P_mm = self.axis_obj.embed_section_points_world_symmetric(
                        S_chunk_mm, yz_points_mm=yz, x_offsets_mm=None,
                        rotation_deg=rot_slice, plan_rotation_deg=plan_slice
                    )
                else:
                    P_mm = self.axis_obj.embed_section_points_world(
                        S_chunk_mm, yz_points_mm=yz, x_offsets_mm=None, rotation_deg=rot_slice, plan_rotation_deg=plan_slice
                    )
            else:
                P_mm = np.zeros((len(S_chunk_mm), len(ids), 3), float)

            chunks_P.append(P_mm); chunks_X.append(X_mm); chunks_Y.append(Y_mm)


        P_out = np.vstack(chunks_P) if chunks_P else np.zeros((0, 0, 3))
        X_out = np.vstack(chunks_X) if chunks_X else np.zeros((0, 0))
        Y_out = np.vstack(chunks_Y) if chunks_Y else np.zeros((0, 0))

        # 6) overlays at MainStations
        overlays: List[Dict] = []
        if self.mainstations and len(S_mm) > 0:
            for ms in self.mainstations:
                if not (s_min - 1e-9 <= ms.station_m <= s_max + 1e-9):
                    continue
                idx = int(np.argmin(np.abs(S_m - float(ms.station_m))))
                n_here = ncs_for_station[idx]
                section = self.sections_by_ncs.get(n_here, self.base_section) if (self.sections_by_ncs and n_here is not None) else self.base_section
                if section is None:
                    # Without a section we cannot generate overlay geometry for this MainStation
                    continue

                var_one = {k: np.asarray(v, float)[idx:idx+1] for k, v in var_arrays_all.items()}
                ids1, X1_mm, Y1_mm, loops1 = section.evaluate(var_one, negate_x=True)

                Nst, Mpt = len(S_mm), X1_mm.shape[1]
                yz_big = np.full((Nst, Mpt, 2), np.nan, float)
                yz_big[idx, :, 0] = X1_mm[0]; yz_big[idx, :, 1] = Y1_mm[0]

                # Build per-station arrays for overlay (reuse base arrays then add ms-specific)
                rot_overlay  = twist_array.copy()
                plan_overlay = plan_array.copy()
                idxs = np.where(np.isclose(S_m, float(ms.station_m)))[0]
                if idxs.size:
                    rot_overlay[idxs[0]] += float(ms.rotation_deg or 0.0)
                if frame_mode.lower() == 'symmetric' and hasattr(self.axis_obj, 'embed_section_points_world_symmetric'):
                    P_all_mm = self.axis_obj.embed_section_points_world_symmetric(
                        S_mm, yz_big, x_offsets_mm=None, rotation_deg=rot_overlay, plan_rotation_deg=plan_overlay
                    )
                else:
                    P_all_mm = self.axis_obj.embed_section_points_world(
                        S_mm, yz_big, x_offsets_mm=None, rotation_deg=rot_overlay, plan_rotation_deg=plan_overlay
                    )
                overlays.append({"kind":"ms","idx":idx,"station_m":float(S_m[idx]),"ids":ids1,"loops_idx":loops1,"P_mm":P_all_mm[idx]})

        # 7) overlays at NCS switches (two sections at the same s)
        if self.ncs_steps:
            
            switch_stations = [float(s) for s,_ in self.ncs_steps[1:]]
            for s in switch_stations:
                if not (s_min - 1e-9 <= s <= s_max + 1e-9):
                    continue
                idx = int(np.argmin(np.abs(S_m - s)))
                prev_ncs = self._ncs_at(s - 1e-9)
                next_ncs = self._ncs_at(s + 1e-9)
                for ncs in [prev_ncs, next_ncs]:
                    section = self.sections_by_ncs.get(ncs, self.base_section) if (self.sections_by_ncs and ncs is not None) else self.base_section
                    if section is None:
                        continue
                    var_one = {k: np.asarray(v, float)[idx:idx+1] for k, v in var_arrays_all.items()}
                    ids1, X1_mm, Y1_mm, loops1 = section.evaluate(var_one, negate_x=True)

                    Nst, Mpt = len(S_mm), X1_mm.shape[1]
                    yz_big = np.full((Nst, Mpt, 2), np.nan, float)
                    yz_big[idx, :, 0] = X1_mm[0]; yz_big[idx, :, 1] = Y1_mm[0]
                    if self.axis_obj:
                        rot_overlay  = twist_array.copy()
                        plan_overlay = plan_array.copy()
                        if frame_mode.lower() == 'symmetric' and hasattr(self.axis_obj, 'embed_section_points_world_symmetric'):
                            P_all_mm = self.axis_obj.embed_section_points_world_symmetric(
                                S_mm, yz_big, x_offsets_mm=None,
                                rotation_deg=rot_overlay, plan_rotation_deg=plan_overlay
                            )
                        else:
                            P_all_mm = self.axis_obj.embed_section_points_world(
                                S_mm, yz_big, x_offsets_mm=None,
                                rotation_deg=rot_overlay, plan_rotation_deg=plan_overlay
                            )
                    else:
                        P_all_mm = np.zeros((Nst, Mpt, 3), float)
                    overlays.append({"kind":"switch","idx":idx,"station_m":float(S_m[idx]),"ids":ids1,"loops_idx":loops1,"P_mm":P_all_mm[idx]})
        
        # 7b) overlays at AxisVariable knots (highlight these)
        for s in self._axis_variable_knots():
            if not (s_min - 1e-9 <= s <= s_max + 1e-9): 
                continue
            if not (ax_min - 1e-9 <= s <= ax_max + 1e-9): 
                continue
            idx = int(np.argmin(np.abs(S_m - float(s))))
            n_here = ncs_for_station[idx]
            section = self.sections_by_ncs.get(n_here, self.base_section) if (self.sections_by_ncs and n_here is not None) else self.base_section
            if section is None:
                continue

            var_one = {k: np.asarray(v, float)[idx:idx+1] for k, v in var_arrays_all.items()}
            ids1, X1_mm, Y1_mm, loops1 = section.evaluate(var_one, negate_x=True)

            Nst, Mpt = len(S_mm), X1_mm.shape[1]
            yz_big = np.full((Nst, Mpt, 2), np.nan, float)
            yz_big[idx, :, 0] = X1_mm[0]; yz_big[idx, :, 1] = Y1_mm[0]
            if self.axis_obj:
                rot_overlay  = twist_array.copy()
                plan_overlay = plan_array.copy()
                if frame_mode.lower() == 'symmetric' and hasattr(self.axis_obj, 'embed_section_points_world_symmetric'):
                    P_all_mm = self.axis_obj.embed_section_points_world_symmetric(
                        S_mm, yz_big, x_offsets_mm=None,
                        rotation_deg=rot_overlay, plan_rotation_deg=plan_overlay
                    )
                else:
                    P_all_mm = self.axis_obj.embed_section_points_world(
                        S_mm, yz_big, x_offsets_mm=None,
                        rotation_deg=rot_overlay, plan_rotation_deg=plan_overlay
                    )
            else:
                P_all_mm = np.zeros((Nst, Mpt, 3), float)

            # add a color you want for AV changes (plotter will use it)
            overlays.append({
                "kind": "var",
                "idx": idx,
                "station_m": float(S_m[idx]),
                "ids": ids1,
                "loops_idx": loops1,
                "P_mm": P_all_mm[idx],
                "color": "orange",
            })


        # 8) merge loop indices across all used sections (mapped to canonical ids)
        merged_loops_idx: List[np.ndarray] = []
        canon_map = {pid: j for j, pid in enumerate(canonical_ids)}
        def _append_section_loops(sec: CrossSection):
            if sec is None:
                return
            take1 = {k: np.asarray(v, float)[:1] for k, v in var_arrays_all.items()}
            ids1, _, _, loops1 = sec.evaluate(take1, negate_x=True)
            for arr in loops1 or []:
                idxs = [canon_map.get(ids1[i]) for i in arr if canon_map.get(ids1[i]) is not None]
                if len(idxs) >= 2:
                    merged_loops_idx.append(np.asarray(idxs, dtype=int))

        if self.base_section:
            _append_section_loops(self.base_section)
        if self.sections_by_ncs:
            for sec in self.sections_by_ncs.values():
                _append_section_loops(sec)

        # # de-dupe overlays by (kind, station)
        # seen = set(); dedup = []
        # for ov in overlays:
        #     key = (ov.get("kind"), self._round_key(float(ov.get("station_m", np.nan)), 6))
        #     if key in seen:
        #         continue
        #     seen.add(key); dedup.append(ov)
        # overlays = dedup


        return {
            "name": self.name,
            "stations_mm": S_mm,
            "ids": canonical_ids,
            "local_Y_mm": X_out,
            "local_Z_mm": Y_out,
            "points_world_mm": P_out,
            "loops_idx": merged_loops_idx,
            "axis": self.axis_obj,
            "section_json": self.base_section.data if self.base_section else None,
            "overlays": overlays,
            # NEW: expose per-station rotation arrays for debug/overlay usage
            "twist_deg_array": twist_array,
            "plan_rotation_deg_array": plan_array,
        }
