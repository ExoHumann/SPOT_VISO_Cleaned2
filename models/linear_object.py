from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from models.main_station import MainStationRef
from models.axis import Axis
from models.cross_section import CrossSection  # adjust import to your layout

def interp_axis_variables(
    deck_axis_vars: List[Dict],
    section_defaults_mm: Dict[str, float],
    stations_m: List[float],
) -> Dict[str, np.ndarray]:
    """
    Per-station arrays from deck 'AxisVariables' rows.
    Raw values here (often meters/deg); CrossSection.evaluate fixes units.
    """
    S = len(stations_m)
    out: Dict[str, np.ndarray] = {}
    # defaults as constants (mm) — placeholders; unit fix happens later
    for k, v in (section_defaults_mm or {}).items():
        out[str(k)] = np.full(S, float(v or 0.0), dtype=float)
    for row in deck_axis_vars or []:
        name = str(row.get("VariableName") or "").strip()
        if not name: continue
        xs = [float(x) for x in (row.get("StationValue") or [])]      # meters
        ys = [float(y) for y in (row.get("VariableValues") or [])]    # raw
        if not xs or not ys: continue
        out[name] = np.interp(stations_m, xs, ys, left=ys[0], right=ys[-1])
    return out

@dataclass
class LinearObject:
    name: str
    axis: Axis
    base_section: CrossSection
    axis_variables_rows: List[Dict]
    mainstations: List[MainStationRef] = None         # optional
    sections_by_name: Dict[str, CrossSection] = None  # optional map for switching

    def _station_array(self, stations_m: Optional[List[float]], cap: Optional[int]) -> np.ndarray:
        if stations_m is None:
            stations_m = list(np.asarray(self.axis.stations, float) / 1000.0)
        if cap and len(stations_m) > cap > 0:
            idx = np.linspace(0, len(stations_m)-1, num=cap).round().astype(int)
            stations_m = [stations_m[i] for i in idx]
        return np.asarray(stations_m, float)

    def _per_station_controls(self, S_m: np.ndarray, twist_deg: float, extra_rot: float) -> tuple[np.ndarray, List[str]]:
        """Return per-station absolute rotation (deg) and section-name for each station."""
        rot = np.full(S_m.shape, float(twist_deg + extra_rot), float)
        sec = [self.base_section.name or "__BASE__"] * len(S_m)

        if not self.mainstations:
            return rot, sec

        # step-wise apply: for each MS, everything >= station_m uses its settings until next MS
        ms_sorted = sorted(self.mainstations, key=lambda m: m.station_m)
        j = 0
        for i, s in enumerate(S_m):
            # advance j while next MS <= current station
            while j + 1 < len(ms_sorted) and ms_sorted[j + 1].station_m <= s + 1e-12:
                j += 1
            ms = ms_sorted[j] if ms_sorted and ms_sorted[0].station_m <= s + 1e-12 else None
            if ms:
                # Convention: ms.rotation_deg is absolute plane rotation at that segment.
                rot[i] = float(ms.rotation_deg + extra_rot)
                if ms.cs_name and (self.sections_by_name and ms.cs_name in self.sections_by_name):
                    sec[i] = ms.cs_name
        return rot, sec

    def build(
        self,
        *,
        stations_m: Optional[List[float]] = None,
        twist_deg: float = 0.0,
        station_cap: Optional[int] = None,
        rotation_override_deg: Optional[float] = None,
    ) -> Dict:
        S_m = self._station_array(stations_m, station_cap)
        S_mm = S_m * 1000.0
        extra_rot = 0.0 if rotation_override_deg is None else float(rotation_override_deg)

        # variables once (vector arrays), later we slice per chunk
        defaults_mm = self.base_section.defaults_mm()
        var_arrays_all = interp_axis_variables(self.axis_variables_rows, defaults_mm, S_m.tolist())

        # decide rotation + section per station
        rot_deg, sec_for_station = self._per_station_controls(S_m, twist_deg, extra_rot)

        # chunk stations into runs of equal section name (to avoid reevaluating mixed sections)
        runs: List[tuple[int,int,str]] = []
        start = 0
        cur = sec_for_station[0]
        for i in range(1, len(sec_for_station)):
            if sec_for_station[i] != cur:
                runs.append((start, i, cur))
                start = i
                cur = sec_for_station[i]
        runs.append((start, len(sec_for_station), cur))

        all_ids = None
        chunks_P = []
        chunks_X = []
        chunks_Y = []
        for a, b, sec_name in runs:
            section = (self.sections_by_name or {}).get(sec_name, self.base_section)

            # slice per-run arrays
            S_chunk_mm = S_mm[a:b]
            rot_chunk  = rot_deg[a:b]
            # build per-run var arrays (same keys, sliced)
            var_arrays = {k: np.asarray(v, float)[a:b] for k, v in var_arrays_all.items()}

            ids, X_mm, Y_mm, loops_idx = section.evaluate(var_arrays, negate_x=True)
            if all_ids is None: 
                all_ids = ids  # require consistent id order across sections

            yz = np.dstack([X_mm, Y_mm])  # (len(run), N, 2)
            P_mm = self.axis.embed_section_points_world(
                S_chunk_mm, yz_points_mm=yz, x_offsets_mm=None, rotation_deg=rot_chunk
            )
            chunks_P.append(P_mm)
            chunks_X.append(X_mm)
            chunks_Y.append(Y_mm)

        P_out = np.vstack(chunks_P) if chunks_P else np.zeros((0,0,3))
        X_out = np.vstack(chunks_X) if chunks_X else np.zeros((0,0))
        Y_out = np.vstack(chunks_Y) if chunks_Y else np.zeros((0,0))
        return {
            "name": self.name,
            "stations_mm": S_mm,
            "ids": all_ids or [],
            "local_Y_mm": X_out,
            "local_Z_mm": Y_out,
            "points_world_mm": P_out,
            # Loops: you can keep the base section’s loops for plotting filters
            "loops_idx": self.base_section.last_loops_idx if hasattr(self.base_section, "last_loops_idx") else [],
            "axis": self.axis,
            "section_json": self.base_section.data,  # not perfect if switching, but good for embed metadata
        }