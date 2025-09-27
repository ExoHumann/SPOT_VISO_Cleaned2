from __future__ import annotations
"""Enhanced minimal PierObject with JSON-driven NCS sequencing.

Features retained:
  * Single vertical extrusion built inside build()
  * Optional multi NCS switching driven by provided JSON offsets
  * Anchor via placement axis sampling or explicit world anchor
  * Meta diagnostics including ncs_steps & sequencing source
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
import math, numpy as np, warnings

from .linear_object import LinearObject
from .axis import Axis
from .cross_section import CrossSection

_DEFAULT_HEIGHT_M = 10.0

@dataclass
class PierObject(LinearObject):
    station_value: float = 0.0
    height_m: Optional[float] = None
    plan_rotation_add_deg: float = 0.0
    # Cross-section refs
    top_cross_section_ncs: int = 0
    bot_cross_section_ncs: int = 0
    internal_cross_section_ncs: List[int] = field(default_factory=list)
    internal_ref_station_offset: List[float] = field(default_factory=list)  # offsets for internal_cross_section_ncs
    top_cross_section_points_name: str = ""
    bot_cross_section_points_name: str = ""
    resolved_anchor_local_yz_mm: Optional[tuple] = None
    resolved_anchor_world_mm: Optional[tuple] = None
    placement_axis_obj: Axis | None = None
    selected_cross_section_ncs: Optional[int] = None
    _all_cross_sections_map: Optional[Dict[int, CrossSection]] = None
    # Compatibility fields accepted from loader (not actively used)
    top_yoffset: float = 0.0
    top_zoffset: float = 0.0
    bot_yoffset: float = 0.0
    bot_zoffset: float = 0.0
    bot_pier_elevation: float = 0.0

    def configure(self,
                  available_axes: Dict[str, Axis],
                  available_cross_sections: Dict[int, CrossSection],
                  available_mainstations: Dict[str, List],
                  axis_name: Optional[str] = None,
                  cross_section_ncs: Optional[List[int]] = None,
                  mainstation_name: Optional[str] = None) -> None:  # noqa: D401
        # Axis selection
        name = axis_name or getattr(self, 'axis_name', None)
        if name and name in available_axes:
            self.axis_obj = available_axes[name]
        if getattr(self, 'axis_obj', None) is not None:
            self.placement_axis_obj = self.axis_obj
        self._all_cross_sections_map = available_cross_sections or {}

        # Pick a base cross-section (top -> bot -> provided list -> any)
        if getattr(self, 'base_section', None) is None and available_cross_sections:
            candidate = None
            if self.top_cross_section_ncs and self.top_cross_section_ncs in available_cross_sections:
                candidate = available_cross_sections[self.top_cross_section_ncs]
                self.selected_cross_section_ncs = self.top_cross_section_ncs
            elif self.bot_cross_section_ncs and self.bot_cross_section_ncs in available_cross_sections:
                candidate = available_cross_sections[self.bot_cross_section_ncs]
                self.selected_cross_section_ncs = self.bot_cross_section_ncs
            elif cross_section_ncs:
                for n in cross_section_ncs:
                    if n in available_cross_sections:
                        candidate = available_cross_sections[n]
                        self.selected_cross_section_ncs = n
                        break
            if candidate is None:
                candidate = next(iter(available_cross_sections.values()))
                for k, v in available_cross_sections.items():
                    if v is candidate:
                        self.selected_cross_section_ncs = k; break
                warnings.warn(f"PierObject '{self.name or self.axis_name}': No matching cross-section found, using arbitrary NCS {self.selected_cross_section_ncs}")
            self.base_section = candidate

        # Ensure placement axis exists for anchor sampling
        if getattr(self, 'placement_axis_obj', None) is None and getattr(self, 'axis_obj', None) is None:
            raise RuntimeError('Placement axis missing during configure()')

    def build(self, *_, **kwargs):  # type: ignore[override]
        vertical_slices = int(kwargs.pop("vertical_slices", 6) or 6)
        twist_deg = float(kwargs.pop("twist_deg", 0.0) or 0.0)
        user_plan_offset_deg = float(kwargs.pop("plan_rotation_deg", 0.0) or 0.0)

        # Height
        h_m = float(self.height_m) if self.height_m is not None else _DEFAULT_HEIGHT_M
        if not math.isfinite(h_m) or h_m <= 0:
            h_m = _DEFAULT_HEIGHT_M
        height_mm = h_m * 1000.0

        # Anchor sampling
        if self.resolved_anchor_world_mm is not None:
            wx, wy, wz = map(float, self.resolved_anchor_world_mm)
            base_pos_mm = np.array([wx, wy, wz], float)
            anchor_point_source = 'world_provided'; anchor_found = True; yaw_deg = 0.0
        elif self.placement_axis_obj is not None:
            station_m = self.placement_axis_obj.clamp_station_m(float(self.station_value))
            s_mm = station_m * 1000.0
            P_exact, T_exact = self.placement_axis_obj._positions_and_tangents(np.array([s_mm], float))
            base_pos_mm = P_exact[0]
            tan_xy = np.array([T_exact[0][0], T_exact[0][1]], float)
            yaw_deg = 0.0 if np.linalg.norm(tan_xy) < 1e-9 else math.degrees(math.atan2(tan_xy[1], tan_xy[0]))
            anchor_point_source = 'axis_sample'; anchor_found = True
        else:
            base_pos_mm = np.zeros(3, float)
            yaw_deg = 0.0; anchor_point_source = 'origin'; anchor_found = False

        plan_rotation_deg = yaw_deg + user_plan_offset_deg + float(self.plan_rotation_add_deg or 0.0)
        anchor_name = ''

        # Precomputed sequencing only (created by loader). Map to section objects if available.
        steps = getattr(self, 'ncs_steps', None)
        if steps and self._all_cross_sections_map:
            tmp_map = {}
            for _, n in steps:
                try:
                    cs = self._all_cross_sections_map.get(int(n))
                except Exception:
                    cs = None
                if cs is not None:
                    tmp_map[int(n)] = cs
            if tmp_map:
                self.sections_by_ncs = tmp_map
        if self.base_section is None and self.sections_by_ncs:
            self.base_section = next(iter(self.sections_by_ncs.values()))

        # Vertical axis top-down
        stations_mm = [0.0, height_mm]
        x_coords = [base_pos_mm[0], base_pos_mm[0]]
        y_coords = [base_pos_mm[1], base_pos_mm[1]]
        z_coords = [base_pos_mm[2], base_pos_mm[2] - height_mm]
        vertical_axis = Axis(stations_mm, x_coords, y_coords, z_coords, units="mm")

        prev_axis = self.axis_obj
        self.axis_obj = vertical_axis
        try:
            stations_m = np.linspace(0.0, height_mm/1000.0, max(2, vertical_slices)).tolist()
            res = super().build(stations_m=stations_m,
                                 twist_deg=twist_deg,
                                 plan_rotation_deg=plan_rotation_deg)
            res['axis'] = vertical_axis
            res.setdefault('meta', {})
            res['meta'].update({
                'base_pos_mm': base_pos_mm.tolist(),
                'yaw_deg': yaw_deg,
                'anchor_mode': 'top_down',
                'anchor_point_name': anchor_name,
                'anchor_found': anchor_found,
                'anchor_point_source': anchor_point_source,
                'anchor_world_provided': bool(self.resolved_anchor_world_mm),
                'ncs_steps': getattr(self, 'ncs_steps', None),
                'selected_ncs': self.selected_cross_section_ncs
            })
            return res
        finally:
            self.axis_obj = prev_axis

    def set_world_anchor(self, world_point_mm: tuple | List[float] | np.ndarray):
        try:
            arr = list(world_point_mm)
            self.resolved_anchor_world_mm = (float(arr[0]), float(arr[1]), float(arr[2]))
        except Exception as e:
            raise ValueError("world_point_mm must be iterable with 3 numeric elements") from e
        return self

    def set_anchor_from_deck(self, deck_result: Dict, point_name: str, *, slice_index: int = 0):
        pts = deck_result.get('points_world_mm')
        ids = deck_result.get('ids')
        if pts is None or ids is None:
            return self
        try:
            ids_arr = np.asarray(ids)
            pts_arr = np.asarray(pts)
            mask = ids_arr == point_name
            if mask.any():
                first_idx = np.where(mask)[0][0]
                coord = pts_arr[slice_index, first_idx]
                self.resolved_anchor_world_mm = (float(coord[0]), float(coord[1]), float(coord[2]))
        except Exception:
            pass
        return self

__all__ = ["PierObject"]
