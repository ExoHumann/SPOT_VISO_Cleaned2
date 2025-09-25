from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np

from .linear_object import LinearObject
from .axis import Axis
from .cross_section import CrossSection
from .axis_variable import AxisVariable

@dataclass
class FoundationObject(LinearObject):
    no: str = ""
    class_name: str = ""
    type: str = ""
    description: str = ""
    name: str = ""
    inactive: str = ""
    object_axis_name: str = ""
    cross_section_type: str = ""
    cross_section_name: str = ""
    ref_placement_id: str = ""
    ref_station_offset: float = 0.0
    station_value: float = 0.0
    cross_section_points_name: str = ""
    foundation_ref_point_y_offset: float = 0.0
    foundation_ref_point_x_offset: float = 0.0
    foundation_level: float = 0.0
    rotation_angle: float = 0.0
    axis_name: str = ""
    pier_object_name: List[str] = field(default_factory=list)
    point1: str = ""
    point2: str = ""
    point3: str = ""
    point4: str = ""
    thickness: str = ""
    grp: str = ""
    cross_section_ncs2: int = 0
    top_z_offset: float = 0.0
    bot_z_offset: float = 0.0
    top_x_offset: float = 0.0
    top_y_offset: float = 0.0
    pile_dir_angle: float = 0.0
    pile_slope: float = 0.0
    kx: float = 0.0
    ky: float = 0.0
    kz: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0
    fixation: str = ""
    eval_pier_object_name: str = ""
    eval_station_value: float = 0.0
    eval_bot_cross_section_points_name: str = ""
    eval_bot_y_offset: float = 0.0
    eval_bot_z_offset: float = 0.0
    eval_bot_pier_elevation: float = 0.0
    internal_placement_id: List[str] = field(default_factory=list)
    internal_ref_placement_id: List[float] = field(default_factory=list)
    internal_ref_station_offset: List[float] = field(default_factory=list)
    internal_station_value: List[float] = field(default_factory=list)
    internal_cross_section_ncs: List[float] = field(default_factory=list)
    grp_offset: List[float] = field(default_factory=list)
    axis_variables: List[Dict] = field(default_factory=list)


    def compute_geometry(
        self,
        *,
        ctx,
        stations_m: Optional[List[float]] = None,
        twist_deg: float = 0.0,
        negate_x: bool = True,
    ) -> Dict[str, object]:
        axis: Optional[Axis] = getattr(self, "axis_obj", None)
        if axis is None:
            return {"ids": [], "stations_mm": np.array([], float), "points_mm": np.zeros((0, 0, 3)), "local_Y_mm": np.zeros((0, 0)), "local_Z_mm": np.zeros((0, 0)), "loops_idx": []}

        if stations_m is None:
            S = np.asarray(getattr(axis, "stations", []), float)
            stations_m = (S / 1000.0).tolist()
        stations_mm = np.asarray(stations_m, float) * 1000.0

        axis_results = AxisVariable.evaluate_at_stations_cached(self.axis_variables_obj or [], stations_m)

        section: Optional[CrossSection] = None
        if hasattr(ctx, "crosssec_by_name") and self.cross_section_name:
            section = ctx.crosssec_by_name.get(self.cross_section_name)
        if section is None and hasattr(ctx, "crosssec_by_ncs"):
            ncs_list = getattr(self, "cross_section_ncs", []) or []
            if ncs_list:
                section = ctx.crosssec_by_ncs.get(int(ncs_list[0]))
        if section is None:
            return {"ids": [], "stations_mm": stations_mm, "points_mm": np.zeros((0, 0, 3)), "local_Y_mm": np.zeros((0, 0)), "local_Z_mm": np.zeros((0, 0)), "loops_idx": []}

        ids, S_mm, P_mm, X_mm, Y_mm, loops_idx = section.compute_embedded_points(
            axis=axis,
            axis_var_results=axis_results,
            stations_m=stations_m,
            twist_deg=float(twist_deg or 0.0),
            negate_x=negate_x,
        )
        return {
            "ids": ids,
            "stations_mm": S_mm,
            "points_mm": P_mm,
            "local_Y_mm": X_mm,
            "local_Z_mm": Y_mm,
            "loops_idx": loops_idx,
        }

    def configure(self, 
                  available_axes: Dict[str, Axis],
                  available_cross_sections: Dict[int, CrossSection], 
                  available_mainstations: Dict[str, List[MainStationRef]],
                  axis_name: Optional[str] = None,
                  cross_section_ncs: Optional[List[int]] = None,
                  mainstation_name: Optional[str] = None) -> None:
        """
        Configure FoundationObject with available components.
        Uses cross_section_ncs2 from foundation data.
        """
        from .main_station import MainStationRef  # Import here to avoid circular imports
        
        axis_name = axis_name or self.axis_name or getattr(self, 'object_axis_name', '')
        mainstation_name = mainstation_name or axis_name
        
        # For foundation, use cross_section_ncs2
        foundation_ncs = []
        if hasattr(self, 'cross_section_ncs2') and self.cross_section_ncs2:
            foundation_ncs.append(self.cross_section_ncs2)
        
        cross_section_ncs = cross_section_ncs or foundation_ncs
        
        # Call parent configure
        super().configure(available_axes, available_cross_sections, available_mainstations,
                         axis_name, cross_section_ncs, mainstation_name)
        
        # For foundation, create a single station step at the foundation location
        if hasattr(self, 'station_value') and isinstance(self.station_value, (int, float)):
            ncs = cross_section_ncs[0] if cross_section_ncs else 1
            self.ncs_steps = [(float(self.station_value), ncs)]