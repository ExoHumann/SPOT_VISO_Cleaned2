from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List

from .base import BaseObject


@dataclass(kw_only=True)
class FoundationObject(BaseObject):
    no: str
    class_name: str
    type: str
    description: str
    name: str
    inactive: str
    object_axis_name: str
    cross_section_type: str
    cross_section_name: str
    ref_placement_id: str
    ref_station_offset: float
    station_value: float
    cross_section_points_name: str
    foundation_ref_point_y_offset: float
    foundation_ref_point_x_offset: float
    foundation_level: float
    rotation_angle: float
    axis_name: str
    pier_object_name: List[str]
    point1: str
    point2: str
    point3: str
    point4: str
    thickness: str
    grp: str
    cross_section_ncs2: int
    top_z_offset: float
    bot_z_offset: float
    top_x_offset: float
    top_y_offset: float
    pile_dir_angle: float
    pile_slope: float
    kx: float
    ky: float
    kz: float
    rx: float
    ry: float
    rz: float
    fixation: str
    eval_pier_object_name: str
    eval_station_value: float
    eval_bot_cross_section_points_name: str
    eval_bot_y_offset: float
    eval_bot_z_offset: float
    eval_bot_pier_elevation: float
    internal_placement_id: List[str]
    internal_ref_placement_id: List[float]
    internal_ref_station_offset: List[float]
    internal_station_value: List[float]
    internal_cross_section_ncs: List[float]
    grp_offset: List[float]
    axis_variables: List[Dict] = field(default_factory=list)
