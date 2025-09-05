from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List

from models.base import BaseObject


@dataclass(kw_only=True)
class PierObject(BaseObject):
    no: str
    class_name: str
    type: str
    description: str
    name: str
    inactive: str
    object_axis_name: str
    cross_section_type: str
    cross_section_name: str
    axis_name: str
    ref_placement_id: str
    ref_station_offset: str
    station_value: float
    top_yoffset: float
    top_zoffset: float
    top_cross_section_ncs: int
    bot_yoffset: float
    bot_zoffset: float
    bot_cross_section_ncs: int
    bot_pier_elevation: float
    rotation_angle: float
    grp: int
    fixation: str
    internal_placement_id: List[str]
    internal_ref_placement_id: List[str]
    internal_ref_station_offset: List[float]
    internal_cross_section_ncs: List[int]
    grp_offset: List[float]
    axis_variables: List[Dict] = field(default_factory=list)

    top_cross_section_points_name: str = ""
    bot_cross_section_points_name: str = ""
    internal_station_value: List[float] = field(default_factory=list)
