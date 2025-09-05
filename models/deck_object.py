from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import List

from .base import BaseObject


@dataclass(kw_only=True)
class DeckObject(BaseObject):
    no: str
    class_name: str
    type: str
    description: str
    name: str
    inactive: str
    cross_section_types: List[str]
    cross_section_names: List[str]
    grp_offset: List[float]
    placement_id: List[str]
    placement_description: List[str]
    ref_placement_id: List[str]
    ref_station_offset: List[float]
    station_value: List[float]
    cross_section_points_name: List[str]
    grp: List[str]
    cross_section_ncs: List[int]

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