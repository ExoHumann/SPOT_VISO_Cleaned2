from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional
import numpy as np
from .linear_object import LinearObject
from .axis import Axis
from .cross_section import CrossSection
from .axis_variable import AxisVariable
from .main_station import MainStationRef
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


@dataclass
class DeckObject(LinearObject):
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

    def configure(self, 
                  available_axes: Dict[str, Axis],
                  available_cross_sections: Dict[int, CrossSection], 
                  available_mainstations: Dict[str, List[MainStationRef]],
                  axis_name: Optional[str] = None,
                  cross_section_ncs: Optional[List[int]] = None,
                  mainstation_name: Optional[str] = None) -> None:
        """
        Configure DeckObject with available components.
        Uses cross_section_ncs from deck data, or provided override.
        """
        axis_name = axis_name or self.axis_name
        mainstation_name = mainstation_name or axis_name
        cross_section_ncs = cross_section_ncs or getattr(self, 'cross_section_ncs', [])
        
        # Call parent configure with deck-specific cross section NCS
        super().configure(available_axes, available_cross_sections, available_mainstations,
                         axis_name, cross_section_ncs, mainstation_name)
        
        # Set up NCS steps for deck (station -> NCS mapping)
        if hasattr(self, 'station_value') and hasattr(self, 'cross_section_ncs') and self.station_value and self.cross_section_ncs:
            self.ncs_steps = list(zip(self.station_value, self.cross_section_ncs))
