import json
import os
import sys
import logging
import hashlib
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Type, Dict, Any, List, Union, Optional, Tuple
import numpy as np
import math
import ast
from functools import lru_cache
from AxisVariables import AxisVariable
from Axis import Axis

logger = logging.getLogger(__name__)

# Fast JSON library (optional)
try:
    import orjson as _fastjson
except Exception:
    _fastjson = None

# Mathematical functions for expression evaluation
_SCALAR_FUNCS = {
    'COS': math.cos, 'SIN': math.sin, 'TAN': math.tan,
    'ACOS': math.acos, 'ASIN': math.asin, 'ATAN': math.atan,
    'cos': math.cos, 'sin': math.sin, 'tan': math.tan,
    'acos': math.acos, 'asin': math.asin, 'atan': math.atan,
    'LOG': math.log, 'EXP': math.exp, 'SQRT': math.sqrt, 'ABS': abs,
    'log': math.log, 'exp': math.exp, 'sqrt': math.sqrt, 'abs': abs,
    'PI': math.pi, 'Pi': math.pi, 'pi': math.pi,
}
_VECTOR_FUNCS = {
    'COS': np.cos, 'SIN': np.sin, 'TAN': np.tan,
    'ACOS': np.arccos, 'ASIN': np.arcsin, 'ATAN': np.arctan,
    'cos': np.cos, 'sin': np.sin, 'tan': np.tan,
    'acos': np.arccos, 'asin': np.arcsin, 'atan': np.arctan,
    'LOG': np.log, 'EXP': np.exp, 'SQRT': np.sqrt, 'ABS': np.abs,
    'log': np.log, 'exp': np.exp, 'sqrt': np.sqrt, 'abs': np.abs,
    'PI': math.pi, 'Pi': math.pi, 'pi': math.pi,
}

_RESERVED_FUNC_NAMES = set(_SCALAR_FUNCS.keys()) | set(_VECTOR_FUNCS.keys())

_ALLOWED_AST = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant, ast.Name, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv, ast.USub, ast.UAdd, ast.Call
)

# Axis caching
_AXIS_CACHE: dict = {}

def _tiny_hash(arr) -> str:
    if arr is None:
        arr = []
    data = (','.join(str(float(v)) for v in arr)).encode('utf-8')
    return hashlib.blake2b(data, digest_size=12).hexdigest()

def get_axis_cached(axis_name, stations, x_coords, y_coords, z_coords):
    name_key = (axis_name or "").strip().lower()
    # Validate and pad lengths
    stations = stations or []
    x_coords = x_coords or []
    y_coords = y_coords or []
    z_coords = z_coords or []
    lengths = [len(stations), len(x_coords), len(y_coords), len(z_coords)]
    max_len = max(lengths)
    if max_len > 0:
        if len(stations) < max_len:
            last = stations[-1] if stations else 0.0
            stations = stations + [last] * (max_len - len(stations))
        if len(x_coords) < max_len:
            last = x_coords[-1] if x_coords else 0.0
            x_coords = x_coords + [last] * (max_len - len(x_coords))
        if len(y_coords) < max_len:
            last = y_coords[-1] if y_coords else 0.0
            y_coords = y_coords + [last] * (max_len - len(y_coords))
        if len(z_coords) < max_len:
            last = z_coords[-1] if z_coords else 0.0
            z_coords = z_coords + [last] * (max_len - len(z_coords))
    key = (
        name_key,
        tuple(float(s) for s in stations),
        tuple(float(v) for v in x_coords),
        tuple(float(v) for v in y_coords),
        tuple(float(v) for v in z_coords),
    )
    ax = _AXIS_CACHE.get(key)
    if ax is None:
        try:
            ax = Axis(stations=stations, x_coords=x_coords, y_coords=y_coords, z_coords=z_coords)
            if len(_AXIS_CACHE) > 128:
                _AXIS_CACHE.clear()
            _AXIS_CACHE[key] = ax
        except Exception as e:
            logger.error(f"Error creating Axis {axis_name}: {e}")
            return None
    return ax

@lru_cache(maxsize=4096)
def _compile_expr(expr_text: str):
    try:
        node = ast.parse(str(expr_text), mode='eval')
        for n in ast.walk(node):
            if not isinstance(n, _ALLOWED_AST):
                raise ValueError(f"Disallowed expression: {expr_text}")
            if isinstance(n, ast.Call):
                if not isinstance(n.func, ast.Name) or n.func.id not in _SCALAR_FUNCS:
                    raise ValueError(f"Disallowed function: {getattr(n.func, 'id', 'unknown')}")
        return compile(node, '<string>', 'eval')
    except Exception as e:
        logger.error(f"Error compiling expression '{expr_text}': {e}")
        raise

def _sanitize_vars(variables):
    result = {}
    for k, v in (variables.items() if isinstance(variables, dict) else variables):
        if k not in _RESERVED_FUNC_NAMES:
            try:
                if hasattr(v, 'variable_value'):
                    result[k] = float(v.variable_value)
                else:
                    result[k] = float(v)
            except (ValueError, TypeError) as e:
                logger.warning(f"Cannot convert variable '{k}' value '{v}' to float: {e}")
                result[k] = 0.0
    return result

def create_axis_variables(json_data: List[Dict], mapping: Dict[str, str]) -> List[AxisVariable]:
    transformed_data = []
    for var in json_data:
        transformed_var = {
            "name": var.get("VariableName"),
            "description": var.get("VariableDescription"),
            "values": [
                {"X": x, "Value": v, "Type": t}
                for x, v, t in zip(
                    var.get("StationValue", []),
                    var.get("VariableValues", []),
                    var.get("VariableIntTypes", ["#"] * len(var.get("StationValue", [])))
                )
            ]
        }
        transformed_data.append(transformed_var)
    logger.debug(f"Transformed axis_variables: {transformed_data}")
    return [AxisVariable(**var) for var in transformed_data]

def _build_axis_index(axis_data: List[Dict], mapping: Dict[str, str]) -> Dict[str, Axis]:
    axis_index = {}
    for ax in axis_data:
        name = ax.get(mapping.get("Name", "Name"), "")
        if not name:
            continue
        axis_index[name] = get_axis_cached(
            axis_name=name,
            stations=ax.get(mapping.get("stations", "StationValue"), []),
            x_coords=ax.get(mapping.get("x_coords", "CurvCoorX"), []),
            y_coords=ax.get(mapping.get("y_coords", "CurvCoorY"), []),
            z_coords=ax.get(mapping.get("z_coords", "CurvCoorZ"), [])
        )
    return axis_index

def load_from_json(cls: Type, json_file_path: Union[str, List[Dict]], mapping: Dict[Type, Dict[str, str]], axis_data: List[Dict] = None):
    data = []
    if isinstance(json_file_path, str):
        try:
            if _fastjson:
                with open(json_file_path, "rb") as f:
                    data = _fastjson.loads(f.read())
            else:
                with open(json_file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON file {json_file_path}: {e}")
            return [], []
    else:
        data = json_file_path

    objects = []
    filtered_data = []
    for obj in data:
        if obj.get("Class") == "ClassInfo":
            continue
        try:
            instance = from_dict(cls, obj, mapping, axis_data)
            objects.append(instance)
            filtered_data.append(obj)
        except Exception as e:
            logger.warning(f"Error creating {cls.__name__} from {obj.get('Name', 'unknown')}: {e}")
            continue
    return objects, filtered_data

def build_viso_object(obj, ctx: 'VisoContext', axis: Optional[Axis] = None, mainstations: Optional[Dict] = None, cross_sections_override: Optional[List['CrossSection']] = None, mapping_cfg: Optional[Dict] = None) -> Dict:
    cs_map = {cs.ncs: cs for cs in (cross_sections_override or [])}
    cs_map.update(ctx.crosssec_by_ncs)
    axis = axis or getattr(obj, 'axis_obj', None) or ctx.axes_by_name.get(getattr(obj, 'axis_name', ''))
    vis_data = obj.get_input_for_visualisation(ctx=ctx, axis_data={axis.name: axis} if axis else None, cross_section_objects=cs_map, mapping=mapping_cfg)
    vis_row = {
        'name': obj.name,
        'class': obj.__class__.__name__,
        'coords': vis_data.get('coords', np.array([])),
        'customdata': vis_data.get('customdata', []),
        'json_file': getattr(obj, 'json_file', None),
        'json_data': None
    }
    return vis_row

@dataclass(kw_only=True)
class VisoContext:
    crosssec_by_ncs: Dict[str, 'CrossSection'] = field(default_factory=dict)
    crosssec_by_name: Dict[str, 'CrossSection'] = field(default_factory=dict)
    deck_objects: List['DeckObject'] = field(default_factory=list)
    pier_objects: List['PierObject'] = field(default_factory=list)
    foundation_objects: List['FoundationObject'] = field(default_factory=list)
    mainstations_by_name: Dict[str, 'MainStation'] = field(default_factory=dict)
    main_stations: List['MainStation'] = field(default_factory=list)
    bearing_articulations: List['BearingArticulation'] = field(default_factory=list)
    axes_by_name: Dict[str, Axis] = field(default_factory=dict)
    other_objects: List[Dict] = field(default_factory=list)

    @classmethod
    def from_json(cls, axis_rows, cross_sections, mainstations, mapping_cfg):
        ctx = cls()
        for cs in cross_sections or []:
            if cs.valid:
                ctx.crosssec_by_ncs[cs.ncs] = cs
                ctx.crosssec_by_name[cs.name] = cs
        for ms in mainstations or []:
            ctx.mainstations_by_name[ms.name] = ms
            ctx.main_stations.append(ms)
        if axis_rows:
            axis_map = _build_axis_index(axis_rows, mapping_cfg.get(Axis, {}))
            ctx.axes_by_name.update(axis_map)
        return ctx

@dataclass(kw_only=True)
class BaseObject:
    no: Optional[str] = field(default=None, metadata={"json_key": "No"})
    class_name: Optional[str] = field(default=None, metadata={"json_key": "Class"})
    type: Optional[str] = field(default=None, metadata={"json_key": "Type"})
    description: Optional[str] = field(default=None, metadata={"json_key": "Description"})
    name: Optional[str] = field(default=None, metadata={"json_key": "Name"})
    inactive: Optional[str] = field(default=None, metadata={"json_key": "InActive"})
    axis_name: str = field(default="", metadata={"json_key": "Axis@Name"})
    axis_obj: Optional[Axis] = None
    axis_variables: Dict[str, AxisVariable] = field(default_factory=dict, metadata={"json_key": "AxisVariables"})
    variables: Dict[str, Variable] = field(default_factory=dict, metadata={"json_key": "Variables"})

    def __post_init__(self):
        if self.class_name and self.class_name not in ["ClassInfo", self.__class__.__name__]:
            logger.warning(f"Unexpected class '{self.class_name}' for {self.name}")

    def set_axis(self, axis_name: str, axis_data: List[Dict]):
        if not axis_data:
            return
        mapping = mapping.get(Axis, {})
        axis_index = _build_axis_index(axis_data, mapping)
        self.axis_obj = axis_index.get(axis_name)
        self.axis_name = axis_name

    def set_axis_variables(self, json_data: List[Dict], mapping: Dict[str, str]):
        self.axis_variables = {v.name: v for v in create_axis_variables(json_data, mapping)}

    def get_input_for_visualisation(self, ctx: 'VisoContext', axis_data: Dict = None, cross_section_objects: Dict = None, **kwargs):
        return {'coords': np.array([]), 'customdata': []}

@dataclass(kw_only=True)
class Variable:
    variable_name: str = field(metadata={"json_key": "VariableName"})
    variable_value: Union[str, float, List[str], List[float]] = field(metadata={"json_key": "VariableValue"})
    variable_unit: str = field(metadata={"json_key": "VariableUnit"})
    variable_description: str = field(metadata={"json_key": "VariableDescription"})

@dataclass(kw_only_only=True)
class Point:
    point_name: str = field(metadata={"json_key": "PointName"})
    coor_y: str = field(metadata={"json_key": "CoorY"})
    coor_z: str = field(metadata={"json_key": "CoorZ"})
    coor_y_val: Union[str, float] = field(metadata={"json_key": "CoorYVal"})
    coor_z_val: Union[str, float] = field(metadata={"json_key": "CoorZVal"})

@dataclass
class ReferenceFrame:
    variables: Dict[str, Variable] = field(default_factory=dict)

    def eval_equation(self, string_equation):
        try:
            return float(string_equation)
        except (TypeError, ValueError):
            pass
        try:
            code = _compile_expr(string_equation)
            env = {**_SCALAR_FUNCS, **_sanitize_vars(self.variables)}
            val = eval(code, {"__builtins__": {}}, env)
            if not math.isfinite(val):
                logger.warning(f"Error 2015: Non-finite result for '{string_equation}'")
                return "Error 2015"
            return float(val)
        except Exception as e:
            logger.warning(f"Error 2015 evaluating '{string_equation}': {e}")
            return "Error 2015"

@dataclass(kw_only=True)
class CrossSection(BaseObject):
    ncs: Optional[int] = field(default=None, metadata={"json_key": "NCS"})
    material1: Optional[int] = field(default=None, metadata={"json_key": "Material1"})
    material2: Optional[int] = field(default=None, metadata={"json_key": "Material2"})
    material_reinf: Optional[int] = field(default=None, metadata={"json_key": "Material_Reinf"})
    json_name: Union[str, List[str], None] = field(default=None, metadata={"json_key": "JSON_name"})
    sofi_code: Union[str, List[str], None] = field(default=None, metadata={"json_key": "SofiCode"})
    points: List[Point] = field(default_factory=list, metadata={"json_key": "Points"})
    reference_frame: Optional[ReferenceFrame] = None
    valid: bool = True

    def __post_init__(self):
        self.valid = len(self.points) > 0
        self.create_reference_frame()

    def create_reference_frame(self):
        if not self.reference_frame:
            self.reference_frame = ReferenceFrame(variables=self.variables)

    def get_input_for_visualisation(self, ctx: 'VisoContext', axis_data: Dict = None, cross_section_objects: Dict = None, **kwargs):
        if not self.points or not self.valid:
            logger.warning(f"No valid points for CrossSection {self.ncs}")
            return {'coords': np.array([]), 'customdata': []}
        coords = []
        customdata = []
        axis = axis_data.get(self.axis_name) if axis_data else self.axis_obj
        if not axis or not axis.stations:
            logger.warning(f"No valid axis for CrossSection {self.ncs}")
            return {'coords': np.array([]), 'customdata': []}
        for point in self.points:
            try:
                x = float(axis.stations[0])
                y = float(point.coor_y_val)
                z = float(point.coor_z_val)
                coords.append([x, y, z])
                customdata.append({'id': point.point_name, 'station': x, 'ncs': self.ncs})
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid coordinates for point {point.point_name} in CrossSection {self.ncs}: {e}")
        return {'coords': np.array(coords), 'customdata': customdata}

@dataclass(kw_only=True)
class DeckObject(BaseObject):
    cross_section_type: List[str] = field(default_factory=list, metadata={"json_key": "CrossSection@Type"})
    cross_section_name: List[str] = field(default_factory=list, metadata={"json_key": "CrossSection@Name"})
    grp_offset: List[float] = field(default_factory=list, metadata={"json_key": "GrpOffset"})
    placement_id: List[str] = field(default_factory=list, metadata={"json_key": "PlacementId"})
    placement_description: List[str] = field(default_factory=list, metadata={"json_key": "PlacementDescription"})
    ref_placement_id: List[str] = field(default_factory=list, metadata={"json_key": "Ref-PlacementId"})
    ref_station_offset: List[float] = field(default_factory=list, metadata={"json_key": "Ref-StationOffset"})
    station_value: List[float] = field(default_factory=list, metadata={"json_key": "StationValue"})
    cross_section_points_name: List[str] = field(default_factory=list, metadata={"json_key": "CrossSection_Points@Name"})
    grp: List[str] = field(default_factory=list, metadata={"json_key": "Grp"})
    cross_section_ncs: List[int] = field(default_factory=list, metadata={"json_key": "CrossSection@Ncs"})

    def get_input_for_visualisation(self, ctx: 'VisoContext', axis_data: Dict = None, cross_section_objects: Dict = None, **kwargs):
        # Use the first cross-section for simplicity
        cross_section = cross_section_objects.get(self.cross_section_name[0]) if cross_section_objects else None
        if not cross_section or not cross_section.points or not cross_section.valid:
            logger.warning(f"No valid cross-section {self.cross_section_name} for DeckObject {self.name}")
            return {'coords': np.array([]), 'customdata': []}
        axis = axis_data.get(self.axis_name) if axis_data else self.axis_obj
        if not axis or not axis.stations:
            logger.warning(f"No valid axis for DeckObject {self.name}")
            return {'coords': np.array([]), 'customdata': []}
        coords = []
        customdata = []
        stations = self.station_value or np.linspace(0, 1, num=10)
        for station in stations:
            for point in cross_section.points:
                try:
                    x = float(station)
                    y = float(point.coor_y_val)
                    z = float(point.coor_z_val)
                    coords.append([x, y, z])
                    customdata.append({'id': point.point_name, 'station': x, 'ncs': self.cross_section_ncs[0] if self.cross_section_ncs else ''})
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid coordinates for point {point.point_name} at station {station} in DeckObject {self.name}: {e}")
        return {'coords': np.array(coords), 'customdata': customdata}

@dataclass(kw_only=True)
class PierObject(BaseObject):
    object_axis_name: Optional[str] = field(default=None, metadata={"json_key": "ObjectAxisName"})
    cross_section_type: Optional[str] = field(default=None, metadata={"json_key": "CrossSection@Type"})
    cross_section_name: Optional[str] = field(default=None, metadata={"json_key": "CrossSection@Name"})
    ref_placement_id: Optional[str] = field(default=None, metadata={"json_key": "Ref-PlacementId"})
    ref_station_offset: Optional[str] = field(default=None, metadata={"json_key": "Ref-StationOffset"})
    station_value: Optional[float] = field(default=None, metadata={"json_key": "StationValue"})
    top_cross_section_points_name: Optional[str] = field(default=None, metadata={"json_key": "Top-CrossSection_Points@Name"})
    top_yoffset: Optional[float] = field(default=None, metadata={"json_key": "Top-Yoffset"})
    top_zoffset: Optional[float] = field(default=None, metadata={"json_key": "Top-Zoffset"})
    top_cross_section_ncs: Optional[int] = field(default=None, metadata={"json_key": "Top-CrossSection@Ncs"})
    bot_cross_section_points_name: Optional[str] = field(default=None, metadata={"json_key": "Bot-CrossSection_Points@Name"})
    bot_yoffset: Optional[float] = field(default=None, metadata={"json_key": "Bot-Yoffset"})
    bot_zoffset: Optional[float] = field(default=None, metadata={"json_key": "Bot-Zoffset"})
    bot_cross_section_ncs: Optional[int] = field(default=None, metadata={"json_key": "Bot-CrossSection@Ncs"})
    bot_pier_elevation: Optional[float] = field(default=None, metadata={"json_key": "Bot-PierElevation"})
    rotation_angle: Optional[float] = field(default=None, metadata={"json_key": "RotationAngle"})
    grp: Optional[str] = field(default=None, metadata={"json_key": "Grp"})
    fixation: Optional[str] = field(default=None, metadata={"json_key": "Fixation"})
    internal_placement_id: List[str] = field(default_factory=list, metadata={"json_key": "Internal-PlacementId"})
    internal_ref_placement_id: List[str] = field(default_factory=list, metadata={"json_key": "Internal_Ref-PlacementId"})
    internal_ref_station_offset: List[float] = field(default_factory=list, metadata={"json_key": "Internal_Ref-StationOffset"})
    internal_station_value: List[float] = field(default_factory=list, metadata={"json_key": "Internal-StationValue"})
    internal_cross_section_ncs: List[int] = field(default_factory=list, metadata={"json_key": "Internal-CrossSection@Ncs"})
    grp_offset: List[float] = field(default_factory=list, metadata={"json_key": "GrpOffset"})

    def get_input_for_visualisation(self, ctx: 'VisoContext', axis_data: Dict = None, cross_section_objects: Dict = None, **kwargs):
        top_cs = cross_section_objects.get(self.top_cross_section_points_name) if cross_section_objects else None
        bot_cs = cross_section_objects.get(self.bot_cross_section_points_name) if cross_section_objects else None
        if not top_cs or not bot_cs or not top_cs.points or not bot_cs.points or not top_cs.valid or not bot_cs.valid:
            logger.warning(f"No valid cross-sections for PierObject {self.name}")
            return {'coords': np.array([]), 'customdata': []}
        axis = axis_data.get(self.axis_name) if axis_data else self.axis_obj
        if not axis or not axis.stations:
            logger.warning(f"No valid axis for PierObject {self.name}")
            return {'coords': np.array([]), 'customdata': []}
        coords = []
        customdata = []
        for station in self.internal_station_value:
            for point in top_cs.points + bot_cs.points:
                try:
                    x = float(station)
                    y = float(point.coor_y_val)
                    z = float(point.coor_z_val)
                    coords.append([x, y, z])
                    customdata.append({'id': point.point_name, 'station': x, 'ncs': self.top_cross_section_points_name or self.bot_cross_section_points_name})
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid coordinates for point {point.point_name} at station {station} in PierObject {self.name}: {e}")
        return {'coords': np.array(coords), 'customdata': customdata}

@dataclass(kw_only=True)
class FoundationObject(BaseObject):
    object_axis_name: Optional[str] = field(default=None, metadata={"json_key": "ObjectAxisName"})
    cross_section_type: Optional[str] = field(default=None, metadata={"json_key": "CrossSection@Type"})
    cross_section_name: Optional[str] = field(default=None, metadata={"json_key": "CrossSection@Name"})
    ref_placement_id: Optional[str] = field(default=None, metadata={"json_key": "Ref-PlacementId"})
    ref_station_offset: Optional[float] = field(default=None, metadata={"json_key": "Ref-StationOffset"})
    station_value: Optional[float] = field(default=None, metadata={"json_key": "StationValue"})
    cross_section_points_name: Optional[str] = field(default=None, metadata={"json_key": "CrossSection_Points@Name"})
    foundation_ref_point_y_offset: Optional[float] = field(default=None, metadata={"json_key": "FoundationRefPoint-YOffset"})
    foundation_ref_point_x_offset: Optional[float] = field(default=None, metadata={"json_key": "FoundationRefPoint-XOffset"})
    foundation_level: Optional[float] = field(default=None, metadata={"json_key": "FoundationLevel"})
    rotation_angle: Optional[float] = field(default=None, metadata={"json_key": "RotationAngle"})
    pier_object_name: Optional[str] = field(default=None, metadata={"json_key": "PierObject@Name"})
    point1: Optional[str] = field(default=None, metadata={"json_key": "Point1"})
    point2: Optional[str] = field(default=None, metadata={"json_key": "Point2"})
    point3: Optional[str] = field(default=None, metadata={"json_key": "Point3"})
    point4: Optional[str] = field(default=None, metadata={"json_key": "Point4"})
    thickness: Optional[str] = field(default=None, metadata={"json_key": "Thickness"})
    grp: Optional[str] = field(default=None, metadata={"json_key": "Grp"})
    cross_section_ncs2: Optional[int] = field(default=None, metadata={"json_key": "CrossSection@Ncs2"})
    top_z_offset: Optional[float] = field(default=None, metadata={"json_key": "Top_ZOffset"})
    bot_z_offset: Optional[float] = field(default=None, metadata={"json_key": "Bot_ZOffset"})
    top_x_offset: Optional[float] = field(default=None, metadata={"json_key": "Top_Xoffset"})
    top_y_offset: Optional[float] = field(default=None, metadata={"json_key": "Top_Yoffset"})
    pile_dir_angle: Optional[float] = field(default=None, metadata={"json_key": "PileDirAngle"})
    pile_slope: Optional[float] = field(default=None, metadata={"json_key": "PileSlope"})
    kx: Optional[float] = field(default=None, metadata={"json_key": "Kx"})
    ky: Optional[float] = field(default=None, metadata={"json_key": "Ky"})
    kz: Optional[float] = field(default=None, metadata={"json_key": "Kz"})
    rx: Optional[float] = field(default=None, metadata={"json_key": "Rx"})
    ry: Optional[float] = field(default=None, metadata={"json_key": "Ry"})
    rz: Optional[float] = field(default=None, metadata={"json_key": "Rz"})
    fixation: Optional[str] = field(default=None, metadata={"json_key": "Fixation"})
    eval_pier_object_name: Optional[str] = field(default=None, metadata={"json_key": "Eval-PierObject@Name"})
    eval_station_value: Optional[float] = field(default=None, metadata={"json_key": "Eval-StationValue"})
    eval_bot_cross_section_points_name: Optional[str] = field(default=None, metadata={"json_key": "Eval_Bot-CrossSection_Points@Name"})
    eval_bot_y_offset: Optional[float] = field(default=None, metadata={"json_key": "Eval_Bot-Yoffset"})
    eval_bot_z_offset: Optional[float] = field(default=None, metadata={"json_key": "Eval_Bot-Zoffset"})
    eval_bot_pier_elevation: Optional[float] = field(default=None, metadata={"json_key": "Eval_Bot-PierElevation"})
    internal_placement_id: List[str] = field(default_factory=list, metadata={"json_key": "Internal-PlacementId"})
    internal_ref_placement_id: List[str] = field(default_factory=list, metadata={"json_key": "Internal_Ref-PlacementId"})
    internal_ref_station_offset: List[float] = field(default_factory=list, metadata={"json_key": "Internal_Ref-StationOffset"})
    internal_station_value: List[float] = field(default_factory=list, metadata={"json_key": "Internal-StationValue"})
    internal_cross_section_ncs: List[int] = field(default_factory=list, metadata={"json_key": "Internal-CrossSection@Ncs"})
    grp_offset: List[float] = field(default_factory=list, metadata={"json_key": "GrpOffset"})

    def get_input_for_visualisation(self, ctx: 'VisoContext', axis_data: Dict = None, cross_section_objects: Dict = None, **kwargs):
        cross_section = cross_section_objects.get(self.cross_section_name) if cross_section_objects else None
        if not cross_section or not cross_section.points or not cross_section.valid:
            logger.warning(f"No valid cross-section {self.cross_section_name} for FoundationObject {self.name}")
            return {'coords': np.array([]), 'customdata': []}
        axis = axis_data.get(self.axis_name) if axis_data else self.axis_obj
        if not axis or not axis.stations:
            logger.warning(f"No valid axis for FoundationObject {self.name}")
            return {'coords': np.array([]), 'customdata': []}
        coords = []
        customdata = []
        for point in cross_section.points:
            try:
                x = float(self.station_value)
                y = float(point.coor_y_val)
                z = float(point.coor_z_val)
                coords.append([x, y, z])
                customdata.append({'id': point.point_name, 'station': x, 'ncs': self.cross_section_name})
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid coordinates for point {point.point_name} in FoundationObject {self.name}: {e}")
        return {'coords': np.array(coords), 'customdata': customdata}

@dataclass(kw_only=True)
class MainStation(BaseObject):
    placement_id: Optional[str] = field(default=None, metadata={"json_key": "PlacementId"})
    ref_placement_id: Optional[str] = field(default=None, metadata={"json_key": "Ref-PlacementId"})
    ref_station_offset: Optional[float] = field(default=None, metadata={"json_key": "Ref-StationOffset"})
    station_value: Optional[float] = field(default=None, metadata={"json_key": "StationValue"})
    station_type: Optional[str] = field(default=None, metadata={"json_key": "StationType"})
    station_rotation_x: Optional[str] = field(default=None, metadata={"json_key": "StationRotationX"})
    station_rotation_z: Optional[str] = field(default=None, metadata={"json_key": "StationRotationZ"})
    sofi_code: Optional[str] = field(default=None, metadata={"json_key": "SofiCode"})

    def __post_init__(self):
        # The original has super().__post_init__(), but since BaseObject has it, it's called automatically
        pass

    def get_input_for_visualisation(self, ctx: 'VisoContext', axis_data: Dict = None, cross_section_objects: Dict = None, **kwargs):
        axis = axis_data.get(self.axis_name) if axis_data else self.axis_obj
        if not axis or not axis.stations:
            logger.warning(f"No valid axis for MainStation {self.name}")
            return {'coords': np.array([]), 'customdata': []}
        coords = [[self.station_value, 0.0, 0.0]]
        customdata = [{'id': self.name, 'station': self.station_value, 'ncs': ''}]
        return {'coords': np.array(coords), 'customdata': customdata}

@dataclass(kw_only=True)
class BearingArticulation(BaseObject):
    ref_placement_id: Optional[str] = field(default=None, metadata={"json_key": "Ref-PlacementId"})
    ref_station_offset: Optional[float] = field(default=None, metadata={"json_key": "Ref-StationOffset"})
    station_value: Optional[float] = field(default=None, metadata={"json_key": "StationValue"})
    pier_object_name: Optional[str] = field(default=None, metadata={"json_key": "PierObject@Name"})
    top_cross_section_points_name: Optional[str] = field(default=None, metadata={"json_key": "TopCrossSection@PointsName"})
    top_yoffset: Optional[float] = field(default=None, metadata={"json_key": "Top-YOffset"})
    top_xoffset: Optional[float] = field(default=None, metadata={"json_key": "Top-XOffset"})
    bot_cross_section_points_name: Optional[str] = field(default=None, metadata={"json_key": "BotCrossSection@PointsName"})
    bot_yoffset: Optional[float] = field(default=None, metadata={"json_key": "Bot-YOffset"})
    bot_xoffset: Optional[float] = field(default=None, metadata={"json_key": "Bot-XOffset"})
    kx: Optional[float] = field(default=None, metadata={"json_key": "Kx"})
    ky: Optional[float] = field(default=None, metadata={"json_key": "Ky"})
    kz: Optional[float] = field(default=None, metadata={"json_key": "Kz"})
    rx: Optional[float] = field(default=None, metadata={"json_key": "Rx"})
    ry: Optional[float] = field(default=None, metadata={"json_key": "Ry"})
    rz: Optional[float] = field(default=None, metadata={"json_key": "Rz"})
    grp_offset: Optional[float] = field(default=None, metadata={"json_key": "GRP-Offset"})
    grp: Optional[str] = field(default=None, metadata={"json_key": "GRP"})
    bearing_dimensions: Optional[str] = field(default=None, metadata={"json_key": "BearingDimensions"})
    rotation_x: Optional[float] = field(default=None, metadata={"json_key": "RotationX"})
    rotation_z: Optional[float] = field(default=None, metadata={"json_key": "RotationZ"})
    fixation: Optional[str] = field(default=None, metadata={"json_key": "Fixation"})
    eval_pier_object_name: Optional[str] = field(default=None, metadata={"json_key": "Eval-PierObject@Name"})
    eval_station_value: Optional[float] = field(default=None, metadata={"json_key": "Eval-StationValue"})
    eval_top_cross_section_points_name: Optional[str] = field(default=None, metadata={"json_key": "Eval-TopCrossSection@PointsName"})
    eval_top_yoffset: Optional[float] = field(default=None, metadata={"json_key": "Eval-Top-YOffset"})
    eval_top_zoffset: Optional[float] = field(default=None, metadata={"json_key": "Eval-Top-ZOffset"})
    sofi_code: Optional[str] = field(default=None, metadata={"json_key": "SOFi_Code"})

    def get_input_for_visualisation(self, ctx: 'VisoContext', axis_data: Dict = None, cross_section_objects: Dict = None, **kwargs):
        axis = axis_data.get(self.axis_name) if axis_data else self.axis_obj
        if not axis or not axis.stations:
            logger.warning(f"No valid axis for BearingArticulation {self.name}")
            return {'coords': np.array([]), 'customdata': []}
        coords = [[self.station_value, 0.0, 0.0]]
        customdata = [{'id': self.name, 'station': self.station_value, 'ncs': ''}]
        return {'coords': np.array(coords), 'customdata': customdata}

@dataclass(kw_only=True)
class TwoAxisBase(BaseObject):
    beg_axis_name: str = field(default="", metadata={"json_key": "Beg-Axis@Name"})
    end_axis_name: str = field(default="", metadata={"json_key": "End-Axis@Name"})
    beg_axis_obj: Optional[Axis] = None
    end_axis_obj: Optional[Axis] = None

@dataclass(kw_only=True)
class SecondaryObject(TwoAxisBase):
    beg_placement_id: Optional[str] = field(default=None, metadata={"json_key": "Beg-PlacementId"})
    beg_placement_description: Optional[str] = field(default=None, metadata={"json_key": "Beg-PlacementDescription"})
    beg_ref_placement_id: Optional[str] = field(default=None, metadata={"json_key": "Beg-Ref-PlacementId"})
    beg_ref_station_offset: Optional[float] = field(default=None, metadata={"json_key": "Beg-Ref-StationOffset"})
    beg_station_value: Optional[float] = field(default=None, metadata={"json_key": "Beg-StationValue"})
    beg_cross_section_points_name: Optional[str] = field(default=None, metadata={"json_key": "Beg-CrossSection@PointsName"})
    beg_ncs: Optional[int] = field(default=None, metadata={"json_key": "Beg-NCS"})
    end_placement_id: Optional[str] = field(default=None, metadata={"json_key": "End-PlacementId"})
    end_placement_description: Optional[str] = field(default=None, metadata={"json_key": "End-PlacementDescription"})
    end_ref_placement_id: Optional[str] = field(default=None, metadata={"json_key": "End-Ref-PlacementId"})
    end_ref_station_offset: Optional[float] = field(default=None, metadata={"json_key": "End-Ref-StationOffset"})
    end_station_value: Optional[float] = field(default=None, metadata={"json_key": "End-StationValue"})
    end_cross_section_points_name: Optional[str] = field(default=None, metadata={"json_key": "End-CrossSection@PointsName"})
    end_ncs: Optional[int] = field(default=None, metadata={"json_key": "End-NCS"})
    grp_offset: Optional[float] = field(default=None, metadata={"json_key": "GRP-Offset"})
    grp: Optional[str] = field(default=None, metadata={"json_key": "GRP"})

    def get_input_for_visualisation(self, ctx: 'VisoContext', axis_data: Dict = None, cross_section_objects: Dict = None, **kwargs):
        # Use beg and end axes for visualization
        beg_axis = axis_data.get(self.beg_axis_name) if axis_data else self.beg_axis_obj
        end_axis = axis_data.get(self.end_axis_name) if axis_data else self.end_axis_obj
        if not beg_axis or not end_axis or not beg_axis.stations or not end_axis.stations:
            logger.warning(f"No valid axes for SecondaryObject {self.name}")
            return {'coords': np.array([]), 'customdata': []}
        coords = []
        customdata = []
        coords.append([self.beg_station_value, 0.0, 0.0])
        customdata.append({'id': self.name, 'station': self.beg_station_value, 'ncs': self.beg_ncs})
        coords.append([self.end_station_value, 0.0, 0.0])
        customdata.append({'id': self.name, 'station': self.end_station_value, 'ncs': self.end_ncs})
        return {'coords': np.array(coords), 'customdata': customdata}

VERBOSE = False

def from_dict(cls: Type, data: Dict, mapping_config: Dict[Type, Dict[str, str]] = None, axis_data: List[Dict] = None):
    mapping_config = mapping_config or mapping
    field_map = {f.metadata.get("json_key", f.name): f.name for f in fields(cls) if f.metadata.get("json_key")}
    variable_map = mapping_config.get(Variable, {})
    point_map = mapping_config.get(Point, {})
    field_info = {f.name: f.type for f in fields(cls)}
    kwargs = {}

    if axis_data and Axis in mapping_config:
        axis_map = mapping_config.get(Axis, {})
        if "_axis_index" not in axis_map:
            axis_map["_axis_index"] = _build_axis_index(axis_data, axis_map)
            mapping_config[Axis] = axis_map

    reverse_key_map = {v: k for k, v in mapping_config.get(cls, {}).items()}
    axis_name_key = reverse_key_map.get('axis_name', 'Axis@Name')

    for json_key, value in data.items():
        field_name = field_map.get(json_key, json_key.lower().replace('@', '_').replace('-', '_'))
        if field_name not in field_info:
            logger.debug(f"Skipping unmapped field {field_name} for {cls.__name__}")
            continue
        if field_name == "points" and isinstance(value, (dict, list)):
            if isinstance(value, dict):
                value = value.get('Points', {})
            processed_points = []
            for p in value if isinstance(value, list) else value.values():
                if isinstance(p, dict):
                    point_data = {
                        point_map.get(k, k): v for k, v in p.items() if k not in ['CoorY', 'CoorZ', 'CoorYVal', 'CoorZVal', 'PointName']
                    }
                    point_data.update({
                        'point_name': p.get('PointName', ''),
                        'coor_y': p.get('CoorY', '0'),
                        'coor_z': p.get('CoorZ', '0'),
                        'coor_y_val': float(p.get('CoorYVal', '0')),
                        'coor_z_val': float(p.get('CoorZVal', '0'))
                    })
                    processed_points.append(Point(**point_data))
                else:
                    logger.warning(f"Invalid point data in CrossSection {data.get('Name', 'unknown')}") 
            value = processed_points
        elif field_name == "variables" and isinstance(value, dict) and value and not isinstance(list(value.values())[0], Variable):
            value = {k: Variable(
                variable_name=k,
                variable_value=v,
                variable_unit="[mm]",
                variable_description=f"{k}:Axis Variable"
            ) for k, v in value.items()}
        elif field_name == "variables" and isinstance(value, list):
            value = {v.variable_name: v for v in [Variable(**{variable_map.get(k, k): v for k, v in v.items()}) for v in value if isinstance(v, dict)]}
        elif field_name == "axis_variables" and isinstance(value, (dict, list)):
            value = {v.name: v for v in create_axis_variables(value.values() if isinstance(value, dict) else value, mapping_config.get(AxisVariable, {}))}
        elif field_name == "axis_obj" and field_info[field_name] == Axis:
            kwargs[field_name] = None
            kwargs['axis_name'] = data.get(axis_name_key, '')
        elif is_dataclass(field_info[field_name]) and isinstance(value, dict):
            kwargs[field_name] = from_dict(field_info[field_name], value, mapping_config, axis_data)
        else:
            kwargs[field_name] = value

    valid_fields = set(field_info.keys())
    invalid_kwargs = set(kwargs.keys()) - valid_fields
    if invalid_kwargs:
        logger.warning(f"Invalid fields for {cls.__name__}: {invalid_kwargs}")
        kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}

    try:
        instance = cls(**kwargs)
        if cls == CrossSection:
            instance.valid = len(instance.points) > 0
            instance.create_reference_frame()
        return instance
    except Exception as e:
        logger.error(f"Error creating {cls.__name__}: {e}")
        raise

mapping: Dict[Type, Dict[str, str]] = {
    Axis: {
        "Class": "Class",
        "Name": "Name",
        "StaionValue": "stations",
        "CurvCoorX": "x_coords",
        "CurvCoorY": "y_coords",
        "CurvCoorZ": "z_coords"
    },
    AxisVariable: {
        "VariableName": "VariableName",
        "StationValue": "VariableStations",
        "VariableValues": "VariableValues",
        "VariableIntTypes": "VariableIntTypes",
        "VariableDescription": "VariableDescription"
    },
    PierObject: {
        "No": "no",
        "Class": "class_name",
        "Type": "type",
        "Description": "description",
        "Name": "name",
        "InActive": "inactive",
        "ObjectAxisName": "object_axis_name",
        "CrossSection@Type": "cross_section_type",
        "CrossSection@Name": "cross_section_name",
        "Axis@Name": "axis_name",
        "Ref-PlacementId": "ref_placement_id",
        "Ref-StationOffset": "ref_station_offset",
        "StationValue": "station_value",
        "Top-CrossSection_Points@Name": "top_cross_section_points_name",
        'Top-CrossSection_Points@Name': 'top_cross_section_points_name',
        "Top-Yoffset": "top_yoffset",
        "Top-Zoffset": "top_zoffset",
        "Top-CrossSection@Ncs": "top_cross_section_ncs",
        'Top-CrossSection@NCS': 'top_cross_section_ncs',
        "Bot-CrossSection_Points@Name": "bot_cross_section_points_name",
        'Bot-CrossSection_Points@Name': 'bot_cross_section_points_name',
        "Bot-Yoffset": "bot_yoffset",
        "Bot-Zoffset": "bot_zoffset",
        'Bot-CrossSection@NCS': 'bot_cross_section_ncs',
        "Bot-CrossSection@Ncs": "bot_cross_section_ncs",
        "Bot-PierElevation": "bot_pier_elevation",
        "RotationAngle": "rotation_angle",
        "Grp": "grp",
        "Fixation": "fixation",
        "Internal-PlacementId": "internal_placement_id",
        "Internal_Ref-PlacementId": "internal_ref_placement_id",
        "Internal_Ref-StationOffset": "internal_ref_station_offset",
        "Internal@StationValue": "internal_station_value",
        'Internal-StationValue': 'internal_station_value',
        "Internal-CrossSection@Ncs": "internal_cross_section_ncs",
        'Internal-CrossSection@NCS': 'internal_cross_section_ncs',
        "GrpOffset": "grp_offset",
        "AxisVariables": "axis_variables"
    },
    FoundationObject: {
        "No": "no",
        "Class": "class_name",
        "Type": "type",
        "Description": "description",
        "Name": "name",
        "InActive": "inactive",
        "ObjectAxisName": "object_axis_name",
        "CrossSection@Type": "cross_section_type",
        "CrossSection@Name": "cross_section_name",
        "Ref-PlacementId": "ref_placement_id",
        "Ref-StationOffset": "ref_station_offset",
        "StationValue": "station_value",
        "CrossSection_Points@Name": "cross_section_points_name",
        "FoundationRefPoint-YOffset": "foundation_ref_point_y_offset",
        "FoundationRefPoint-XOffset": "foundation_ref_point_x_offset",
        "FoundationLevel": "foundation_level",
        "RotationAngle": "rotation_angle",
        "Axis@Name": "axis_name",
        "PierObject@Name": "pier_object_name",
        "Point1": "point1",
        "Point2": "point2",
        "Point3": "point3",
        "Point4": "point4",
        "Thickness": "thickness",
        "Grp": "grp",
        "CrossSection@Ncs2": "cross_section_ncs2",
        "Top_ZOffset": "top_z_offset",
        "Bot_ZOffset": "bot_z_offset",
        "Top_Xoffset": "top_x_offset",
        "Top_Yoffset": "top_y_offset",
        "PileDirAngle": "pile_dir_angle",
        "PileSlope": "pile_slope",
        "Kx": "kx",
        "Ky": "ky",
        "Kz": "kz",
        "Rx": "rx",
        "Ry": "ry",
        "Rz": "rz",
        "Fixation": "fixation",
        "Eval-PierObject@Name": "eval_pier_object_name",
        "Eval-StationValue": "eval_station_value",
        "Eval_Bot-CrossSection_Points@Name": "eval_bot_cross_section_points_name",
        "Eval_Bot-Yoffset": "eval_bot_y_offset",
        "Eval_Bot-Zoffset": "eval_bot_z_offset",
        "Eval_Bot-PierElevation": "eval_bot_pier_elevation",
        "Internal-PlacementId": "internal_placement_id",
        "Internal_Ref-PlacementId": "internal_ref_placement_id",
        "Internal_Ref-StationOffset": "internal_ref_station_offset",
        "Internal-StationValue": "internal_station_value",
        "Internal-CrossSection@Ncs": "internal_cross_section_ncs",
        "GrpOffset": "grp_offset",
        "AxisVariables": "axis_variables"
    },
    CrossSection: {
        "No": "no",
        "Class": "class_name",
        "Type": "type",
        "Description": "description",
        "Name": "name",
        "InActive": "inactive",
        "NCS": "ncs",
        "Material1": "material1",
        "Material2": "material2",
        "Material_Reinf": "material_reinf",
        "JSON_name": "json_name",
        "SofiCode": "sofi_code",
        "Points": "points",
        "Variables": "variables"
    },

     MainStation: {
        "No": "no",
        "Class": "class_name",
        "Type": "type",
        "Description": "description",
        "Name": "name",
        "InActive": "inactive",
        "Axis@Name": "axis_name",
        "PlacementId": "placement_id",
        "Ref-PlacementId": "ref_placement_id",
        "Ref-StationOffset": "ref_station_offset",
        "StationValue": "station_value",
        "StationType": "station_type",
        "StationRotationX": "station_rotation_x",
        "StationRotationZ": "station_rotation_z",
        "SOFi_Code": "sofi_code",
        "SofiCode": "sofi_code",
    },

    BearingArticulation: {
        "No": "no",
        "Class": "class_name",
        "Type": "type",
        "Description": "description",
        "Name": "name",
        "InActive": "inactive",
        "Ref-PlacementId": "ref_placement_id",
        "Ref-StationOffset": "ref_station_offset",
        "StationValue": "station_value",
        "PierObject@Name": "pier_object_name",
        "TopCrossSection@PointsName": "top_cross_section_points_name",
        "Top-YOffset": "top_yoffset",
        "Top-XOffset": "top_xoffset",
        "BotCrossSection@PointsName": "bot_cross_section_points_name",
        "Bot-YOffset": "bot_yoffset",
        "Bot-XOffset": "bot_xoffset",
        "Kx": "kx",
        "Ky": "ky",
        "Kz": "kz",
        "Rx": "rx",
        "Ry": "ry",
        "Rz": "rz",
        "GRP-Offset": "grp_offset",
        "GRP": "grp",
        "BearingDimensions": "bearing_dimensions",
        "RotationX": "rotation_x",
        "RotationZ": "rotation_z",
        "Fixation": "fixation",
        "Eval-PierObject@Name": "eval_pier_object_name",
        "Eval-StationValue": "eval_station_value",
        "Eval-TopCrossSection@PointsName": "eval_top_cross_section_points_name",
        "Eval-Top-YOffset": "eval_top_yoffset",
        "Eval-Top-ZOffset": "eval_top_zoffset",
        "SOFi_Code": "sofi_code",
    },

    SecondaryObject: {
        "No": "no",
        "Class": "class_name",
        "Type": "type",
        "Description": "description",
        "Name": "name",
        "InActive": "inactive",

        # axes (two)
        "Beg-Axis@Name": "beg_axis_name",
        "End-Axis@Name": "end_axis_name",

        # beg side
        "Beg-PlacementId": "beg_placement_id",
        "Beg-PlacementDescription": "beg_placement_description",
        "Beg-Ref-PlacementId": "beg_ref_placement_id",
        "Beg-Ref-StationOffset": "beg_ref_station_offset",
        "Beg-StationValue": "beg_station_value",
        "Beg-CrossSection@PointsName": "beg_cross_section_points_name",
        "Beg-NCS": "beg_ncs",

        # end side
        "End-PlacementId": "end_placement_id",
        "End-PlacementDescription": "end_placement_description",
        "End-Ref-PlacementId": "end_ref_placement_id",
        "End-Ref-StationOffset": "end_ref_station_offset",
        "End-StationValue": "end_station_value",
        "End-CrossSection@PointsName": "end_cross_section_points_name",
        "End-NCS": "end_ncs",

        "GRP-Offset": "grp_offset",
        "GRP": "grp",
    }
}

VERBOSE = False  # Set to True for detailed debug output