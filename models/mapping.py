from __future__ import annotations
from typing import Any, Dict

_MAPPING_BY_NAME: Dict[str, Dict[str, Any]] = {
    "DeckObject": {
        "No": "no",
        "Class": "class_name",
        "Type": "type",
        "Description": "description",
        "Name": "name",
        "InActive": "inactive",
        "Axis@Name": "axis_name",
        "CrossSection@Type": "cross_section_types",
        "CrossSection@Name": "cross_section_names",
        "GrpOffset": "grp_offset",
        "PlacementId": "placement_id",
        "PlacementDescription": "placement_description",
        "Ref-PlacementId": "ref_placement_id",
        "Ref-StationOffset": "ref_station_offset",
        "StationValue": "station_value",
        "CrossSection_Points@Name": "cross_section_points_name",
        "Grp": "grp",
        "CrossSection@Ncs": "cross_section_ncs",
        "AxisVariables": "axis_variables"
    },
    "Axis": {
        "Class": "Class",
        "Name": "Name",
        "StaionValue": "stations",
        "CurvCoorX": "x_coords",
        "CurvCoorY": "y_coords",
        "CurvCoorZ": "z_coords"
    },
    "AxisVariable": {
        "VariableName": "VariableName",
        "StationValue": "VariableStations",
        "VariableValues": "VariableValues",
        "VariableIntTypes": "VariableIntTypes",
        "VariableDescription": "VariableDescription"
    },
    "PierObject": {
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
        "Top-CrossSection_Point@Name": "top_cross_section_points_name",
        'Top-CrossSection_Points@Name': 'top_cross_section_points_name',
        "Top-Yoffset": "top_yoffset",
        "Top-Zoffset": "top_zoffset",
        "Top-CrossSection@Ncs": "top_cross_section_ncs",
        'Top-CrossSection@NCS': 'top_cross_section_ncs',
        "Bot-CrossSection_Point@Name": "bot_cross_section_points_name",
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
    "FoundationObject": {
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
    "CrossSection": {
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

    "MainStation": {
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
    "BearingArticulation": {
        "No": "no",
        "Class": "class_name",
        "Type": "type",
        "Description": "description",
        "Name": "name",
        "InActive": "inactive",
        "Axis@Name": "axis_name",
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
    "SecondaryObject": {
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

def _name_of(key: Any) -> str:
    if isinstance(key, str):
        return key
    n = getattr(key, "__name__", None)
    if n:
        return n
    cls = getattr(key, "__class__", None)
    if cls and getattr(cls, "__name__", None):
        return cls.__name__
    return str(key)

class _MappingFacade:
    """Dict-like facade so existing code can keep calling mapping.get(Axis, {})."""

    def get(self, key: Any, default=None):
        # IMPORTANT: return the provided default when missing
        return _MAPPING_BY_NAME.get(_name_of(key), default)

    def __getitem__(self, key: Any):
        return _MAPPING_BY_NAME[_name_of(key)]

    def __contains__(self, key: Any):
        return _name_of(key) in _MAPPING_BY_NAME

    @property
    def by_name(self) -> Dict[str, Dict[str, Any]]:
        return _MAPPING_BY_NAME

mapping = _MappingFacade()