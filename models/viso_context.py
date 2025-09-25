# models/viso_context.py
from __future__ import annotations

import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict

from .axis import Axis
from .cross_section import CrossSection
from .main_station import MainStationRef
from .base import BaseObject

logger = logging.getLogger(__name__)


class VisoContext:
    """
    Context holding all geometric and structural data for a VISo project.
    
    Provides organized access to axes, cross-sections, main stations, and objects.
    Handles wiring of axis references and variable mappings.
    """
    
    def __init__(self):
        self.axes_by_name: Dict[str, Axis] = {}
        self.crosssec_by_ncs: Dict[int, CrossSection] = {}
        self.crosssec_by_name: Dict[str, CrossSection] = {}
        self.mainstations_by_name: Dict[str, MainStationRef] = {}
        self.objects: List[BaseObject] = []
        
    @classmethod
    def from_json(
        cls,
        axis_rows: List[Dict[str, Any]],
        cross_sections: List[CrossSection],
        mainstations: List[MainStationRef],
        mapping_cfg: Dict[str, Any],
        verbose: bool = False
    ) -> VisoContext:
        """
        Create VisoContext from raw JSON data and instantiated objects.
        
        Args:
            axis_rows: Raw axis data from JSON
            cross_sections: Instantiated CrossSection objects
            mainstations: Instantiated MainStation objects
            mapping_cfg: Field mapping configuration
            verbose: Enable verbose logging
        """
        ctx = cls()
        
        # Create Axis objects from raw data
        axis_mapping = mapping_cfg.get("Axis", {})
        for row in axis_rows:
            try:
                # Map JSON fields to Axis constructor parameters
                stations = row.get(axis_mapping.get("stations", "StaionValue"), [])
                x_coords = row.get(axis_mapping.get("x_coords", "CurvCoorX"), [])
                y_coords = row.get(axis_mapping.get("y_coords", "CurvCoorY"), [])
                z_coords = row.get(axis_mapping.get("z_coords", "CurvCoorZ"), [])
                name = row.get(axis_mapping.get("name", "Name"), "")
                
                if not all([stations, x_coords, y_coords, z_coords, name]):
                    if verbose:
                        logger.warning(f"Skipping incomplete axis data: {name}")
                    continue
                    
                # Convert string arrays to float arrays if needed
                stations = [float(s) for s in stations] if isinstance(stations, list) else stations
                x_coords = [float(x) for x in x_coords] if isinstance(x_coords, list) else x_coords
                y_coords = [float(y) for y in y_coords] if isinstance(y_coords, list) else y_coords
                z_coords = [float(z) for z in z_coords] if isinstance(z_coords, list) else z_coords
                
                axis = Axis(
                    stations=stations,
                    x_coords=x_coords,
                    y_coords=y_coords,
                    z_coords=z_coords,
                    units="m"  # JSON data appears to be in meters
                )
                ctx.axes_by_name[name] = axis
                
                if verbose:
                    logger.info(f"Created axis: {name} with {len(stations)} stations")
                    
            except Exception as e:
                if verbose:
                    logger.warning(f"Failed to create axis from row: {e}")
                continue
        
        # Store cross-sections by NCS and name
        for cs in cross_sections:
            # Get NCS from data dict
            ncs = cs.data.get('ncs') if hasattr(cs, 'data') and isinstance(cs.data, dict) else None
            if ncs is not None:
                ctx.crosssec_by_ncs[ncs] = cs
            if hasattr(cs, 'name') and cs.name:
                ctx.crosssec_by_name[cs.name] = cs
                
        # Store main stations by placement_id (acting as name)
        for ms in mainstations:
            if hasattr(ms, 'placement_id') and ms.placement_id:
                ctx.mainstations_by_name[ms.placement_id] = ms
                
        if verbose:
            logger.info(f"VisoContext created: {len(ctx.axes_by_name)} axes, "
                       f"{len(ctx.crosssec_by_ncs)} cross-sections, "
                       f"{len(ctx.mainstations_by_name)} main stations")
                       
        return ctx
    
    def add_objects(self, *object_lists, axis_var_map: Optional[Dict[str, Any]] = None):
        """
        Add objects to the context and wire their axis references.
        
        Args:
            *object_lists: Lists of objects to add
            axis_var_map: Optional axis variable mappings
        """
        for obj_list in object_lists:
            if obj_list:
                for obj in obj_list:
                    if isinstance(obj, BaseObject):
                        self.objects.append(obj)
                        # Wire axis reference if the object has an axis_name
                        if hasattr(obj, 'axis_name') and obj.axis_name:
                            if obj.axis_name in self.axes_by_name:
                                obj.axis_obj = self.axes_by_name[obj.axis_name]
                            else:
                                logger.warning(f"Axis '{obj.axis_name}' not found for object {getattr(obj, 'name', obj)}")
                        
                        # Set axis variables if available
                        if hasattr(obj, 'set_axis_variables'):
                            obj.set_axis_variables(axis_var_map)
    
    def wire_objects(self, axis_var_map: Optional[Dict[str, Any]] = None):
        """
        Wire axis references and variables for all objects in the context.
        """
        for obj in self.objects:
            if hasattr(obj, 'axis_name') and obj.axis_name and not hasattr(obj, 'axis_obj'):
                if obj.axis_name in self.axes_by_name:
                    obj.axis_obj = self.axes_by_name[obj.axis_name]
                    
            if hasattr(obj, 'set_axis_variables'):
                obj.set_axis_variables(axis_var_map)