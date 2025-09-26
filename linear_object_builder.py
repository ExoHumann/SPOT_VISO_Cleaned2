"""
LinearObjectBuilder - Streamlined workflow for creating LinearObjects from SPOT data.

This module provides a unified interface for loading SPOT JSON data and creating
configured LinearObject instances (Deck, Pier, Foundation) with minimal boilerplate.
"""

from __future__ import annotations
import json
import os
from typing import Dict, List, Optional, Type, TypeVar, Union
from dataclasses import dataclass
from pathlib import Path

# Import the core components
from models.base import (
    load_axis_from_rows, index_cross_sections_by_ncs, 
    load_section_for_ncs, from_dict
)
from models.main_station import load_mainstations_from_rows
from models.mapping import mapping
from models.linear_object import LinearObject
from models.deck_object import DeckObject  
from models.pier_object import PierObject
from models.foundation_object import FoundationObject
from models.axis import Axis
from models.cross_section import CrossSection
from models.main_station import MainStationRef

# Import SPOT_Filters components
from SPOT_Filters import get_subfolders, get_json_files, load_json_objects, SpotJsonObject

T = TypeVar('T', bound=LinearObject)

@dataclass
class ComponentData:
    """Container for parsed component data."""
    axes: Dict[str, Axis]
    cross_sections: Dict[int, CrossSection]
    mainstations: Dict[str, List[MainStationRef]]
    raw_objects: List[dict]

class LinearObjectBuilder:
    """
    Streamlined builder for creating LinearObject instances from SPOT data.
    
    This class simplifies the workflow by:
    - Centralizing data loading and parsing
    - Automatically configuring objects with available components
    - Providing validation and error handling
    - Supporting both file-based and SPOT folder-based workflows
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._components: Optional[ComponentData] = None
        
    def load_from_files(self, 
                       axis_json: str,
                       cross_json: str, 
                       obj_json: str,
                       mainstation_json: str,
                       section_json: str) -> 'LinearObjectBuilder':
        """
        Load component data from individual JSON files.
        
        This method loads data in the format used by run_linear.py, run_pier.py, etc.
        """
        if self.verbose:
            print(f"Loading data from files:")
            print(f"  Axis: {axis_json}")
            print(f"  Cross: {cross_json}")
            print(f"  Objects: {obj_json}")
            print(f"  MainStation: {mainstation_json}")
            print(f"  Section: {section_json}")
        
        # Load raw data
        axis_rows = json.load(open(axis_json, "r", encoding="utf-8"))
        cross_rows = json.load(open(cross_json, "r", encoding="utf-8"))
        obj_rows = json.load(open(obj_json, "r", encoding="utf-8"))
        ms_rows = json.load(open(mainstation_json, "r", encoding="utf-8"))
        
        # Parse components
        self._components = self._parse_components(axis_rows, cross_rows, obj_rows, ms_rows, section_json)
        
        if self.verbose:
            print(f"Loaded {len(self._components.axes)} axes")
            print(f"Loaded {len(self._components.cross_sections)} cross sections")
            print(f"Loaded {len(self._components.raw_objects)} objects")
        
        return self
    
    def load_from_spot_folder(self, 
                             master_folder: str, 
                             branch: str = "MAIN") -> 'LinearObjectBuilder':
        """
        Load component data from SPOT folder structure using SPOT_Filters.
        
        This method uses the SPOT_Filters system to load data from a complete
        SPOT project folder structure.
        """
        if self.verbose:
            print(f"Loading SPOT data from: {master_folder}/{branch}")
            
        # Use SPOT_Filters to load data
        branch_folder = os.path.join(master_folder, branch)
        if not os.path.exists(branch_folder):
            raise ValueError(f"Branch folder not found: {branch_folder}")
            
        json_files = get_json_files(branch_folder)
        if not json_files:
            raise ValueError(f"No JSON files found in {branch_folder}")
            
        raw_objects = load_json_objects(json_files)
        if not raw_objects:
            raise ValueError(f"No objects loaded from {json_files}")
        
        # Convert to SpotJsonObject for consistency
        spot_objects = [SpotJsonObject(obj) if isinstance(obj, dict) else obj for obj in raw_objects]
        
        # Separate by class type
        axes_data = [obj.to_dict() for obj in spot_objects if obj.get("Class") == "Axis"]
        cross_data = [obj.to_dict() for obj in spot_objects if obj.get("Class") == "CrossSection"]  
        ms_data = [obj.to_dict() for obj in spot_objects if obj.get("Class") == "MainStation"]
        obj_data = [obj.to_dict() for obj in spot_objects 
                   if obj.get("Class") in ["DeckObject", "PierObject", "FoundationObject"]]
        
        # For section_json, we need to find section data files
        section_json = self._find_section_json(branch_folder)
        
        # Parse components
        self._components = self._parse_components(axes_data, cross_data, obj_data, ms_data, section_json)
        
        if self.verbose:
            print(f"Loaded {len(self._components.axes)} axes")
            print(f"Loaded {len(self._components.cross_sections)} cross sections") 
            print(f"Loaded {len(self._components.raw_objects)} objects")
        
        return self
    
    def create_object(self, 
                     obj_type: str, 
                     name: Optional[str] = None,
                     obj_data: Optional[dict] = None) -> LinearObject:
        """
        Create and configure a LinearObject from loaded data.
        
        Args:
            obj_type: Type of object to create ("DeckObject", "PierObject", "FoundationObject")  
            name: Name of specific object to create (if None, uses first available)
            obj_data: Explicit object data (if None, searches loaded data)
        
        Returns:
            Configured LinearObject instance
        """
        if not self._components:
            raise ValueError("No data loaded. Call load_from_files() or load_from_spot_folder() first.")
        
        # Determine object class and mapping
        obj_class = self._get_object_class(obj_type)
        obj_mapping = mapping.get(obj_type)
        if not obj_mapping:
            raise ValueError(f"No mapping found for object type: {obj_type}")
        
        # Find object data
        if obj_data:
            obj_row = obj_data
        else:
            obj_row = self._find_object_data(obj_type, name)
            
        if self.verbose:
            print(f"Creating {obj_type}: {obj_row.get('name', obj_row.get('Name', 'Unnamed'))}")
        
        # Create object using mapping
        obj = from_dict(obj_class, obj_row, mapping)
        
        # Auto-configure with available components
        self._auto_configure(obj)
        
        # Validate configuration
        self._validate_configuration(obj)
        
        return obj
    
    def create_all_objects(self) -> Dict[str, List[LinearObject]]:
        """
        Create all available objects from loaded data.
        
        Returns:
            Dictionary mapping object type to list of created objects
        """
        if not self._components:
            raise ValueError("No data loaded. Call load_from_files() or load_from_spot_folder() first.")
            
        results = {"DeckObject": [], "PierObject": [], "FoundationObject": []}
        
        for obj_row in self._components.raw_objects:
            obj_class_name = obj_row.get("Class")
            if obj_class_name in results:
                try:
                    obj = self.create_object(obj_class_name, obj_data=obj_row)
                    results[obj_class_name].append(obj)
                except Exception as e:
                    if self.verbose:
                        print(f"Failed to create {obj_class_name}: {e}")
                        
        return results
    
    def build_geometry(self, 
                      obj: LinearObject,
                      stations_m: Optional[List[float]] = None,
                      twist_deg: float = 0.0,
                      plan_rotation_deg: float = 0.0,
                      station_cap: Optional[int] = None,
                      **kwargs) -> dict:
        """
        Build 3D geometry for a LinearObject with standard parameters.
        
        Args:
            obj: Configured LinearObject instance
            stations_m: Evaluation stations (None for automatic)
            twist_deg: Extra in-plane rotation
            plan_rotation_deg: Plan rotation in XY plane
            station_cap: Maximum number of stations
            **kwargs: Additional build parameters
            
        Returns:
            Geometry result dictionary
        """
        try:
            result = obj.build(
                stations_m=stations_m,
                twist_deg=twist_deg,
                plan_rotation_deg=plan_rotation_deg, 
                station_cap=station_cap,
                **kwargs
            )
            
            if self.verbose:
                name = getattr(obj, 'name', 'Unnamed')
                print(f"Built geometry for {obj.__class__.__name__}: {name}")
                if 'stations_mm' in result:
                    print(f"  Generated {len(result['stations_mm'])} stations")
                    
            return result
            
        except Exception as e:
            name = getattr(obj, 'name', 'Unnamed')
            raise ValueError(f"Failed to build geometry for {obj.__class__.__name__} '{name}': {e}") from e
    
    # Private helper methods
    
    def _parse_components(self, axis_rows, cross_rows, obj_rows, ms_rows, section_json) -> ComponentData:
        """Parse raw data into component objects."""
        
        # Parse axes
        available_axes = {}
        for axis_row in axis_rows:
            if axis_row.get("Class") == "Axis":
                axis_name = axis_row.get("Name", "")
                if axis_name:
                    try:
                        available_axes[axis_name] = load_axis_from_rows(axis_rows, axis_name)
                    except Exception as e:
                        if self.verbose:
                            print(f"Warning: Failed to load axis '{axis_name}': {e}")
        
        # Parse cross sections
        available_cross_sections = {}
        by_ncs = index_cross_sections_by_ncs(cross_rows)
        for ncs in by_ncs.keys():
            try:
                available_cross_sections[ncs] = load_section_for_ncs(ncs, by_ncs, section_json)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed to load cross section NCS {ncs}: {e}")
        
        # Parse mainstations
        available_mainstations = {}
        for axis_name in available_axes.keys():
            try:
                available_mainstations[axis_name] = load_mainstations_from_rows(ms_rows, axis_name=axis_name)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed to load mainstations for axis '{axis_name}': {e}")
        
        return ComponentData(
            axes=available_axes,
            cross_sections=available_cross_sections,
            mainstations=available_mainstations,
            raw_objects=obj_rows
        )
    
    def _get_object_class(self, obj_type: str) -> Type[LinearObject]:
        """Get the class for a given object type string."""
        class_map = {
            "DeckObject": DeckObject,
            "PierObject": PierObject,
            "FoundationObject": FoundationObject
        }
        if obj_type not in class_map:
            raise ValueError(f"Unknown object type: {obj_type}")
        return class_map[obj_type]
    
    def _find_object_data(self, obj_type: str, name: Optional[str]) -> dict:
        """Find object data by type and optional name."""
        candidates = [obj for obj in self._components.raw_objects if obj.get("Class") == obj_type]
        
        if not candidates:
            raise ValueError(f"No {obj_type} objects found in loaded data")
        
        if name:
            named = [obj for obj in candidates if obj.get("name", obj.get("Name")) == name]
            if not named:
                raise ValueError(f"No {obj_type} named '{name}' found")
            return named[0]
        
        return candidates[0]  # Return first available
    
    def _auto_configure(self, obj: LinearObject):
        """Automatically configure object with available components."""
        obj.configure(
            available_axes=self._components.axes,
            available_cross_sections=self._components.cross_sections,
            available_mainstations=self._components.mainstations
        )
    
    def _validate_configuration(self, obj: LinearObject):
        """Validate that object configuration is complete."""
        issues = []
        
        # Check for required axis
        if not hasattr(obj, 'axis_obj') or obj.axis_obj is None:
            axis_name = getattr(obj, 'axis_name', 'Unknown')
            issues.append(f"No axis configured (axis_name: {axis_name})")
        
        # Check for cross sections (for objects that need them)
        if hasattr(obj, 'sections_by_ncs') and obj.sections_by_ncs is None:
            if hasattr(obj, 'base_section') and obj.base_section is None:
                issues.append("No cross sections configured")
        
        if issues:
            name = getattr(obj, 'name', 'Unnamed')
            raise ValueError(f"Configuration issues for {obj.__class__.__name__} '{name}': {', '.join(issues)}")
    
    def _find_section_json(self, folder_path: str) -> str:
        """Find section JSON file in folder."""
        # Look for common section file names
        candidates = ["SectionData.json", "sections.json", "MASTER_SECTION.json"]
        
        for candidate in candidates:
            path = os.path.join(folder_path, candidate)
            if os.path.exists(path):
                return path
                
        # Look in parent folders
        parent = Path(folder_path).parent
        for candidate in candidates:
            path = parent / candidate
            if path.exists():
                return str(path)
                
        # Default fallback
        return os.path.join(folder_path, "SectionData.json")