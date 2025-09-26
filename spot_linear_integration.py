"""
Enhanced SPOT integration for LinearObject workflow.

This module extends the existing SPOT_Filters and SpotLoader functionality
to provide seamless integration with the LinearObject framework.
"""

from __future__ import annotations
import os
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

# Import existing SPOT components
from SPOT_Filters import get_subfolders, get_json_files, load_json_objects, SpotJsonObject
from spot_loader import SpotLoader
from linear_object_builder import LinearObjectBuilder
from models.linear_object import LinearObject


class SpotLinearLoader(SpotLoader):
    """
    Enhanced SpotLoader that integrates directly with LinearObject workflow.
    
    This class extends SpotLoader to provide:
    - Direct LinearObject creation and configuration
    - Simplified batch processing of multiple objects  
    - Better integration with SPOT_Filters workflow
    - Unified interface for both file-based and folder-based loading
    """
    
    def __init__(self, master_folder: str, branch: str = "MAIN", verbose: bool = False):
        """
        Initialize with SPOT folder structure.
        
        Args:
            master_folder: Path to SPOT master folder
            branch: Branch subfolder to use (default: "MAIN") 
            verbose: Enable verbose logging
        """
        super().__init__(master_folder, branch, verbose)
        self._builder = LinearObjectBuilder(verbose=verbose)
        self._linear_objects: Dict[str, List[LinearObject]] = {}
        
    def load_and_build_all(self) -> Dict[str, List[LinearObject]]:
        """
        Load SPOT data and create all LinearObjects.
        
        Returns:
            Dictionary mapping object type to list of LinearObject instances
        """
        # Use parent class to load SPOT data
        self.build_all_with_context(verbose=self.verbose)
        
        # Create LinearObjects from loaded data
        self._linear_objects = {
            "DeckObject": self._create_linear_objects("DeckObject", self.deck_objects),
            "PierObject": self._create_linear_objects("PierObject", self.pier_objects),  
            "FoundationObject": self._create_linear_objects("FoundationObject", self.foundation_objects)
        }
        
        return self._linear_objects
    
    def get_linear_objects(self, obj_type: str) -> List[LinearObject]:
        """Get LinearObjects of specified type."""
        if not self._linear_objects:
            self.load_and_build_all()
        return self._linear_objects.get(obj_type, [])
    
    def get_object_by_name(self, obj_type: str, name: str) -> Optional[LinearObject]:
        """Get specific LinearObject by type and name."""
        objects = self.get_linear_objects(obj_type)
        for obj in objects:
            obj_name = getattr(obj, 'name', '') or getattr(obj, 'Name', '')
            if obj_name == name:
                return obj
        return None
    
    def build_geometry_for_all(self, **build_params) -> Dict[str, Dict[str, dict]]:
        """
        Build geometry for all loaded LinearObjects.
        
        Args:
            **build_params: Parameters to pass to build() method
            
        Returns:
            Dictionary mapping obj_type -> obj_name -> geometry_result
        """
        if not self._linear_objects:
            self.load_and_build_all()
            
        results = {}
        
        for obj_type, objects in self._linear_objects.items():
            results[obj_type] = {}
            for obj in objects:
                try:
                    obj_name = getattr(obj, 'name', '') or getattr(obj, 'Name', '') or f"Unnamed_{id(obj)}"
                    result = self._builder.build_geometry(obj, **build_params)
                    results[obj_type][obj_name] = result
                except Exception as e:
                    if self.verbose:
                        print(f"Failed to build geometry for {obj_type} '{obj_name}': {e}")
                        
        return results
    
    def create_builder(self) -> LinearObjectBuilder:
        """
        Create a LinearObjectBuilder configured with this loader's data.
        
        Returns:
            Configured LinearObjectBuilder instance
        """
        if not hasattr(self, '_loaded') or not self._loaded:
            self.build_all_with_context(verbose=self.verbose)
            
        # Create builder and load from SPOT folder
        builder = LinearObjectBuilder(verbose=self.verbose)
        builder.load_from_spot_folder(self.master_folder, self.branch)
        return builder
    
    # Private helper methods
    
    def _create_linear_objects(self, obj_type: str, spot_objects: List) -> List[LinearObject]:
        """Create LinearObject instances from loaded SPOT objects."""
        if not spot_objects:
            return []
            
        linear_objects = []
        
        for spot_obj in spot_objects:
            try:
                # Convert SPOT object to dict if needed
                if hasattr(spot_obj, 'to_dict'):
                    obj_data = spot_obj.to_dict()
                elif hasattr(spot_obj, '__dict__'):
                    obj_data = vars(spot_obj)
                else:
                    obj_data = dict(spot_obj) if isinstance(spot_obj, dict) else {}
                
                # Ensure Class field is set
                obj_data['Class'] = obj_type
                
                # Use builder to create configured LinearObject
                builder = self.create_builder()
                linear_obj = builder.create_object(obj_type, obj_data=obj_data)
                linear_objects.append(linear_obj)
                
            except Exception as e:
                if self.verbose:
                    name = getattr(spot_obj, 'name', getattr(spot_obj, 'Name', 'Unknown'))
                    print(f"Failed to create {obj_type} '{name}': {e}")
                    
        return linear_objects


def create_linear_objects_from_spot(master_folder: str, 
                                   branch: str = "MAIN",
                                   obj_types: Optional[List[str]] = None,
                                   verbose: bool = False) -> Dict[str, List[LinearObject]]:
    """
    Convenience function to create LinearObjects from SPOT folder.
    
    Args:
        master_folder: Path to SPOT master folder
        branch: Branch subfolder to use
        obj_types: List of object types to load (None for all)
        verbose: Enable verbose logging
        
    Returns:
        Dictionary mapping object type to list of LinearObject instances
    """
    if obj_types is None:
        obj_types = ["DeckObject", "PierObject", "FoundationObject"]
        
    loader = SpotLinearLoader(master_folder, branch, verbose)
    all_objects = loader.load_and_build_all()
    
    # Filter to requested types
    return {obj_type: objects for obj_type, objects in all_objects.items() if obj_type in obj_types}


def create_linear_objects_from_files(axis_json: str,
                                    cross_json: str,
                                    obj_json: str, 
                                    mainstation_json: str,
                                    section_json: str,
                                    obj_types: Optional[List[str]] = None,
                                    verbose: bool = False) -> Dict[str, List[LinearObject]]:
    """
    Convenience function to create LinearObjects from individual files.
    
    Args:
        axis_json: Path to axis JSON file
        cross_json: Path to cross section JSON file
        obj_json: Path to objects JSON file
        mainstation_json: Path to mainstation JSON file  
        section_json: Path to section data JSON file
        obj_types: List of object types to load (None for all)
        verbose: Enable verbose logging
        
    Returns:
        Dictionary mapping object type to list of LinearObject instances
    """
    if obj_types is None:
        obj_types = ["DeckObject", "PierObject", "FoundationObject"]
        
    builder = LinearObjectBuilder(verbose=verbose)
    builder.load_from_files(axis_json, cross_json, obj_json, mainstation_json, section_json)
    
    all_objects = builder.create_all_objects()
    
    # Filter to requested types
    return {obj_type: objects for obj_type, objects in all_objects.items() if obj_type in obj_types}