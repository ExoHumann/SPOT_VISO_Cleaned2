# Streamlined LinearObject Workflow Architecture

## Overview
This document outlines the streamlined architecture for integrating SPOT_Filters with the LinearObject workflow, focusing on the main entry points: `run_pier.py`, `run_deck.py`, and `run_linear.py`.

## Core Architecture Principles

### 1. LinearObject as Universal Base
All structural elements (Deck, Pier, Foundation) inherit from LinearObject:
```python
class LinearObject(BaseObject):
    def configure(self, available_axes, available_cross_sections, available_mainstations)
    def build(self, stations_m, twist_deg, plan_rotation_deg) -> Dict
```

### 2. SPOT_Filters as Data Source
SPOT_Filters provides the data loading infrastructure:
```python
# Data loading workflow:
json_files = get_json_files(folder_path)
raw_objects = load_json_objects(json_files) 
typed_objects = SpotLoader._load_typed(ObjectClass, raw_data)
```

### 3. Unified Runner Interface
All runners follow the same pattern:
- **run_linear.py**: Generic LinearObject handler (supports all types)
- **run_pier.py**: Pier-specific workflow with optimizations
- **run_deck.py**: Deck-specific workflow with optimizations
- **run_foundation.py**: Foundation-specific workflow

## Streamlined Workflow

### Phase 1: Data Loading
```python
# Common pattern across all runners:
axis_rows = json.load(open(axis_json))
cross_rows = json.load(open(cross_json))
obj_rows = json.load(open(obj_json))
ms_rows = json.load(open(mainstation_json))
```

### Phase 2: Component Preparation
```python
# Parse available components:
available_axes = {name: load_axis_from_rows(axis_rows, name) for name in axis_names}
available_cross_sections = {ncs: load_section_for_ncs(ncs, by_ncs, section_json) 
                          for ncs in by_ncs.keys()}
available_mainstations = {name: load_mainstations_from_rows(ms_rows, name) 
                        for name in available_axes.keys()}
```

### Phase 3: Object Creation & Configuration
```python
# Create object from JSON using mapping:
obj = from_dict(ObjectClass, obj_row, mapping)

# Configure with available components:
obj.configure(available_axes, available_cross_sections, available_mainstations)
```

### Phase 4: Geometry Generation
```python
# Build 3D geometry:
result = obj.build(
    stations_m=None,
    twist_deg=twist_deg,
    plan_rotation_deg=plan_rotation_deg,
    station_cap=max_stations
)
```

### Phase 5: Visualization
```python
# Generate plot traces and output:
traces, *_ = get_plot_traces_matrix(
    result["axis"], result["section_json"], result["stations_mm"],
    loop_stride=loop_stride, long_stride=long_stride
)
```

## Architecture Improvements

### Current State Analysis

#### Strengths
- **LinearObject Framework**: Solid foundation for all structural elements
- **SPOT_Filters**: Robust data loading from JSON files
- **Multiple Entry Points**: Specialized runners for different use cases
- **Consistent Interface**: configure() and build() pattern across objects

#### Areas for Improvement
- **Configuration Complexity**: Too many manual steps in setup
- **Code Duplication**: Similar logic repeated across runners  
- **Error Handling**: Limited validation of component compatibility
- **Integration Gaps**: SPOT_Filters and LinearObject not fully integrated

### Proposed Streamlining

#### 1. Unified LinearObjectBuilder
Create a builder class that simplifies the workflow:

```python
class LinearObjectBuilder:
    def __init__(self, data_root: str):
        self.data_root = data_root
        self._axes = {}
        self._cross_sections = {}
        self._mainstations = {}
    
    def load_data(self, axis_json, cross_json, mainstation_json, section_json):
        """Load and parse all component data"""
        # Centralized loading logic
        pass
    
    def create_object(self, obj_type: str, obj_json: str):
        """Create and configure LinearObject from JSON"""
        # Unified object creation
        pass
    
    def build_geometry(self, obj, **build_params):
        """Generate 3D geometry with standard parameters"""
        # Standardized build process
        pass
```

#### 2. Enhanced SPOT_Filters Integration
Extend SPOT_Filters to work directly with LinearObject:

```python
class SpotLinearLoader(SpotLoader):
    def create_linear_object(self, obj_type: str, name: str = None):
        """Create configured LinearObject from loaded SPOT data"""
        # Direct integration with LinearObject workflow
        pass
    
    def build_all_linear_objects(self):
        """Build geometry for all loaded objects"""
        # Batch processing of multiple objects
        pass
```

#### 3. Configuration Automation
Reduce manual configuration steps:

```python
class LinearObject:
    def auto_configure(self, available_components: Dict):
        """Automatically configure from available components"""
        # Smart configuration based on object data
        pass
    
    def validate_configuration(self):
        """Validate that configuration is complete and compatible"""
        # Enhanced error checking
        pass
```

## Implementation Plan

### Phase A: Documentation & Analysis (Current)
- [x] Create architectural documentation (this file)
- [x] Document pier construction workflow
- [x] Document deck construction workflow  
- [x] Document foundation construction workflow
- [ ] Analyze current code for streamlining opportunities

### Phase B: Core Streamlining
- [ ] Create LinearObjectBuilder class
- [ ] Enhance SPOT_Filters integration
- [ ] Add configuration validation
- [ ] Implement auto-configuration logic

### Phase C: Runner Refactoring
- [ ] Update run_linear.py to use LinearObjectBuilder
- [ ] Update run_pier.py with streamlined workflow
- [ ] Update run_deck.py with streamlined workflow
- [ ] Update run_foundation.py with streamlined workflow

### Phase D: Testing & Validation
- [ ] Create comprehensive tests for streamlined workflow
- [ ] Validate backward compatibility
- [ ] Performance testing and optimization
- [ ] Documentation updates

## Benefits of Streamlined Workflow

### For Developers
- **Reduced Boilerplate**: Less repetitive configuration code
- **Better Error Handling**: Automatic validation and clear error messages
- **Consistent Interface**: Same patterns across all object types
- **Easier Testing**: Simplified mocking and unit testing

### For Users  
- **Simpler CLI**: Fewer required parameters with better defaults
- **Better Error Messages**: Clear feedback when configuration fails
- **Faster Iteration**: Quicker setup for common use cases
- **More Reliable**: Fewer opportunities for configuration errors

### For Architecture
- **Better Separation**: Clear boundaries between data loading, configuration, and geometry
- **Extensibility**: Easier to add new object types and features
- **Maintainability**: Less code duplication and clearer responsibilities
- **Integration**: Better coordination between SPOT_Filters and LinearObject systems

## Migration Strategy

### Backward Compatibility
- Keep existing runner interfaces unchanged
- Implement streamlined workflow as optional enhancement
- Provide migration path for existing scripts

### Incremental Adoption
- Start with run_linear.py as most generic
- Extend to specialized runners (pier, deck, foundation)
- Allow opt-in to new features

### Testing Strategy
- Comprehensive regression testing
- Performance benchmarking
- Integration testing with real SPOT data

This streamlined architecture maintains the flexibility of the current system while reducing complexity and improving reliability. The LinearObject framework provides a solid foundation, and SPOT_Filters integration can be enhanced to provide a more seamless workflow for building bridge structural elements.