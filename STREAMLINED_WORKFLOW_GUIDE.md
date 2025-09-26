# Streamlined LinearObject Workflow - User Guide

This document provides a user guide for the streamlined LinearObject workflow that integrates SPOT_Filters with the LinearObject framework for building bridge structural elements.

## Quick Start

### Using the Streamlined Runners

All main runners (`run_linear.py`, `run_pier.py`, `run_foundation.py`) now support a streamlined workflow that automatically handles:
- Data loading and parsing
- Object configuration  
- Component validation
- Geometry generation

```bash
# Generic linear object creation (deck, pier, foundation)
python run_linear.py --axis axis.json --cross cross.json --obj objects.json \
                    --main mainstation.json --section section.json \
                    --obj-type DeckObject --out deck.html

# Pier-specific workflow
python run_pier.py --axis axis.json --cross cross.json --obj pier.json \
                   --main mainstation.json --section section.json \
                   --out pier.html

# Foundation-specific workflow  
python run_foundation.py --axis axis.json --cross cross.json --obj foundation.json \
                         --main mainstation.json --section section.json \
                         --out foundation.html
```

### Using the LinearObjectBuilder Directly

For programmatic use, the LinearObjectBuilder provides a fluent API:

```python
from linear_object_builder import LinearObjectBuilder

# Create builder
builder = LinearObjectBuilder(verbose=True)

# Load data from files
builder.load_from_files(
    axis_json="axis.json",
    cross_json="cross.json", 
    obj_json="objects.json",
    mainstation_json="mainstation.json",
    section_json="section.json"
)

# Create specific object
pier = builder.create_object("PierObject", name="Pier1")

# Build geometry
result = builder.build_geometry(
    pier, 
    twist_deg=0.0,
    plan_rotation_deg=0.0
)
```

### Using SPOT Folder Integration

For complete SPOT project folders:

```python
from spot_linear_integration import create_linear_objects_from_spot

# Load all objects from SPOT folder
objects = create_linear_objects_from_spot(
    master_folder="/path/to/SPOT/project",
    branch="MAIN",
    verbose=True
)

# Access created objects
decks = objects["DeckObject"]
piers = objects["PierObject"] 
foundations = objects["FoundationObject"]
```

## Architecture Overview

### Key Components

1. **LinearObjectBuilder**: Central class that simplifies object creation and configuration
2. **SpotLinearLoader**: Enhanced SpotLoader with direct LinearObject integration
3. **Integration Functions**: Convenience functions for common workflows

### Data Flow

```
JSON Files → LinearObjectBuilder → Configured LinearObject → 3D Geometry
    ↓              ↓                        ↓                     ↓
Component     Auto-configure         Validate              Build mesh
Parsing       with available         configuration         and traces
              resources
```

## Benefits of Streamlined Workflow

### For Users
- **Simplified CLI**: Fewer manual steps and clearer error messages
- **Automatic Configuration**: No need to manually wire components together
- **Consistent Interface**: Same pattern across all object types
- **Better Error Handling**: Clear validation and fallback mechanisms

### For Developers  
- **Reduced Boilerplate**: Less repetitive configuration code
- **Better Separation**: Clear boundaries between loading, configuration, and geometry
- **Easier Testing**: Simplified mocking and unit testing
- **Extensibility**: Easy to add new object types and features

## Object Construction Details

### Deck Construction
- Follows alignment axis with variable cross-sections
- Supports continuous geometry variation via axis variables
- Handles complex section sequencing along spans

### Pier Construction  
- Creates vertical axis from base position + height
- Supports top/bottom cross-section definitions
- Includes anchor positioning from deck or world coordinates

### Foundation Construction
- Positioning via axis + station + offsets
- Supports multiple foundation types (footings, pile caps, piles)
- Integrates with pier objects for coordination

## Backward Compatibility

The streamlined workflow maintains full backward compatibility:
- Existing CLI interfaces unchanged
- Legacy workflows still supported via fallback mechanisms
- Gradual migration path for existing scripts

## Error Handling

The streamlined workflow provides enhanced error handling:

```python
# Automatic fallback to legacy method
try:
    obj = builder.create_object("PierObject")
    result = builder.build_geometry(obj)
except Exception as e:
    print(f"Streamlined approach failed ({e}), falling back to legacy method...")
    # Legacy workflow continues automatically
```

## Configuration Validation

Objects are automatically validated during creation:
- Required axis presence
- Cross-section availability  
- Component compatibility
- Clear error messages for issues

## Advanced Usage

### Custom Object Creation

```python
# Create object with specific data
custom_data = {
    "Class": "PierObject",
    "name": "CustomPier",
    "axis_name": "MainAxis",
    "height_m": 15.0
}

pier = builder.create_object("PierObject", obj_data=custom_data)
```

### Batch Processing

```python
# Create all available objects
all_objects = builder.create_all_objects()

# Build geometry for all
results = {}
for obj_type, objects in all_objects.items():
    results[obj_type] = {}
    for obj in objects:
        name = getattr(obj, 'name', f"Unnamed_{id(obj)}")
        results[obj_type][name] = builder.build_geometry(obj)
```

### SPOT Integration

```python
from spot_linear_integration import SpotLinearLoader

# Load from SPOT folder structure
loader = SpotLinearLoader("/path/to/SPOT", "MAIN", verbose=True)
objects = loader.load_and_build_all()

# Build geometry for all objects
geometry_results = loader.build_geometry_for_all(
    twist_deg=0.0,
    plan_rotation_deg=0.0,
    station_cap=400
)
```

This streamlined workflow significantly reduces the complexity of working with SPOT data while maintaining the full power and flexibility of the LinearObject framework.