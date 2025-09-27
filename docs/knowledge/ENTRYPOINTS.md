# SPOT VISO Entry Points

This document catalogs the exact commands, runners, CLIs, and scripts available in the system along with their intended responsibilities.

## Primary Object Runners

### `run_deck.py`
**Purpose**: Test and visualize DeckObject with configurable data sources
**Responsibility**: Deck-specific multi-station cross section visualization

**Default Command**:
```bash
python -u run_deck.py --out deck_main.html
```

**Key Parameters**:
- `--axis PATH`: Axis JSON file (default: `GIT/MAIN/_Axis_JSON.json`)
- `--cross PATH`: CrossSection JSON file (default: `GIT/MAIN/_CrossSection_JSON.json`)  
- `--obj PATH`: DeckObject JSON file (default: `GIT/MAIN/_DeckObject_JSON.json`)
- `--main PATH`: MainStation JSON file (default: `GIT/MAIN/_MainStation_JSON.json`)
- `--section PATH`: Master section file (default: `MASTER_SECTION/SectionData.json`)
- `--out FILE`: Output HTML file
- `--plan-rotation DEG`: Plan yaw rotation in degrees  
- `--twist DEG`: In-plane twist rotation in degrees
- `--frame-mode MODE`: Frame mode (`pt` or `symmetric`)
- `--rotation-mode MODE`: Rotation application (`additive` or `absolute`)

**Example Variations**:
```bash
# Alternative data source
python -u run_deck.py --axis GIT/RCZ_new1/_Axis_JSON.json --cross GIT/RCZ_new1/_CrossSection_JSON.json --obj GIT/RCZ_new1/_DeckObject_JSON.json --main GIT/RCZ_new1/_MainStation_JSON.json --out deck_rcz_new1.html

# With rotations
python -u run_deck.py --plan-rotation 45 --twist 30 --out deck_rotated.html

# Symmetric frame mode  
python -u run_deck.py --frame-mode symmetric --out deck_symmetric.html
```

### `run_pier2.py` 
**Purpose**: Test and visualize PierObject (current version)
**Responsibility**: Pier-specific top/bottom cross section configuration

**Default Command**:
```bash
python -u run_pier2.py --out pier_main.html
```

**Parameters**: Similar to run_deck.py but pier-specific configuration
- Supports top/bottom cross section NCS specification
- Internal station configuration
- Y/Z offset parameters

### `run_foundation.py`
**Purpose**: Test and visualize FoundationObject  
**Responsibility**: Foundation-specific geometry and positioning

**Default Command**:
```bash
python -u run_foundation.py --out foundation_main.html
```

**Note**: RCZ_new1 foundation data may require specific cross section configuration

## Unified Runner

### `run_linear.py`
**Purpose**: Unified runner for any object type
**Responsibility**: Single entry point for all object types with type selection

**Usage Pattern**:
```bash
python -u run_linear.py --axis AXIS_PATH --cross CROSS_PATH --obj OBJ_PATH --main MAIN_PATH --section SECTION_PATH --obj-type OBJECT_TYPE --out OUTPUT.html
```

**Object Type Examples**:
```bash
# Deck
python -u run_linear.py --axis GIT/MAIN/_Axis_JSON.json --cross GIT/MAIN/_CrossSection_JSON.json --obj GIT/MAIN/_DeckObject_JSON.json --main GIT/MAIN/_MainStation_JSON.json --section MASTER_SECTION/MASTER_DeckCrBm-1Gird-Slab.json --obj-type DeckObject --out deck_linear_main.html

# Pier  
python -u run_linear.py --obj-type PierObject --section MASTER_SECTION/MASTER_Pier.json --out pier_linear_main.html

# Foundation
python -u run_linear.py --obj-type FoundationObject --section MASTER_SECTION/MASTER_Foundation.json --out foundation_linear_main.html
```

## Demonstration Scripts

### `demo_flexibility.py`
**Purpose**: Comprehensive demonstration of modular architecture flexibility
**Responsibility**: Generate multiple visualizations showing data source mixing capabilities

**Command**:
```bash
python -u demo_flexibility.py
```

**Outputs**: Multiple HTML files demonstrating:
- MAIN data baseline and rotated
- RCZ_new1 data variants  
- Mixed MAIN/RCZ_new1 source combinations

### `demo_creative_plots.py`
**Purpose**: Batch generation of multiple visualization variants
**Responsibility**: Create many plot variations in one execution

**Command**:
```bash
python -u demo_creative_plots.py  
```

**Output**: Multiple HTML files saved in project folder

## Legacy/Alternative Runners

### `run_pier2_spotjson.py`
**Purpose**: Pier runner with SPOT JSON configuration
**Responsibility**: Alternative pier configuration approach

### `run_deck_pier_simple.py` 
**Purpose**: Simplified deck-pier combination runner
**Responsibility**: Basic dual-object visualization

### `run_deck_pier_demo.py`
**Purpose**: Demo script (currently empty placeholder)

## Analysis and Testing Scripts

### `analyze_rotation.py`
**Purpose**: Rotation analysis and debugging
**Responsibility**: Examine rotation behavior and frame calculations
**Usage**: Development/debugging tool for rotation mechanics

### `debug_rotation.py`  
**Purpose**: Rotation debugging utilities
**Responsibility**: Debug rotation-related issues

### `check_coords.py`
**Purpose**: Coordinate system validation
**Responsibility**: Verify coordinate transformations and frame calculations

## Testing Scripts

### `test_rotation_comprehensive.py`
**Purpose**: Comprehensive rotation testing
**Responsibility**: Validate all rotation modes and combinations

### `test_rotation_math.py`
**Purpose**: Mathematical rotation testing  
**Responsibility**: Test rotation matrix calculations and frame math

### `test_actual_data.py`
**Purpose**: Real data validation testing
**Responsibility**: Test with actual project data files

### `test_data_flow.py`
**Purpose**: Data flow validation
**Responsibility**: Test data loading and processing pipelines

### `test_spotloaer.py`
**Purpose**: SpotLoader testing
**Responsibility**: Test JSON loading and type mapping functionality

## Core Application

### `main.py`
**Purpose**: Main application entry point  
**Responsibility**: Primary SPOT VISO application interface
**Note**: Contains legacy PyInstaller configuration for standalone builds

## Data Loading Utilities

### `spot_loader.py` 
**Purpose**: JSON data loading and type mapping
**Responsibility**: Core data loading infrastructure
**Key Classes**: `SpotLoader`
**Mapping**: Defined in `models/mapping.py`

### `SPOT_Filters.py`
**Purpose**: File system utilities and JSON loading helpers
**Responsibility**: Directory scanning, JSON file loading
**Functions**: `get_subfolders()`, `get_json_files()`, `load_json_objects()`

## Standard Data Paths

**Main Dataset**:
- Axis: `GIT/MAIN/_Axis_JSON.json`
- CrossSection: `GIT/MAIN/_CrossSection_JSON.json`  
- DeckObject: `GIT/MAIN/_DeckObject_JSON.json`
- PierObject: `GIT/MAIN/_PierObject_JSON.json`
- FoundationObject: `GIT/MAIN/_FoundationObject_JSON.json`
- MainStation: `GIT/MAIN/_MainStation_JSON.json`

**Alternative Dataset (RCZ_new1)**:
- Same structure under `GIT/RCZ_new1/` 

**Master Sections**:
- `MASTER_SECTION/SectionData.json`
- `MASTER_SECTION/MASTER_DeckCrBm-1Gird-Slab.json`
- `MASTER_SECTION/MASTER_Pier.json`  
- `MASTER_SECTION/MASTER_Foundation.json`

## Common Parameter Patterns

**Data Source Selection**:
```bash
--axis PATH --cross PATH --obj PATH --main PATH --section PATH
```

**Rotation Control**:
```bash
--plan-rotation DEGREES --twist DEGREES --frame-mode MODE --rotation-mode MODE
```

**Output Control**:
```bash  
--out FILENAME.html
```

**Help Access**:
```bash
python -u SCRIPT_NAME.py --help
```

## Script Responsibilities Summary

| Script | Object Type | Purpose | Output |
|--------|-------------|---------|---------|
| `run_deck.py` | Deck | Deck visualization | Single HTML |
| `run_pier2.py` | Pier | Pier visualization | Single HTML |  
| `run_foundation.py` | Foundation | Foundation visualization | Single HTML |
| `run_linear.py` | Any | Unified runner | Single HTML |
| `demo_flexibility.py` | Multiple | Architecture demo | Multiple HTML |
| `demo_creative_plots.py` | Multiple | Batch plotting | Multiple HTML |
| `main.py` | Any | Main application | Interactive |
| `analyze_rotation.py` | N/A | Analysis | Debug output |
| `test_*.py` | N/A | Testing | Test results |