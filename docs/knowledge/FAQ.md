# SPOT VISO Frequently Asked Questions

This document addresses common confusions and questions that arise when working with the SPOT VISO codebase.

## Q1: What units are used throughout the system and when do conversions happen?

**A1**: The system uses a mixed unit approach with clear conversion boundaries:

- **Internal storage**: All geometric values in millimeters (mm)
  - `Axis` coordinates, `CrossSection` points, station arrays are in mm
  - Conversion happens at data loading via `units="m"` parameter or `_fix_var_units_inplace()`
  
- **Display/Plotting**: Converted to meters (m) for human readability  
  - `Plotter._to_m()` divides by 1000.0 before sending to plotly
  - Hover text shows: "X: 123.456 m"
  
- **Command line inputs**: Context-dependent
  - `--plan-rotation`, `--twist` are in degrees
  - Station values from JSON may be in meters (auto-converted)

**Rule of thumb**: If you see `_mm` suffix, it's millimeters. If you see `_m` suffix, it's meters.

## Q2: What coordinate system and axis naming convention is used?

**A2**: The system uses a right-handed coordinate system with:

- **World coordinates**: X, Y, Z (global reference frame)
- **Local frame along axis**: T, N, B (Tangent, Normal, Binormal)
  - T: Forward along axis (local X)  
  - N: Perpendicular to T, typically horizontal (local Y)
  - B: T × N, typically vertical (local Z)
  
- **Cross section coordinates**: Y, Z in local frame
  - Y corresponds to local N direction
  - Z corresponds to local B direction
  
**Important**: All frame vectors (T, N, B) are **unit normalized**.

## Q3: Where do cross sections come from and how are they selected?

**A3**: Cross sections come from multiple sources with a hierarchy:

1. **Master sections**: `MASTER_SECTION/*.json` files contain the geometric definitions
2. **Object configuration**: JSON files specify which sections to use via:
   - `CrossSection@NCS` numbers (numeric identifiers)
   - `CrossSection@Name` strings (named references)  
   - `CrossSection@Type` categories

3. **Selection priority**:
   - Per-station schedule (most flexible)
   - Per-station name/NCS arrays  
   - Constant fallback from first available

**File paths**: 
- `models/cross_section.py`: Evaluation engine
- `models/mapping.py`: JSON field mapping
- `models/base.py`: Selection logic

## Q4: What's the difference between "pt" and "symmetric" frame modes?

**A4**: These are two different algorithms for computing reference frames along the axis:

- **"pt" (Parallel Transport)**: `models/axis.py:parallel_transport_frames()`
  - Minimal twist accumulation using rotation-minimizing frames
  - Best for smooth, long curves
  - Uses Rodrigues rotation for mathematical stability
  
- **"symmetric"**: `models/axis.py:frame_at_stations()` with middle-plane tangent
  - Estimates tangent using neighbor points: `T = P(s+δ) - P(s-δ)`
  - Better for abrupt curvature transitions  
  - Configurable delta range via `symmetric_delta_min_mm`, `symmetric_delta_max_mm`

**Usage**: Set via `--frame-mode pt` or `--frame-mode symmetric` in runners.

## Q5: How do rotations work and what's the difference between plan rotation and twist?

**A5**: Two types of rotations are applied in sequence:

1. **Plan Rotation** (`--plan-rotation`): 
   - Yaw around global Z-axis
   - Changes bridge orientation in XY plane (plan view)
   - Affects N/B frame vectors only, not T
   
2. **In-plane Rotation/Twist** (`--twist`):
   - Rotation within the local YZ cross-section plane
   - Changes how the section is oriented relative to the local frame
   - Applied after plan rotation

**Per-station overrides**: MainStation JSON can specify:
- `StationRotationX`: per-station twist 
- `StationRotationZ`: per-station plan yaw
- Applied additively or absolutely based on `--rotation-mode`

## Q6: What does the shape (Nstations, Npoints, 3) mean for world arrays?

**A6**: This is the standard shape for embedded world coordinates:

- **Nstations**: Number of stations along the axis where sections are placed
- **Npoints**: Number of points in each cross section  
- **3**: X, Y, Z world coordinates

**Access pattern**: `W[i, k, :]` gives world position of point k at station i.

**Source**: `Axis.embed_section_points_world()` returns arrays in this shape.

## Q7: Where is overlap detection and collision handling implemented?

**A7**: Currently, overlap/collision detection is **not implemented** as a core feature in the current codebase. The system focuses on:

- Geometric visualization and positioning
- Cross section placement along axes  
- Frame calculations and rotations

**Future implementation** would likely belong in:
- `models/collision.py` (new module)
- Object-level methods for self-intersection checks
- Integration with plotting for collision visualization

## Q8: How do I switch between different data sources (MAIN vs RCZ_new1)?

**A8**: Use the path parameters to specify alternative data sources:

**Default (MAIN)**:
```bash  
python -u run_deck.py --out deck_main.html
```

**Alternative (RCZ_new1)**:
```bash
python -u run_deck.py --axis GIT/RCZ_new1/_Axis_JSON.json --cross GIT/RCZ_new1/_CrossSection_JSON.json --obj GIT/RCZ_new1/_DeckObject_JSON.json --main GIT/RCZ_new1/_MainStation_JSON.json --out deck_rcz.html
```

**Mixed sources** (MAIN axis + RCZ_new1 sections):
```bash
python -u run_deck.py --axis GIT/MAIN/_Axis_JSON.json --cross GIT/RCZ_new1/_CrossSection_JSON.json --out mixed.html
```

## Q9: Why do I get unit mismatch warnings and how do I fix them?

**A9**: Unit auto-detection in `CrossSection._fix_var_units_inplace()` sometimes struggles to determine if input data is in meters or millimeters.

**Debug**: Enable debug output:
```python
_CS_DEBUG_UNITS = True  # in models/cross_section.py
```

**Force units**: Override auto-detection:
```python
cs.evaluate(var_arrays, force_var_scale=1000.0)  # Force mm from m
```

**Root cause**: Input values too close to defaults or inconsistent magnitude patterns.

## Q10: What's the difference between DeckObject, PierObject, and FoundationObject?

**A10**: These represent different structural elements with distinct configuration needs:

- **DeckObject** (`models/deck_object.py`):
  - Multiple cross sections along multiple stations
  - Continuous deck structure  
  - Station-specific section selection

- **PierObject** (`models/pier_object.py`):
  - Top/bottom cross section configuration
  - Internal stations for intermediate geometry
  - Y/Z offsets for precise positioning
  - Single-station primary placement
  
- **FoundationObject** (`models/foundation_object.py`):
  - Foundation-level positioning
  - Reference point offsets (X/Y)
  - Bearing and pile configuration
  - Ground interface geometry

## Q11: How do axis variables work and where are they evaluated?

**A11**: Axis variables provide parametric input to cross section evaluation:

**Definition**: `models/axis_variable.py` - station-dependent variable values
**Usage**: Cross sections reference variables like `H_QS`, `B_TR` in their point expressions
**Evaluation**: `CrossSection.evaluate()` receives `var_arrays_all` dict mapping variable names to station arrays

**Flow**:
1. Load axis variables from JSON via `SpotLoader`  
2. Object configuration specifies which variables to use
3. Cross section expressions reference variables: `"Coord": ["0", "H_QS"]`
4. Evaluation substitutes actual values per station

## Q12: What do the loop indices represent in cross section evaluation?

**A12**: Loop indices define closed regions within cross sections:

**Purpose**: Identify which points form closed loops (boundaries of material regions)
**Format**: List of numpy arrays, each containing point indices forming a loop
**Usage**: 
- Plotting closed section boundaries
- Material region definition
- Area calculations
- Finite element mesh generation

**Source**: `CrossSection.evaluate()` returns `loops_idx` as fourth return value
**Example**: `loops_idx[0] = [0, 1, 2, 3]` means points 0→1→2→3→0 form first loop.

**Implementation**: `models/cross_section.py:_loops_idx()` extracts from JSON section data.