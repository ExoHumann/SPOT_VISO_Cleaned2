# SPOT VISO Code Contracts

This document defines authoritative statements that the code must uphold. These are invariants and conventions that must remain consistent across all refactoring.

## Units Convention

**Internal Storage**: All geometric values are stored internally in **millimeters (mm)**
- `Axis` class stores stations, coordinates in mm via constructor conversion factor
- `CrossSection` coordinates and variables are converted to mm via `_fix_var_units_inplace()`
- Station arrays (`stations_mm`) are always in mm
- Local section points (`yz_points_mm`) are in mm

**Display/Plotting**: All plotting outputs convert to **meters (m)**
- `Plotter._to_m()` divides by 1000.0 for display
- Plotly traces show coordinates in meters for human readability
- Hover text displays: `'X: %{customdata[2]:.3f} m'` etc.

**External APIs**: Mixed depending on context
- `Axis.point_at_m()` and `clamp_station_m()` accept/return meters
- Command line `--plan-rotation` and `--twist` are in degrees
- JSON loading may be meters (converted via units="m" parameter)

## Frame Coordinate System

**Axis Frame Components**: `models/axis.py`
- `T` (Tangent): Unit tangent along axis direction
- `N` (Normal): Unit normal (perpendicular to T, in "horizontal" plane when possible)  
- `B` (Binormal): Unit binormal = T × N (typically "vertical" direction)
- **CONTRACT**: All frame vectors are **unit-norm**: `||T|| = ||N|| = ||B|| = 1.0`

**Frame Modes**: `Axis.frame_at_stations(mode=...)`
- `"pt"`: Parallel transport frames (minimal twist accumulation)
- `"symmetric"`: Middle-plane tangent estimation using `P(s+δ) - P(s-δ)`

**Local Coordinate System**:
- X-axis: Along tangent T (forward direction)
- Y-axis: Along normal N (typically horizontal)
- Z-axis: Along binormal B (typically vertical)

## World Array Shapes

**Embedded World Coordinates**: `Axis.embed_section_points_world()` returns
- **Shape**: `(Nstations, Npoints, 3)` 
- `W[i, k, :]` = world coordinates of point k at station i
- Last dimension: `[x, y, z]` in world coordinates

**Station Arrays**:
- **Shape**: `(Nstations,)` - 1D array of station values in mm

**Section Points**:
- **Shape**: `(Npoints, 2)` or `(Nstations, Npoints, 2)`
- Local YZ coordinates in mm
- If 2D `(Npoints, 2)`: broadcast to all stations
- If 3D: station-specific section shapes

## Supported Frame Modes

**Parallel Transport (`"pt"`)**: `models/axis.py:parallel_transport_frames()`
- Minimal twist accumulation along axis
- Uses Rodrigues rotation for frame transport
- Best for smooth, long curves
- Implemented via rotation-minimizing frame algorithm

**Symmetric (`"symmetric"`)**: `models/axis.py:frame_at_stations()`
- Middle-plane tangent estimation
- Uses neighbor points: `P(s+δ) - P(s-δ)` for tangent
- Parameters: `symmetric_frac_of_neighbor_gap`, `symmetric_delta_min_mm`, `symmetric_delta_max_mm`
- Better for abrupt curvature transitions

## Cross Section Contracts

**CrossSection Evaluation**: `models/cross_section.py:CrossSection.evaluate()`
- **Input**: Variable arrays in any units (auto-detected/converted to mm)
- **Output**: `(ids, X, Y, loops_idx)` where X,Y are in mm
- **Units Auto-Detection**: Compares with defaults via `_fix_var_units_inplace()`

**Point Topology**: Cross sections maintain:
- Named point IDs for referencing
- Loop connectivity for closed regions
- Topological ordering for evaluation dependencies

## Rotation Contracts

**Plan Rotation**: Around global Z-axis (yaw)
- Applied via `plan_rotation_deg` parameter
- Affects XY plane orientation only
- Rotates N/B frame vectors around Z

**In-Plane Rotation**: Within local YZ cross-section plane  
- Applied via `rotation_deg` parameter (twist)
- Rotates section within its local plane
- Combined with plan rotation for full 3D orientation

**Per-Station Rotations**: From MainStation JSON
- `StationRotationX`: twist (in-plane rotation)
- `StationRotationZ`: plan yaw
- Applied additively or absolutely based on rotation mode

## Object Type Contracts

**DeckObject**: `models/deck_object.py`
- Multiple cross sections along stations
- Support for station-specific section selection
- Axis variables for parametric evaluation

**PierObject**: `models/pier_object.py`  
- Top/bottom cross section configuration
- Internal stations for intermediate sections
- Y/Z offsets for section positioning

**FoundationObject**: `models/foundation_object.py`
- Foundation-level positioning
- Reference point offsets
- Bearing/pile configuration

## Data Source Contracts

**JSON Loading**: Via `SpotLoader` and `models/mapping.py`
- Type-specific field mapping from JSON keys to dataclass fields
- Automatic unit detection and conversion
- Fallback mechanisms for missing/optional fields

**File Paths**: Relative to repository root
- `GIT/MAIN/_*_JSON.json`: Main dataset
- `GIT/RCZ_new1/_*_JSON.json`: Alternative dataset  
- `MASTER_SECTION/*.json`: Cross section definitions

## Invariants to Maintain

1. **Unit Consistency**: Internal mm, display m, clear conversion boundaries
2. **Frame Orthonormality**: T⊥N⊥B, all unit vectors
3. **Array Shape Contracts**: World arrays always (N,M,3), station arrays (N,)
4. **Mode Support**: Both "pt" and "symmetric" frame modes must work
5. **Cross Section Topology**: Named points, loops, dependency order preserved
6. **Object Configuration**: Each object type maintains its specific setup contract