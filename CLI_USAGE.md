### `demo_flexibility.py`
Comprehensive demonstration of the modular architecture's flexibility with current rotation controls (plan yaw and twist).
```bash
python -u demo_flexibility.py
```
Generates multiple visualizations:
- MAIN data only (baseline and rotated)
- RCZ_new1 data only
- Mixed MAIN/RCZ_new1 sources

### `run_deck.py`
Tests DeckObject with different data sources.
```bash
# MAIN data (baseline)
python -u run_deck.py --out deck_main.html

# RCZ_new1 data (different axis and cross sections)
python -u run_deck.py --axis GIT/RCZ_new1/_Axis_JSON.json --cross GIT/RCZ_new1/_CrossSection_JSON.json --obj GIT/RCZ_new1/_DeckObject_JSON.json --main GIT/RCZ_new1/_MainStation_JSON.json --out deck_rcz_new1.html

# Plan yaw example (+45°)
python -u run_deck.py --plan-rotation 45 --out deck_plan45.html

# Twist example (+30°)
python -u run_deck.py --twist 30 --out deck_twist30.html

# Symmetric vs Parallel-Transport frames
python -u run_deck.py --frame-mode symmetric --out deck_frame_symmetric.html
python -u run_deck.py --frame-mode pt --out deck_frame_pt.html

# Absolute per-station rotations (override instead of add)
python -u run_deck.py --rotation-mode absolute --out deck_absolute_rot.html

# Combined: use per-station X/Z (StationRotationX / StationRotationZ) + global yaw
python -u run_deck.py --plan-rotation 10 --twist 5 --out deck_station_overrides.html
```

### `run_pier.py`
Tests PierObject with different data sources.
```bash
# MAIN data (baseline)
python -u run_pier.py --out pier_main.html

# RCZ_new1 data
python -u run_pier.py --axis GIT/RCZ_new1/_Axis_JSON.json --cross GIT/RCZ_new1/_CrossSection_JSON.json --obj GIT/RCZ_new1/_PierObject_JSON.json --main GIT/RCZ_new1/_MainStation_JSON.json --out pier_rcz_new1.html

# Plan yaw example (+30°) and twist (+15°)
python -u run_pier.py --plan-rotation 30 --twist 15 --out pier_plan30_twist15.html
```

### `run_foundation.py`
Tests FoundationObject with different data sources.
```bash
# MAIN data (baseline)
python -u run_foundation.py --out foundation_main.html

# Note: RCZ_new1 foundation data may need cross section configuration

# Plan yaw example (+30°)
python -u run_foundation.py --plan-rotation 30 --out foundation_plan30.html
```

## Demonstrated Flexibility

The scripts show how the modular architecture allows:

1. **Different Axes**: Objects can use different axis definitions
   - MAIN: Axis with ~35 stations (-100mm to ~201722mm)
   - RCZ_new1: Axis with 4 stations (0mm to 300000mm)

2. **Different Cross Sections**: Objects can use different cross section sets
   - MAIN: Various NCS values (111, 501, 4000, ...)
   - RCZ_new1: Additional NCS values (121, 3001, ...)

3. **Component Mixing**: Objects can be configured with components from different data sources

## Key Features

- **Modular Configuration**: Each object type has its own `configure()` method
- **Component Selection**: Objects choose appropriate components from available options
- **Data Source Flexibility**: Load axes, cross sections, and main stations from different JSON files
- **Type-Specific Logic**: Each object type implements its own configuration logic
- **Modern Rotation Controls**: Use `--plan-rotation` (XY yaw) and `--twist` (in-plane) instead of legacy FLIP90
- **Per-Station Rotations**: `StationRotationX` (twist) and `StationRotationZ` (plan yaw) from MainStation JSON rows are applied (additive or absolute)
- **Frame Modes**: `--frame-mode symmetric` (middle-plane tangent using s+δ/s-δ) or `--frame-mode pt` (parallel transport minimal twist)
- **Rotation Modes**: `--rotation-mode additive` (default: global + per-station) or `--rotation-mode absolute` (global acts as base, station overrides added to base not cumulative)

## Usage Examples

```bash
# Mix MAIN axis with RCZ_new1 cross sections
python -u run_deck.py --axis GIT/MAIN/_Axis_JSON.json --cross GIT/RCZ_new1/_CrossSection_JSON.json --out mixed_sources.html

# Use a specific master section file
python -u run_deck.py --section MASTER_SECTION/Kulosaari_MasterCrossSection.json --out custom_section.html

# Switch to PT frames and absolute station rotations
python -u run_deck.py --frame-mode pt --rotation-mode absolute --out deck_pt_abs.html
```

## Unified Runner (all objects)
Run any object type with one script by selecting `--obj-type`.
```bash
# Deck
python -u run_linear.py --axis GIT/MAIN/_Axis_JSON.json --cross GIT/MAIN/_CrossSection_JSON.json --obj GIT/MAIN/_DeckObject_JSON.json --main GIT/MAIN/_MainStation_JSON.json --section MASTER_SECTION/MASTER_DeckCrBm-1Gird-Slab.json --obj-type DeckObject --out deck_linear_main.html --plan-rotation 0 --twist 0

# Pier
python -u run_linear.py --axis GIT/MAIN/_Axis_JSON.json --cross GIT/MAIN/_CrossSection_JSON.json --obj GIT/MAIN/_PierObject_JSON.json --main GIT/MAIN/_MainStation_JSON.json --section MASTER_SECTION/MASTER_Pier.json --obj-type PierObject --out pier_linear_main.html --plan-rotation 0 --twist 0

# Foundation
python -u run_linear.py --axis GIT/MAIN/_Axis_JSON.json --cross GIT/MAIN/_CrossSection_JSON.json --obj GIT/MAIN/_FoundationObject_JSON.json --main GIT/MAIN/_MainStation_JSON.json --section MASTER_SECTION/MASTER_Foundation.json --obj-type FoundationObject --out foundation_linear_main.html --plan-rotation 0 --twist 0
```

## Creative Plots Batch
Generate many views in one go.
```bash
python -u demo_creative_plots.py
```
Outputs are saved in the project folder.

## Troubleshooting

- Different Python launcher/version:
```bash
py -3.12 run_linear.py --help
```
- Missing packages:
```bash
py -3.12 -m pip install plotly numpy
```
- Paths with spaces: wrap them in quotes
- OneDrive sync delay: output HTML may take a moment to sync
- Unexpected orientation? Try switching frame construction:
   - Symmetric better near abrupt curvature transitions.
   - Parallel transport better for smooth long curves minimizing twist accumulation.
- Per-station rotations not visible? Confirm fields `StationRotationX` / `StationRotationZ` exist (non-null) in your `_MainStation_JSON.json` and that station values match sampled stations (station capping/downsampling may skip them if outside domain).

## Help
List options for any runner, e.g.:
```bash
python -u run_linear.py --help
```
