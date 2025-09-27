# SPOT VISO - Architecture Map

## Purpose
SPOT VISO is a 3D bridge structure visualization system that processes parametric geometric definitions to generate interactive Plotly-based visualizations. It transforms JSON-defined bridge components (decks, piers, foundations) with their cross-sections and centerline axes into world coordinates through section embedding algorithms and parallel transport frame computations.

## Module Structure

```
SPOT_VISO_Cleaned2/
├── models/                          # Core domain objects and algorithms
│   ├── axis.py                      # Centerline geometry, station interpolation, parallel transport frames
│   ├── cross_section.py             # Local Y,Z coordinate evaluation, world embedding via axis
│   ├── deck_object.py               # Bridge deck structural component (LinearObject subclass)
│   ├── pier_object.py               # Pier structural component (LinearObject subclass)
│   ├── foundation_object.py         # Foundation structural component (LinearObject subclass)
│   ├── linear_object.py             # Base class for extruded bridge components
│   ├── base.py                      # Data loading utilities, base object classes
│   ├── viso_context.py              # Object graph management and cross-references
│   ├── axis_variable.py             # Parametric variable evaluation along axes
│   ├── main_station.py              # Main station reference definitions
│   ├── plotter.py                   # 3D plotting configuration and utilities
│   ├── viso_ploter.py               # Local coordinate plotting and visualization
│   └── utils.py                     # Mathematical utilities, expression evaluation
├── spot_loader.py                   # Data aggregation from JSON sources, object instantiation
├── SPOT_Filters.py                  # JSON processing, filtering, and enrichment utilities
├── main.py                          # Interactive visualization entry point with full workflow
└── run_*.py                         # CLI entry points for specific object types
```

## Entry Points and Runners

### Primary Entry Points
- **`main.py`** - Interactive visualization mode with complete SpotLoader workflow, loads all object types
- **`run_linear.py`** - Unified CLI runner supporting all object types (DeckObject, PierObject, FoundationObject)

### Object-Specific Runners
- **`run_deck.py`** - Bridge deck visualization with rotation and frame control options
- **`run_pier.py`** - Pier structure analysis and rendering with plan rotation support
- **`run_foundation.py`** - Foundation geometry processing and visualization

### Test and Analysis Scripts
- **`test_*.py`** - Manual test scripts for data flow, rotation math, and spot loader validation
- **`analyze_rotation.py`** - Rotation transformation analysis utilities
- **`demo_*.py`** - Demonstration scripts for creative plotting and flexibility features

All runners support common parameters: `--axis`, `--cross`, `--obj`, `--main`, `--section`, `--out`, `--plan-rotation`, `--twist` for consistent data input and transformation control.