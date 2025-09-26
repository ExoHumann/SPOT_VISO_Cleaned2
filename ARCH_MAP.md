# ARCH_MAP.md

## Purpose & Domain
SPOT VISO is a parametric structural visualization and geometry computation system for bridge modeling and analysis. The system transforms JSON-based bridge component definitions (axes, cross-sections, deck/pier/foundation objects) into precise 3D geometric representations through coordinate embedding algorithms. It supports multi-branch data organization, complex parametric variable evaluation, station interpolation, and coordinate transformations including parallel transport frame calculations for accurate structural visualization in engineering workflows.

## Current Package/Module Layout

```
SPOT_VISO_Cleaned2/
├── models/                    # Core domain architecture
│   ├── axis.py               # Centerline geometry & frame transport algorithms
│   ├── cross_section.py      # Parametric section evaluation & coordinate computation  
│   ├── deck_object.py        # Bridge deck structural component with embedding
│   ├── pier_object.py        # Pier structural component with geometry computation
│   ├── foundation_object.py  # Foundation structural component
│   ├── viso_context.py       # Object graph & reference resolution context
│   ├── axis_variable.py      # Parametric variable evaluation with caching
│   ├── base.py              # Base classes & JSON loading utilities
│   ├── linear_object.py     # Shared linear object behavior & interfaces
│   ├── main_station.py      # Station reference objects
│   ├── mapping.py           # Field mapping configuration for JSON loading
│   ├── utils.py             # Math utilities & expression compilation
│   ├── plotter.py           # Visualization trace generation (Plotly)
│   └── __init__.py          # Package API with lazy loading to avoid cycles
├── spot_loader.py           # Data aggregation & object instantiation orchestrator
├── SPOT_Filters.py          # JSON processing & filtering utilities for multi-branch data
├── main.py                  # Interactive visualization entry point with compute orchestration
├── run_*.py                 # CLI runners for specific object types (deck, pier, foundation)
├── GIT/                     # Multi-branch data repository
│   ├── MAIN/                # Primary dataset (35 stations, -100mm to 201722mm)
│   ├── RCZ_new1/            # Alternative dataset (4 stations, 0-300000mm)
│   └── */                   # Additional project branches
└── MASTER_SECTION/          # Master cross-section JSON definitions
```

## Runtime Data Flow

```mermaid
graph TB
    subgraph "Data Layer"
        GIT[GIT/* Branches<br/>JSON Data Files]
        MASTER[MASTER_SECTION/<br/>Cross-section definitions]
    end
    
    subgraph "Loading & Processing"
        FILTERS[SPOT_Filters<br/>JSON scanning & filtering]
        LOADER[SpotLoader<br/>Aggregation & instantiation]
        CONTEXT[VisoContext<br/>Object graph wiring]
    end
    
    subgraph "Core Domain Objects"
        AXIS[Axis<br/>Centerline geometry<br/>Parallel transport frames]
        CROSS[CrossSection<br/>Parametric evaluation<br/>Local Y,Z coordinates]
        AXISVAR[AxisVariable<br/>Variable evaluation<br/>with LRU caching]
    end
    
    subgraph "Structural Components"
        DECK[DeckObject<br/>Bridge deck geometry]
        PIER[PierObject<br/>Pier geometry]
        FOUND[FoundationObject<br/>Foundation geometry]
    end
    
    subgraph "Computation Engine"
        EMBED[Embedding Algorithms<br/>World coordinate projection<br/>Station interpolation]
        GEOM[compute_object_geometry<br/>Dispatch to object.compute_geometry()]
    end
    
    subgraph "Visualization"
        PLOT[Plotly Traces<br/>3D mesh generation]
        HTML[HTML Output<br/>Interactive browser viz]
    end
    
    GIT --> FILTERS
    MASTER --> FILTERS
    FILTERS --> LOADER
    LOADER --> CONTEXT
    CONTEXT --> AXIS
    CONTEXT --> CROSS
    CONTEXT --> AXISVAR
    AXIS --> DECK
    AXIS --> PIER  
    AXIS --> FOUND
    CROSS --> DECK
    CROSS --> PIER
    CROSS --> FOUND
    AXISVAR --> EMBED
    DECK --> GEOM
    PIER --> GEOM
    FOUND --> GEOM
    GEOM --> EMBED
    EMBED --> PLOT
    PLOT --> HTML
```

## Principal Entry Points

### CLI Scripts
- **`main.py`** - Main visualization entry point, loads full project data and renders combined 3D scene
- **`run_deck.py`** - Deck-specific visualization with rotation controls and frame mode options  
- **`run_pier.py`** - Pier object visualization with station sampling and height controls
- **`run_foundation.py`** - Foundation object rendering and geometry validation
- **`run_linear.py`** - Generic linear object visualization for custom components

### Data Flow Through Entry Points
1. **Data Loading**: Entry points use `SpotLoader` to scan GIT branch folders via `SPOT_Filters`
2. **Context Building**: `SpotLoader.build_all_with_context()` creates `VisoContext` with object graph
3. **Object Instantiation**: Typed objects (Deck/Pier/Foundation) created with axis and cross-section references
4. **Geometry Computation**: `compute_object_geometry()` dispatches to object-specific `compute_geometry()` methods
5. **Visualization**: Plotly traces generated via `get_plot_traces_matrix()` and output as interactive HTML

## Key Domain Entities

### Core Geometry (`models/axis.py`)
- **`Axis`** - Centerline with stations, coordinates; provides parallel transport frames
  - Methods: `get_position_at_station()`, `parallel_transport_frames()`, `embed_section_points_world()`
  - Units: Internal storage in millimeters, input/output configurable

### Section Evaluation (`models/cross_section.py`)  
- **`CrossSection`** - Parametric section evaluator for local Y,Z coordinates
  - Methods: `evaluate()`, `compute_local_points()`, `compute_embedded_points()`
  - Handles: Expression compilation, variable scaling, unit normalization

### Structural Objects (`models/deck_object.py`, `models/pier_object.py`, `models/foundation_object.py`)
- **`DeckObject`**, **`PierObject`**, **`FoundationObject`** - Inherit from `LinearObject`
  - Methods: `compute_geometry()` - main computation pipeline
  - Properties: `axis_obj`, `cross_section_obj`, `axis_variables_obj` 
  - Output: Dictionary with `{"ids", "stations_mm", "points_mm", "local_Y_mm", "local_Z_mm", "loops_idx"}`

### Variable System (`models/axis_variable.py`)
- **`AxisVariable`** - Parametric variable evaluation with station-based interpolation
  - Methods: `evaluate()`, `evaluate_at_stations_cached()` (LRU cached)
  - Handles: Expression evaluation, spline interpolation, unit conversions

### Context Management (`models/viso_context.py`)
- **`VisoContext`** - Central object registry and reference resolver
  - Properties: `axes_by_name`, `crosssec_by_ncs`, `mainstations_by_name`
  - Methods: `from_json()` - factory method for building complete context from raw data

### Data Loading (`spot_loader.py`)
- **`SpotLoader`** - Orchestrates data loading and object instantiation
  - Methods: `load_raw()`, `group_by_class()`, `build_all_with_context()`
  - Provides: Type-safe object collections, context-aware object wiring