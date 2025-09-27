# SPOT VISO - Call Graph & Data Flow

This document outlines the key function call relationships and data flow paths in SPOT VISO. Function signatures are automatically validated by `scripts/check_docs_sync.py`.

## Core Processing Pipeline

### 1. Data Loading Flow
```
spot_loader.SpotLoader.load_raw()
  └── SPOT_Filters.load_json_objects()
      └── load_axis_from_rows() from models/base.py
      └── index_cross_sections_by_ncs() from models/base.py  
      └── load_section_for_ncs() from models/base.py
```

**Key Function Signatures:**
```python
# models/base.py
def load_axis_from_rows(axis_rows: List[Dict], axis_name: str) -> Axis
def index_cross_sections_by_ncs(cross_rows: List[Dict]) -> Dict[int, Dict]
def load_section_for_ncs(ncs: int, by_ncs: Dict[int, Dict], fallback_path: str) -> CrossSection
```

### 2. Object Configuration Flow  
```
{DeckObject|PierObject|FoundationObject}.configure()
  └── configure() methods in respective object classes
```

**Key Function Signatures:**
```python
# models/deck_object.py
def configure(self, available_axes: Dict[str, Axis], available_cross_sections: Dict[int, CrossSection], available_mainstations: Dict[str, List[MainStationRef]], axis_name: Optional[str] = None, cross_section_ncs: Optional[List[int]] = None, mainstation_name: Optional[str] = None) -> None

# models/pier_object.py  
def configure(self, available_axes: Dict[str, Axis], available_cross_sections: Dict[int, CrossSection], available_mainstations: Dict[str, List], axis_name: Optional[str] = None, cross_section_ncs: Optional[List[int]] = None, mainstation_name: Optional[str] = None) -> None

# models/foundation_object.py
def configure(self, available_axes: Dict[str, Axis], available_cross_sections: Dict[int, CrossSection], available_mainstations: Dict[str, List[MainStationRef]], axis_name: Optional[str] = None, cross_section_ncs: Optional[List[int]] = None, mainstation_name: Optional[str] = None) -> None
```

### 3. Geometry Computation Flow
```
LinearObject.build()
  └── interp_axis_variables() from models/linear_object.py
  └── compute_embedded_points() from models/cross_section.py
      └── embed_section_points_world() from models/axis.py
          └── parallel_transport_frames() from models/axis.py
          └── get_position_at_station() from models/axis.py
```

**Key Function Signatures:**
```python
# models/linear_object.py
def interp_axis_variables(axis_var_rows: List[Dict], section_defaults_mm: Dict[str, float], stations_m: List[float]) -> Dict[str, np.ndarray]

# models/cross_section.py  
def compute_embedded_points(self) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]

# models/axis.py
def embed_section_points_world(self, stations_mm: np.ndarray, yz_points_mm: np.ndarray, x_offsets_mm: np.ndarray | None = None, rotation_deg: float | np.ndarray = 0.0, plan_rotation_deg: float | np.ndarray = 0.0) -> np.ndarray
def parallel_transport_frames(self, stations_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]
def get_position_at_station(self, station: float) -> Tuple[float, float, float]
```

### 4. Validation & Safety Checks
```python
# Core validation functions
def validate(self) -> None  # Found in Axis and AxisVariable classes
def safe_eval_scalar(expr: str, vars_: Dict[str, float]) -> float  # models/utils.py
```

### 5. Visualization Pipeline
```
Plotter class methods (called by main.py and run_*.py scripts):
  └── models/plotter.py:build_axis_trace(self, traces: List)
  └── models/plotter.py:build_overlay_traces(self, traces: List, overlays: Optional[List[Dict]])
  └── models/plotter.py:build_first_station_points(self, traces: List)
  └── models/plotter.py:build_loop_traces(self, traces: List)
  └── models/plotter.py:build_longitudinal_traces(self, traces: List)
```

## Critical Dependencies

### Axis Class (Core Geometry Engine)
- **Called by**: CrossSection.compute_embedded_points()

**Key methods:**
```python
def point_at_m(self, s_m: float) -> np.ndarray
def get_position_at_station(self, station: float) -> Tuple[float, float, float]
```

### CrossSection Class (Section Evaluation)
- **Called by**: All *Object classes during build()

**Key methods:**
```python
def from_file(cls, path: str, name: Optional[str] = None) -> 'CrossSection'
def defaults_mm(self) -> Dict[str, float]
def evaluate(self, var_arrays_all: Dict[str, np.ndarray]) -> Tuple[List[str], np.ndarray, np.ndarray, List[np.ndarray]]
```

### Variable System
**Key methods:**
```python
def create_axis_variables(var_list: List[Dict]) -> List['AxisVariable']
def evaluate_at_stations_cached(axis_variables: List['AxisVariable'], stations: List[float]) -> List[Dict[str, float]]
def evaluate(self, station: float) -> float
```

## Data Flow Patterns

1. **JSON → Objects**: `SPOT_Filters` → `SpotLoader` → `models/base.py` functions
2. **Objects → Geometry**: `configure()` → `build()` → `compute_embedded_points()`  
3. **Geometry → Visualization**: Geometry results → `Plotter` methods → Plotly traces
4. **Validation**: Scattered `validate()` calls and `safe_eval_scalar()` for expressions

---
**Note**: This call graph is automatically validated by `scripts/check_docs_sync.py`. If function signatures change, the validation will catch drift and fail builds.