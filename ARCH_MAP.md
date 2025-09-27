# SPOT VISO - Architecture Map & Public API

This document lists the public function signatures for the core SPOT VISO modules. These signatures are automatically checked for drift using `scripts/check_docs_sync.py`.

## Core Domain Models

### models/axis.py
```python
def validate(self) -> None
def clamp_station_m(self, s_m: float) -> float
def point_at_m(self, s_m: float) -> np.ndarray
def get_segment_for_station(self, station: float) -> Tuple[Optional[int], Optional[float]]
def get_position_at_station(self, station: float) -> Tuple[float, float, float]
def parallel_transport_frames(self, stations_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]
def embed_section_points_world(self, stations_mm: np.ndarray, yz_points_mm: np.ndarray, x_offsets_mm: np.ndarray | None = None, rotation_deg: float | np.ndarray = 0.0, plan_rotation_deg: float | np.ndarray = 0.0) -> np.ndarray
def embed_section_points_world_symmetric(self, stations_mm: np.ndarray, yz_points_mm: np.ndarray, x_offsets_mm: np.ndarray | None = None, rotation_deg: float | np.ndarray = 0.0, plan_rotation_deg: float | np.ndarray = 0.0) -> np.ndarray
def frame_at_stations(self, stations_mm: np.ndarray) -> dict
def frame_at_station(self, station_mm: float) -> dict
```

### models/cross_section.py
```python
def from_file(cls, path: str, name: Optional[str] = None) -> 'CrossSection'
def defaults_mm(self) -> Dict[str, float]
def evaluate(self, var_arrays_all: Dict[str, np.ndarray]) -> Tuple[List[str], np.ndarray, np.ndarray, List[np.ndarray]]
def compute_local_points(self) -> Tuple[List[str], np.ndarray, np.ndarray, List[np.ndarray]]
def compute_embedded_points(self) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]
```

### models/deck_object.py
```python
def get_object_metadata(self) -> Dict
def configure(self, available_axes: Dict[str, Axis], available_cross_sections: Dict[int, CrossSection], available_mainstations: Dict[str, List[MainStationRef]], axis_name: Optional[str] = None, cross_section_ncs: Optional[List[int]] = None, mainstation_name: Optional[str] = None) -> None
```

### models/pier_object.py
```python
def configure(self, available_axes: Dict[str, Axis], available_cross_sections: Dict[int, CrossSection], available_mainstations: Dict[str, List], axis_name: Optional[str] = None, cross_section_ncs: Optional[List[int]] = None, mainstation_name: Optional[str] = None) -> None
def build(self)
def set_world_anchor(self, world_point_mm: tuple | List[float] | np.ndarray)
def set_anchor_from_deck(self, deck_result: Dict, point_name: str)
```

### models/foundation_object.py
```python
def compute_geometry(self) -> Dict[str, object]
def configure(self, available_axes: Dict[str, Axis], available_cross_sections: Dict[int, CrossSection], available_mainstations: Dict[str, List[MainStationRef]], axis_name: Optional[str] = None, cross_section_ncs: Optional[List[int]] = None, mainstation_name: Optional[str] = None) -> None
```

## Utility Functions

### models/utils.py
```python
def safe_eval_scalar(expr: str, vars_: Dict[str, float]) -> float
```

### models/axis_variable.py
```python
def validate(self)
def evaluate(self, station: float) -> float
def plot(self, start: float = 0, end: float = 100, step: float = 1)
def create_axis_variables(var_list: List[Dict]) -> List['AxisVariable']
def evaluate_at_stations_cached(axis_variables: List['AxisVariable'], stations: List[float]) -> List[Dict[str, float]]
```

## Data Loading & Configuration

### models/base.py
```python
def load_axis_from_rows(axis_rows: List[Dict], axis_name: str) -> Axis
def index_cross_sections_by_ncs(cross_rows: List[Dict]) -> Dict[int, Dict]
def choose_section_path_for_ncs(cs_row: Dict, fallback_path: str) -> str
def load_section_for_ncs(ncs: int, by_ncs: Dict[int, Dict], fallback_path: str) -> CrossSection
def cs_steps_from_deck_row(deck_row: Dict) -> List[Tuple[float, int]]
```

### models/main_station.py
```python
def load_mainstations_from_rows(rows: List[dict]) -> List[MainStationRef]
```

## Visualization

### models/plotter.py
```python
def build_axis_trace(self, traces: List)
def build_overlay_traces(self, traces: List, overlays: Optional[List[Dict]])
def build_first_station_points(self, traces: List)
def build_loop_traces(self, traces: List)
def build_longitudinal_traces(self, traces: List)
```

---
**Note**: This document is automatically validated by `scripts/check_docs_sync.py`. If a function signature changes in the code, the script will detect the drift and fail CI/pre-commit checks.