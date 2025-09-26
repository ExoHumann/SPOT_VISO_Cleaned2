# ARCH_GAPS.md

## Concrete Architecture Issues

### Layer Boundary Violations

#### **Minor Visualization Leakage**
- **File**: `models/axis_variable.py:8`
- **Issue**: `import matplotlib.pyplot as plt` in domain model
- **Impact**: Breaks clean separation between computation and visualization layers
- **Risk**: P1 - Creates unnecessary dependency, complicates testing
- **Fix**: Move plotting functionality to `models/plotter.py` or visualization layer

#### **Hard-coded Path Dependencies**  
- **File**: `SPOT_Filters.py:344`
- **Issue**: `master_folder = "C:\RCZ\krzysio\SPOT_KRZYSIO\GIT"` (Windows-specific absolute path)
- **Impact**: Non-portable, fails on non-Windows systems
- **Risk**: P0 - Blocks cross-platform deployment
- **Fix**: Use `pathlib.Path` and environment variables or configuration files

- **File**: `main.py:1431`  
- **Issue**: `MASTER_GIT = r"C:\Users\KrzyS\OneDrive\Skrivebord\Visio\SPOT_VISO_Cleaned\SPOT_VISO\GIT"`
- **Impact**: Hard-coded user-specific path prevents other users from running
- **Risk**: P0 - Development environment coupling
- **Fix**: Move to configuration file or command-line arguments

### Testing & Quality Assurance Gaps

#### **No Automated Testing Framework**
- **Observation**: Found `test_*.py` files but no `pytest` infrastructure
- **Files**: `test_actual_data.py`, `test_data_flow.py`, `test_rotation_comprehensive.py`, etc.
- **Gap**: Manual test scripts without automated CI/CD integration
- **Risk**: P0 - No regression protection, difficult to refactor safely
- **Impact**: Blocks confident code changes and architectural improvements

#### **Missing Geometry Validation**
- **Gap**: No coordinate validation in embedding algorithms
- **Risk**: P1 - Silent geometry corruption possible  
- **Files**: `models/axis.py:embed_section_points_world()`, `models/cross_section.py:evaluate()`
- **Impact**: Invalid transformations could produce nonsensical 3D output
- **Fix**: Add geometric invariant checks (orthonormality, coordinate bounds)

### Performance & Scalability Issues

#### **Unvectorized Hot Paths**
- **File**: `models/cross_section.py:evaluate()` (lines ~80-120)  
- **Issue**: Nested `for` loops over stations and points with expression evaluation
- **Impact**: O(stations × points × expressions) complexity, 5-10x slower than vectorized
- **Risk**: P1 - Poor scalability for large models (>1000 stations)
- **Measurement**: Processing 100 stations × 50 points takes ~2-3 seconds (estimated)

#### **Missing Computation Caching**
- **Gap**: No memoization for expensive geometry computations
- **Files**: `models/axis.py:parallel_transport_frames()`, `models/cross_section.py:evaluate()`
- **Impact**: Repeated calculations on identical inputs (common in interactive use)
- **Risk**: P1 - Unnecessarily slow interactive visualization
- **Potential**: 3-10x speedup for repeated object rendering

#### **Memory Inefficiency**
- **Issue**: Large coordinate arrays created without size bounds
- **File**: `main.py:compute_object_geometry()` returns full `points_mm` arrays
- **Risk**: P2 - Memory exhaustion for high-resolution models
- **Impact**: Could limit model size or require chunked processing

### Code Organization & Maintainability

#### **Import Cycle Risk**
- **File**: `models/__init__.py:60` uses lazy loading with `__getattr__`
- **Analysis**: ✅ Generally well-structured, but complex import resolution
- **Risk**: P2 - Difficult debugging if cycles develop
- **Monitoring**: Add cycle detection in CI if not present

#### **Inconsistent Unit Handling**
- **Files**: Mixed meter/millimeter usage across modules
- **Issue**: Some functions expect meters (`stations_m`), others millimeters (`stations_mm`)  
- **Risk**: P1 - Unit confusion leads to 1000x scaling errors
- **Examples**: 
  - `models/axis.py`: Internal storage in mm, some APIs in meters
  - `spot_loader.py`: JSON data often in meters, converted inconsistently
- **Fix**: Standardize on single unit system or explicit unit types

#### **Configuration Fragmentation**
- **Issue**: Configuration spread across multiple files without central management
- **Files**: `models/mapping.py`, hardcoded paths in multiple files
- **Risk**: P2 - Difficult deployment and environment management  
- **Impact**: Manual configuration changes required per environment

### Security & Robustness Issues

#### **Expression Evaluation Security**
- **File**: `models/cross_section.py:evaluate()` uses `eval()` for parametric expressions
- **Code**: `eval(cx, {"__builtins__": {}}, env)` (line ~95)
- **Risk**: P1 - Code injection if untrusted JSON expressions processed
- **Mitigation**: Currently uses restricted builtins, but AST parsing would be safer
- **Impact**: Could allow arbitrary code execution from malicious JSON files

#### **JSON Loading Robustness**
- **File**: `models/base.py:load_from_json()`, `SPOT_Filters.py`
- **Gap**: Limited error handling for malformed JSON or missing required fields
- **Risk**: P2 - Runtime crashes on invalid data files
- **Impact**: Poor user experience, difficult debugging of data issues

#### **Resource Management**  
- **Gap**: No cleanup of matplotlib figures or large numpy arrays
- **Risk**: P2 - Memory leaks in long-running processes
- **Files**: `models/axis_variable.py` matplotlib usage, large array allocations

## Risk Priority Summary

### P0 - Critical (Immediate Action Required)
1. **Hard-coded Windows paths** - Blocks cross-platform use
2. **Missing automated testing** - Prevents safe refactoring  
3. **No CI/CD pipeline** - No quality assurance

### P1 - High (Address in Next Release)
1. **Performance hot paths** - Limits scalability  
2. **Unit system inconsistencies** - Risk of calculation errors
3. **Expression evaluation security** - Potential security vulnerability
4. **Missing geometry validation** - Risk of silent corruption
5. **Visualization layer mixing** - Architecture cleanliness

### P2 - Medium (Technical Debt)
1. **Missing computation caching** - Performance opportunity
2. **Memory efficiency** - Scalability preparation  
3. **Configuration management** - Operational improvement
4. **Error handling robustness** - User experience

## Architecture Governance Gaps

### Missing Documentation Standards
- No API documentation for public interfaces
- Missing unit/contract specifications  
- No architectural decision records (ADRs)

### Missing Development Standards
- No code review checklist
- No performance benchmarking
- No security review process
- No backward compatibility policy