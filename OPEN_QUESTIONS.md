# OPEN_QUESTIONS.md

## Critical Design Decisions

### 1. Unit System Standardization
**Question**: Should the system standardize on millimeters or meters throughout, and how should unit conversion be handled?

**Context**: Currently mixed usage - `stations_m` parameters expect meters while `Axis` stores internally in millimeters. JSON data often in meters but converted inconsistently.

**Options**:
- **A) Pure millimeter**: All internal storage and APIs use mm, explicit conversion at boundaries
- **B) Pure meter**: Convert everything to meters, aligning with engineering conventions  
- **C) Typed units**: Use libraries like `pint` for unit-aware calculations
- **D) Dual API**: Provide both `_mm` and `_m` versions of functions

**Impact**:
- **A**: Requires extensive refactoring but eliminates conversion errors  
- **B**: Simpler for users but risks precision loss in fine details
- **C**: Type safety but adds dependency and complexity
- **D**: Maintains compatibility but increases API surface area

**Dependencies**: Affects all coordinate calculations, file I/O, and user APIs

### 2. Expression Evaluation Security
**Question**: How should parametric expressions in JSON be evaluated safely while maintaining performance?

**Context**: Current implementation uses `eval()` with restricted builtins for cross-section parametric expressions. Potential security risk with untrusted JSON.

**Options**:  
- **A) Continue eval() with sandboxing**: Enhance current approach with more restrictions
- **B) AST-based parsing**: Use `ast.parse()` with whitelisted operations only
- **C) Domain-specific language**: Create mini-language for geometric expressions
- **D) Pre-compiled expressions**: Compile all expressions at load time, cache results

**Impact**:
- **A**: Minimal changes but security risk remains
- **B**: Safer but potentially slower, limits expression complexity  
- **C**: Maximum safety and clarity but requires language design
- **D**: Best performance but breaks dynamic expression capability

**Dependencies**: Core to cross-section evaluation pipeline, affects all parametric geometry

### 3. Coordinate Frame Convention  
**Question**: What should be the canonical coordinate system and handedness throughout the system?

**Context**: Current code shows mixed usage - section coordinates as (Y,Z) vs (X,Z), plan rotations affecting XY vs YZ planes.

**Options**:
- **A) Civil engineering standard**: X=longitudinal, Y=lateral, Z=vertical (right-handed)
- **B) Graphics standard**: X=right, Y=up, Z=forward (right-handed)  
- **C) Current mixed approach**: Different conventions per module
- **D) Configurable frames**: Allow user to specify coordinate conventions

**Impact**:
- **A**: Aligns with domain but requires coordinate mapping documentation
- **B**: Easier for visualization libraries but may confuse engineers
- **C**: Status quo but source of ongoing confusion
- **D**: Flexible but increases complexity and testing burden

**Dependencies**: Affects axis definitions, section embeddings, visualization output

### 4. Geometry Computation Caching Strategy
**Question**: How should expensive geometry computations be cached for performance?

**Context**: Current system has minimal caching (only AxisVariable evaluation). Geometry recomputation is expensive for interactive use.

**Options**:
- **A) Object-level caching**: Cache results at DeckObject/PierObject level
- **B) Function-level caching**: LRU cache on individual computation functions  
- **C) Context-aware caching**: Cache based on object+context+parameters hash
- **D) Persistent caching**: Save computed geometry to disk between runs

**Impact**:
- **A**: Simple but may cache too coarsely, invalidation complexity
- **B**: Fine-grained but complex cache key management
- **C**: Most accurate but complex dependency tracking  
- **D**: Fastest but storage management and invalidation challenges

**Dependencies**: Performance characteristics, memory usage, API design for cache invalidation

### 5. Error Handling and Validation Strategy
**Question**: How should the system handle invalid geometry, malformed data, and computation failures?

**Context**: Current implementation has minimal validation. Silent failures can produce invalid 3D geometry.

**Options**:
- **A) Fail-fast validation**: Validate all inputs immediately, raise exceptions on errors
- **B) Graceful degradation**: Continue with warnings, provide fallback geometry
- **C) Configurable tolerance**: Allow users to set validation strictness levels
- **D) Validation layers**: Different validation at data loading vs computation vs rendering

**Impact**:
- **A**: Safest but may break existing workflows with slightly invalid data
- **B**: Better user experience but risk of silent corruption
- **C**: Flexible but adds configuration complexity 
- **D**: Most comprehensive but complex to implement and test

**Dependencies**: User experience, data quality requirements, debugging capabilities

### 6. Multi-threading and Parallelization  
**Question**: Should geometry computation be parallelized, and if so, at what granularity?

**Context**: Current implementation is single-threaded. Large models with many objects could benefit from parallel processing.

**Options**:
- **A) Object-level parallelization**: Compute multiple objects concurrently
- **B) Station-level parallelization**: Parallel processing within objects across stations
- **C) Vectorized operations only**: Rely on numpy/scipy vectorization, no explicit threading
- **D) Configurable parallelization**: Allow users to choose parallelization strategy

**Impact**:
- **A**: Simple implementation, good for multi-object scenes
- **B**: Better for single large objects, more complex implementation
- **C**: Simplest, relies on library optimization
- **D**: Maximum flexibility but increases complexity

**Dependencies**: Performance requirements, thread safety, dependency on numpy threading

### 7. Data Loading and Branch Management
**Question**: How should multiple data branches (GIT/MAIN, GIT/RCZ_new1, etc.) be handled in the API?

**Context**: Current implementation can load different branches but lacks systematic branch management or merging capabilities.

**Options**:
- **A) Single branch per context**: Load one branch at a time, simple API
- **B) Multi-branch contexts**: Allow loading multiple branches, resolve conflicts
- **C) Branch inheritance**: Define branch hierarchies with override capabilities  
- **D) Dynamic branch switching**: Change branches without full reload

**Impact**:
- **A**: Simplest but limits analysis of cross-branch differences  
- **B**: Flexible but complex conflict resolution needed
- **C): Most powerful but requires branch relationship modeling
- **D**: Best user experience but complex state management

**Dependencies**: Data organization, API complexity, performance for large datasets

### 8. Visualization Pipeline Coupling
**Question**: How tightly should visualization be coupled to geometry computation?

**Context**: Current implementation mixes computation and visualization concerns. Visualization requirements sometimes drive geometry computation decisions.

**Options**:
- **A) Strict separation**: Geometry computation produces standard format, visualization layer consumes
- **B) Computation hints**: Allow visualization requirements to influence computation parameters  
- **C) Integrated pipeline**: Compute geometry optimized for specific visualization backends
- **D) Pluggable backends**: Support multiple visualization outputs from same computation

**Impact**:
- **A**: Clean architecture but may miss optimization opportunities
- **B**: Practical compromise but increases coupling
- **C**: Fastest for specific use cases but reduces flexibility  
- **D**: Most flexible but complex interface design

**Dependencies**: Performance optimization opportunities, support for different output formats

### 9. Configuration and Environment Management
**Question**: What configuration format and hierarchy should be used for different deployment scenarios?

**Context**: Current system has configuration spread across multiple files with some hard-coded values.

**Options**:
- **A) Single TOML file**: All configuration in one file with sections
- **B) Hierarchical configs**: System/user/project level configuration files
- **C) Environment-first**: Environment variables with file fallbacks
- **D) Runtime configuration**: Programmatic configuration with sensible defaults

**Impact**:
- **A**: Simple but may not scale to complex deployment scenarios
- **B**: Flexible but complex precedence rules
- **C**: Good for containerized deployments but harder for desktop use
- **D**: Most flexible but requires good default discovery

**Dependencies**: Deployment patterns, user experience, DevOps requirements

### 10. API Versioning and Backward Compatibility  
**Question**: How should API evolution be managed while maintaining compatibility with existing workflows?

**Context**: Current codebase lacks formal API versioning. Changes could break existing scripts and workflows.

**Options**:
- **A) Semantic versioning with deprecation warnings**: Standard approach with migration periods
- **B) Parallel API versions**: Maintain multiple API versions simultaneously
- **C) Feature flags**: Enable/disable new behavior with configuration flags
- **D) Automatic migration**: Detect old usage patterns and adapt automatically

**Impact**:
- **A**: Standard practice but requires discipline in change management  
- **B**: Maximum compatibility but increases maintenance burden
- **C**: Flexible but can lead to complex configuration matrices
- **D**: Best user experience but complex detection and migration logic

**Dependencies**: User adoption patterns, maintenance resources, API stability requirements

### 11. Performance vs. Precision Trade-offs
**Question**: Where should the system allow users to trade precision for performance in geometry computation?

**Context**: High-precision geometry computation can be slow. Different use cases may have different precision requirements.

**Options**:
- **A) Single precision level**: Choose good defaults, no user control
- **B) Global precision setting**: System-wide precision configuration
- **C) Per-operation precision**: Allow precision control at function level
- **D) Adaptive precision**: Automatically adjust precision based on geometry complexity

**Impact**:
- **A**: Simplest but may not satisfy all use cases
- **B**: Good balance of simplicity and control
- **C**: Maximum control but complex API surface
- **D**: Intelligent but complex heuristics required

**Dependencies**: Performance requirements, numerical accuracy needs, API complexity

### 12. Extension and Plugin Architecture
**Question**: Should the system support plugins or extensions for custom object types and computations?

**Context**: Current system has fixed object types (Deck, Pier, Foundation). Users may need custom structural components.

**Options**:
- **A) Fixed architecture**: Support only built-in object types
- **B) Inheritance-based extension**: Allow subclassing existing objects
- **C) Plugin system**: Formal plugin architecture with registration  
- **D) Script integration**: Support external computation scripts

**Impact**:
- **A**: Simplest but limits extensibility
- **B**: Natural Python approach but requires understanding internals
- **C**: Most flexible but adds architectural complexity
- **D**: Very flexible but integration and validation challenges

**Dependencies**: User extensibility needs, API stability, testing complexity