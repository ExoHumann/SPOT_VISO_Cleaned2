# Smoke Tests

This directory contains minimal smoke tests that assert the most important invariants for refactoring safety.

## Tests

### test_axis_frame.py
- Builds a tiny synthetic axis (3-point straight line)
- Tests `frame_at_stations()` and `frame_at_station()` functions
- Verifies both "pt" (parallel transport) and "symmetric" modes
- Asserts:
  - Unit-norm tangents, normals, and binormals
  - Orthogonality of frame vectors (T·N=0, T·B=0, N·B=0)
  - No NaN values in output
  - Correct array shapes

### test_embed_basic.py  
- Creates a tiny rectangular section (500mm x 300mm)
- Tests embedding at 3 stations on a curved axis
- Compares default vs symmetric embedding modes
- Verifies:
  - Stations are processed monotonically
  - All arrays are finite (no NaN/inf)
  - Embedded points vary appropriately across stations
  - Different modes produce appropriately different results
  - Edge cases (single station) work correctly

## Running Tests

Use the CLI command:
```bash
python spotviso.py test -m smoke
```

Or run individual tests:
```bash
python tests/smoke/test_axis_frame.py
python tests/smoke/test_embed_basic.py
```

## Properties

- **Fast**: Tests run in ~0.2 seconds total
- **Independent**: Tests use synthetic data, no dependencies on GIT/* files
- **Minimal**: Focus on core invariants needed for safe refactoring
- **Robust**: Test both normal cases and edge cases