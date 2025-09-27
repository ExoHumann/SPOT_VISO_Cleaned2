# SPOT VISO Scripts

This directory contains automation scripts for the SPOT VISO project.

## check_docs_sync.py

Automates detection of documentation drift when public function signatures change.

### Usage

```bash
# Check if all documented signatures match code
python scripts/check_docs_sync.py

# Verbose output with details
python scripts/check_docs_sync.py --verbose

# Check specific project root
python scripts/check_docs_sync.py --project-root /path/to/project
```

### What it checks

The script scans:
- `ARCH_MAP.md` - Public API function signatures
- `CALL_GRAPH.md` - Function call relationships and signatures

It verifies that all documented function signatures match the actual signatures in the source code files under `models/`.

### Exit codes

- `0` - All signatures match ✅
- `1` - Signature mismatches found ❌
- `2` - Script error (file not found, parse error, etc.) ⚠️

### Integration

The script is integrated into:
- **Pre-commit hooks** - `.pre-commit-config.yaml` 
- **CI/CD pipeline** - `.github/workflows/validate-agent-pr.yml`

This ensures that any changes to public function signatures are immediately detected and require documentation updates.

### Supported file patterns

- **Function definitions**: `def function_name(args) -> return_type:`
- **Code blocks**: Python code blocks in markdown files
- **Module sections**: `### models/filename.py` headers

### Examples

When a function signature changes, you'll see output like:

```
DOCS DRIFT DETECTED!
==================================================
In ARCH_MAP.md:
MISMATCH: models/axis.py:get_position_at_station
  Doc:  def get_position_at_station(self, station: float) -> Tuple[float, float]  
  Code: def get_position_at_station(self, station: float) -> Tuple[float, float, float] (line 150)

Please update the documentation files to match current function signatures.
```

This indicates that the documentation shows a 2-tuple return type but the actual code returns a 3-tuple.