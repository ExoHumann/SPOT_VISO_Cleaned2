# SPOT VISO CLI Commands

This document describes the unified CLI interface for SPOT VISO operations.

## Installation

Ensure you have the required dependencies:

```bash
pip install numpy plotly jinja2 scipy matplotlib
```

## Commands Overview

SPOT VISO provides four main CLI commands:

- `spotviso check` - Validate data integrity and geometry
- `spotviso test -m smoke` - Quick validation suite (<60s)
- `spotviso test-all` - Full test suite
- `spotviso viz --case <id>` - Visualization by case identifier

## Command Details

### spotviso check

Validates data integrity and basic geometry of SPOT VISO datasets.

```bash
# Check default data path (GIT/MAIN)
spotviso check

# Check specific data path
spotviso check --data-path GIT/RCZ_new1
```

**What it checks:**
- ✅ Required JSON files exist (`_Axis_JSON.json`, `_CrossSection_JSON.json`, `_MainStation_JSON.json`)
- ✅ Optional object files exist (`_DeckObject_JSON.json`, `_PierObject_JSON.json`, `_FoundationObject_JSON.json`) 
- ✅ JSON files are valid and contain expected data structures
- ✅ MASTER_SECTION directory exists with section definitions
- ⚠️ Reports warnings for missing optional components

**Exit codes:**
- `0` - All checks passed (with or without warnings)
- `1` - Critical errors found

### spotviso test -m smoke

Quick validation suite designed to complete in under 60 seconds.

```bash
# Run smoke tests on default data
spotviso test -m smoke

# Run smoke tests on specific data
spotviso test -m smoke --data-path GIT/RCZ_new1
```

**What it tests:**
1. **Data Validation** - Runs `spotviso check` first
2. **Import Tests** - Verifies core modules can be imported
3. **Data Loading** - Tests SpotLoader initialization
4. **Performance** - Reports completion time (warns if >60s)

**Exit codes:**
- `0` - All smoke tests passed
- `1` - One or more tests failed

### spotviso test-all

Comprehensive test suite including smoke tests plus advanced functionality.

```bash
# Run full test suite
spotviso test-all

# Run full test suite on specific data
spotviso test-all --data-path GIT/MAIN
```

**What it tests:**
- ✅ All smoke tests
- ✅ Multiple data branches validation
- ✅ Advanced functionality checks
- 📊 Reports on available data branches

**Exit codes:**
- `0` - All tests passed
- `1` - One or more tests failed

### spotviso viz --case <id>

Generate visualizations by case identifier.

```bash
# Generate demo visualization
spotviso viz --case demo

# Generate demo with custom output
spotviso viz --case demo --output my_viz.html

# Use specific data path
spotviso viz --case demo --data-path GIT/RCZ_new1
```

**Available cases:**
- `demo` - Basic demonstration visualization

**Output:**
- HTML file with interactive visualization
- Default output: `spotviso_viz_<case>.html`

**Exit codes:**
- `0` - Visualization generated successfully
- `1` - Visualization failed

## Usage Examples

### Basic Workflow

```bash
# 1. Check data integrity
spotviso check

# 2. Run quick validation
spotviso test -m smoke

# 3. Generate demo visualization
spotviso viz --case demo
```

### Working with Different Data Branches

```bash
# Check different data branches
spotviso check --data-path GIT/MAIN
spotviso check --data-path GIT/RCZ_new1
spotviso check --data-path GIT/RCZ_new

# Run tests on specific branch
spotviso test -m smoke --data-path GIT/RCZ_new1

# Generate visualization from specific branch
spotviso viz --case demo --data-path GIT/RCZ_new1
```

### Integration with Existing Tools

The new CLI commands complement existing SPOT VISO tools:

```bash
# Use CLI for validation, then existing tools for detailed work
spotviso check
python run_linear.py --obj-type DeckObject --out deck.html

# Quick validation before running existing scripts
spotviso test -m smoke && python run_deck.py --out deck_main.html
```

## Global Installation (Optional)

To make `spotviso` available globally:

```bash
# Add bin directory to PATH
export PATH="$PWD/bin:$PATH"

# Now you can run from anywhere
spotviso check
```

Or create a symbolic link:

```bash
ln -s "$PWD/bin/spotviso" /usr/local/bin/spotviso
```

## Error Handling

All commands provide clear error messages and appropriate exit codes:

- **Exit Code 0**: Success
- **Exit Code 1**: Error or failure
- **Warnings**: Reported but don't cause failure (exit code 0)

## Integration with CI/CD

The CLI commands are designed for automation:

```yaml
# Example GitHub Actions step
- name: Validate SPOT VISO Data
  run: |
    spotviso check
    spotviso test -m smoke

# Exit code handling in bash
if ! spotviso check; then
  echo "Data validation failed"
  exit 1
fi
```

## Development and Testing

For developers working on SPOT VISO:

```bash
# Run all validation before making changes
spotviso check && spotviso test-all

# Quick validation during development
spotviso test -m smoke

# Test CLI functionality
python test_cli.py
```