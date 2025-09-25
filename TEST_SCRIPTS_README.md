### `demo_flexibility.py`
Comprehensive demonstration of the modular architecture's flexibility with FLIP90.
```bash
python demo_flexibility.py
```
Generates 3 different visualizations (all with FLIP90 rotation):
- MAIN data only + FLIP90
- RCZ_new1 data only + FLIP90
- Mixed MAIN/RCZ_new1 sources + FLIP90

### `run_deck.py`
Tests DeckObject with different data sources.
```bash
# Test with MAIN data
python run_deck.py --out deck_main.html

# Test with RCZ_new1 data (different axis and cross sections)
python run_deck.py --axis GIT/RCZ_new1/_Axis_JSON.json --cross GIT/RCZ_new1/_CrossSection_JSON.json --obj GIT/RCZ_new1/_DeckObject_JSON.json --main GIT/RCZ_new1/_MainStation_JSON.json --out deck_rcz_new1.html

# Test with FLIP90 rotation
python run_deck.py --flip90 --out deck_flip90.html
```

### `run_pier.py`
Tests PierObject with different data sources.
```bash
# Test with MAIN data
python run_pier.py --out pier_main.html

# Test with RCZ_new1 data
python run_pier.py --axis GIT/RCZ_new1/_Axis_JSON.json --cross GIT/RCZ_new1/_CrossSection_JSON.json --obj GIT/RCZ_new1/_PierObject_JSON.json --main GIT/RCZ_new1/_MainStation_JSON.json --out pier_rcz_new1.html

# Test with FLIP90 rotation
python run_pier.py --flip90 --out pier_flip90.html
```

### `run_foundation.py`
Tests FoundationObject with different data sources.
```bash
# Test with MAIN data
python run_foundation.py --out foundation_main.html

# Note: RCZ_new1 foundation data may need cross section configuration

# Test with FLIP90 rotation
python run_foundation.py --flip90 --out foundation_flip90.html
```

## Demonstrated Flexibility

The scripts show how the modular architecture allows:

1. **Different Axes**: Objects can use different axis definitions
   - MAIN: Axis with 35 stations (-100mm to 201722mm)
   - RCZ_new1: Axis with 4 stations (0mm to 300000mm)

2. **Different Cross Sections**: Objects can use different cross section sets
   - MAIN: Various NCS values (111, 501, 4000, etc.)
   - RCZ_new1: Additional NCS values (121, 3001, etc.)

3. **Component Mixing**: Objects can be configured with components from different data sources

## Key Features

- **Modular Configuration**: Each object type has its own `configure()` method
- **Component Selection**: Objects choose appropriate components from available options
- **Data Source Flexibility**: Can load axes, cross sections, and main stations from different JSON files
- **Type-Specific Logic**: Each object type implements its own configuration logic
- **Original Functionality Preserved**: FLIP90 and other transformations still work

## Usage Examples

```bash
# Mix MAIN axis with RCZ_new1 cross sections
python run_deck.py --axis GIT/MAIN/_Axis_JSON.json --cross GIT/RCZ_new1/_CrossSection_JSON.json --out mixed_sources.html

# Use specific section files
python run_deck.py --section MASTER_SECTION/Kulosaari_MasterCrossSection.json --out custom_section.html
```

This demonstrates how the refactored architecture enables flexible component composition for different structural analysis scenarios.