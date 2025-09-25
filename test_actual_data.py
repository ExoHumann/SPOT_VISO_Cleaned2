#!/usr/bin/env python3
"""
Test actual cross section data to see coordinate interpretation.
"""

import numpy as np
import json
from models.cross_section import CrossSection

def test_actual_cross_section():
    print("=== Testing Actual Cross Section Data ===\n")

    # Load the deck cross section
    with open('MASTER_SECTION/MASTER_DeckCrBm-1Gird-Slab.json', 'r') as f:
        section_data = json.load(f)

    cs = CrossSection(name="DeckTest", data=section_data)

    # Get the variables used
    from models.cross_section import _collect_points, _used_vars
    points = _collect_points(section_data)
    used_vars = _used_vars(points)
    print(f"Variables used in cross section: {sorted(used_vars)}")

    # Create some reasonable default values for testing
    # Based on the variable names, these seem to be geometric parameters
    defaults = {}

    # Basic geometry variables
    defaults['HORZ_OO'] = 0.0      # Origin horizontal
    defaults['VERT_OO'] = 0.0      # Origin vertical
    defaults['WIDH_1000'] = 1000.0 # Width
    defaults['SLOP_1000'] = 0.0    # Slope

    # Add some basic values for the numbered variables
    for i in range(1000, 1030):
        defaults[f'HORZ_{i}'] = 100.0 + (i-1000)*10  # Increasing horizontal
        defaults[f'VERT_{i}'] = 50.0                  # Constant vertical
        defaults[f'SLOP_{i}'] = 0.0                  # No slope

    for i in range(1050, 1080):
        defaults[f'HORZ_{i}'] = 100.0 + (i-1050)*10
        defaults[f'VERT_{i}'] = 50.0
        defaults[f'SLOP_{i}'] = 0.0

    for i in range(2000, 2020):
        defaults[f'HORZ_{i}'] = 100.0 + (i-2000)*10
        defaults[f'VERT_{i}'] = 50.0
        defaults[f'SLOP_{i}'] = 0.0

    for i in range(2050, 2070):
        defaults[f'HORZ_{i}'] = 100.0 + (i-2050)*10
        defaults[f'VERT_{i}'] = 50.0
        defaults[f'SLOP_{i}'] = 0.0

    # Special points
    defaults['HORZ_K000'] = 500.0
    defaults['VERT_K000'] = 100.0
    defaults['HORZ_K050'] = 500.0
    defaults['VERT_K050'] = 100.0
    defaults['HORZ_G010'] = 0.0
    defaults['VERT_G010'] = 0.0

    for i in range(2010, 2020, 2):
        defaults[f'HORZ_{i}'] = 200.0
        defaults[f'VERT_{i}'] = 150.0
        defaults[f'SLOP_{i}'] = 0.1

    # Create variable arrays (single station)
    var_arrays = {name: np.array([value]) for name, value in defaults.items()}

    # Evaluate cross section
    ids, X_mm, Y_mm, loops = cs.evaluate(var_arrays, negate_x=True)

    print(f"\nCross section evaluation:")
    print(f"  Number of points: {len(ids)}")
    print(f"  X_mm range: {np.nanmin(X_mm):.1f} to {np.nanmax(X_mm):.1f} mm")
    print(f"  Y_mm range: {np.nanmin(Y_mm):.1f} to {np.nanmax(Y_mm):.1f} mm")

    # Show some sample points
    print(f"\nSample evaluated points:")
    valid_indices = []
    for i in range(len(ids)):
        if not (np.isnan(X_mm[0,i]) or np.isnan(Y_mm[0,i])):
            valid_indices.append(i)
            if len(valid_indices) <= 10:  # Show first 10 valid points
                print(f"  {ids[i]}: X={X_mm[0,i]:.1f} mm, Y={Y_mm[0,i]:.1f} mm")

    print(f"\nTotal valid points: {len(valid_indices)}")

    # Analyze coordinate patterns
    if len(valid_indices) > 0:
        X_vals = X_mm[0, valid_indices]
        Y_vals = Y_mm[0, valid_indices]

        print(f"\nCoordinate analysis:")
        print(f"  X (Coord[0], negated): min={np.min(X_vals):.1f}, max={np.max(X_vals):.1f}, range={np.max(X_vals)-np.min(X_vals):.1f}")
        print(f"  Y (Coord[1]): min={np.min(Y_vals):.1f}, max={np.max(Y_vals):.1f}, range={np.max(Y_vals)-np.min(Y_vals):.1f}")

        # Check if this looks like a deck cross section
        x_range = np.max(X_vals) - np.min(X_vals)
        y_range = np.max(Y_vals) - np.min(Y_vals)

        print(f"\nInterpretation:")
        if x_range > y_range:
            print(f"  X varies more than Y → X is transverse ({x_range:.1f}mm), Y is height ({y_range:.1f}mm)")
        else:
            print(f"  Y varies more than X → Y is transverse ({y_range:.1f}mm), X is height ({x_range:.1f}mm)")

        # Check some specific points from the JSON
        print(f"\nSpecific point analysis:")
        for pid in ['RA', 'OO', 'HW0', '1000', '1050']:
            if pid in ids:
                idx = ids.index(pid)
                if not (np.isnan(X_mm[0,idx]) or np.isnan(Y_mm[0,idx])):
                    print(f"  {pid}: X={X_mm[0,idx]:.1f}, Y={Y_mm[0,idx]:.1f}")

if __name__ == "__main__":
    test_actual_cross_section()