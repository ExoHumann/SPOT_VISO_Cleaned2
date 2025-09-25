#!/usr/bin/env python3
"""
Advanced rotation analysis showing what LOCAL vs GLOBAL rotation should look like.
This demonstrates the difference between rotating in the cross section's local plane
vs rotating the entire structure globally.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

def analyze_rotation_issue():
    """Analyze and explain the rotation behavior."""

    print("=== FLIP90 Rotation Analysis ===\n")

    print("ISSUE IDENTIFIED:")
    print("The current FLIP90 implementation applies rotation_override_deg=90.0,")
    print("which rotates ALL cross sections by the same +90° angle in their LOCAL coordinate system.")
    print("This means every cross section gets rotated by +90° relative to its own orientation.")
    print("")
    print("If the cross sections are already oriented consistently along the axis,")
    print("this uniform +90° rotation will look like a GLOBAL rotation of the entire structure.")
    print("")

    print("WHAT TO LOOK FOR:")
    print("1. deck_normal.html - Baseline orientation")
    print("2. deck_flip90_current.html - All cross sections rotated +90° locally")
    print("3. deck_twist_90.html - Gradual twist rotation along the axis")
    print("")

    print("EXPECTED BEHAVIOR for LOCAL rotation:")
    print("- Each cross section should rotate in its own local YZ plane")
    print("- The rotation should be relative to the cross section's current orientation")
    print("- If all cross sections start with the same orientation, +90° local rotation")
    print("  will make the entire structure appear rotated globally")
    print("")

    print("If FLIP90 appears to rotate the GLOBAL structure rather than LOCAL cross sections,")
    print("then the current implementation may actually be working correctly, but the")
    print("visual result looks 'global' because all cross sections have consistent orientation.")
    print("")

    print("To verify LOCAL vs GLOBAL rotation:")
    print("- Create cross sections with DIFFERENT initial orientations")
    print("- Apply FLIP90 - each should rotate in its own local plane")
    print("- Result: different cross sections rotate differently")
    print("")

    print("The current FLIP90 behavior may be correct - it rotates each cross section")
    print("locally by +90°, but since they all start aligned, the result looks global.")

if __name__ == "__main__":
    analyze_rotation_issue()