#!/usr/bin/env python3
"""
Clear demonstration of rotation differences: baseline, plan rotation (XY yaw), and twist.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from run_deck import main as run_deck_main

def demo_rotation_comparison():
    """Create clear HTML files showing baseline, plan rotation, and twist behaviors."""

    print("=== Rotation Analysis Demo ===\n")

    print("1. NORMAL ORIENTATION (no rotation):")
    print("   - Shows baseline cross section orientation")
    print("   - Output: deck_normal.html")
    run_deck_main(
        axis_json="GIT/MAIN/_Axis_JSON.json",
        cross_json="GIT/MAIN/_CrossSection_JSON.json",
        obj_json="GIT/MAIN/_DeckObject_JSON.json",
        mainstation_json="GIT/MAIN/_MainStation_JSON.json",
        section_json="MASTER_SECTION/MASTER_DeckCrBm-1Gird-Slab.json",
        out_html="deck_normal.html",
        plan_rotation_deg=0.0
    )

    print("\n2. PLAN ROTATION (yaw +90°):")
    print("   - Apply +90° yaw in XY to the section orientation")
    print("   - Output: deck_plan_rot_90.html")
    run_deck_main(
        axis_json="GIT/MAIN/_Axis_JSON.json",
        cross_json="GIT/MAIN/_CrossSection_JSON.json",
        obj_json="GIT/MAIN/_DeckObject_JSON.json",
        mainstation_json="GIT/MAIN/_MainStation_JSON.json",
        section_json="MASTER_SECTION/MASTER_DeckCrBm-1Gird-Slab.json",
        out_html="deck_plan_rot_90.html",
        plan_rotation_deg=90.0
    )

    print("\n3. ALTERNATIVE: +90° TWIST (gradual rotation):")
    print("   - Uses twist_deg=+90 instead of rotation_override_deg")
    print("   - May show different rotation behavior")
    print("   - Output: deck_twist_90.html")
    run_deck_main(
        axis_json="GIT/MAIN/_Axis_JSON.json",
        cross_json="GIT/MAIN/_CrossSection_JSON.json",
        obj_json="GIT/MAIN/_DeckObject_JSON.json",
        mainstation_json="GIT/MAIN/_MainStation_JSON.json",
        section_json="MASTER_SECTION/MASTER_DeckCrBm-1Gird-Slab.json",
        out_html="deck_twist_90.html",
        twist_deg=90.0  # Use twist instead of flip
    )

    print("\n=== Analysis Complete ===")
    print("Generated files for comparison:")
    print("- deck_normal.html (baseline - no rotation)")
    print("- deck_plan_rot_90.html (plan rotation yaw +90°)")
    print("- deck_twist_90.html (twist-based rotation along length)")
    print("\nCompare these files to understand the rotation behavior!")

if __name__ == "__main__":
    demo_rotation_comparison()