#!/usr/bin/env python3
"""
Demo script showing the flexibility of the modular architecture
by mixing components from different data sources.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from run_deck import main as run_deck_main

def demo_mixed_sources():
    """Demonstrate the flexibility of the modular architecture
    with plan (XY yaw) rotation control."""

    print("=== SPOT VISO Modular Architecture Demo ===\n")

    print("1. Testing DeckObject with MAIN data:")
    print("   - Axis: MAIN (35 stations)")
    print("   - Cross sections: MAIN")
    print("   - Plan rotation: 0°")
    print("   - Output: deck_main_demo.html")
    run_deck_main(
        axis_json="GIT/MAIN/_Axis_JSON.json",
        cross_json="GIT/MAIN/_CrossSection_JSON.json",
        obj_json="GIT/MAIN/_DeckObject_JSON.json",
        mainstation_json="GIT/MAIN/_MainStation_JSON.json",
        section_json="MASTER_SECTION/MASTER_DeckCrBm-1Gird-Slab.json",
        out_html="deck_main_demo.html",
        plan_rotation_deg=0.0
    )

    print("\n2. Testing DeckObject with RCZ_new1 data:")
    print("   - Axis: RCZ_new1 (4 stations)")
    print("   - Cross sections: RCZ_new1 (includes NCS 121)")
    print("   - Plan rotation: 0°")
    print("   - Output: deck_rcz_demo.html")
    run_deck_main(
        axis_json="GIT/RCZ_new1/_Axis_JSON.json",
        cross_json="GIT/RCZ_new1/_CrossSection_JSON.json",
        obj_json="GIT/RCZ_new1/_DeckObject_JSON.json",
        mainstation_json="GIT/RCZ_new1/_MainStation_JSON.json",
        section_json="MASTER_SECTION/MASTER_DeckCrBm-1Gird-Slab.json",
        out_html="deck_rcz_demo.html",
        plan_rotation_deg=0.0
    )

    print("\n3. Testing DeckObject with MIXED sources:")
    print("   - Axis: MAIN (35 stations)")
    print("   - Cross sections: RCZ_new1 (includes NCS 121)")
    print("   - Plan rotation: 30°")
    print("   - Output: deck_mixed_demo.html")
    run_deck_main(
        axis_json="GIT/MAIN/_Axis_JSON.json",  # MAIN axis
        cross_json="GIT/RCZ_new1/_CrossSection_JSON.json",  # RCZ cross sections
        obj_json="GIT/MAIN/_DeckObject_JSON.json",
        mainstation_json="GIT/MAIN/_MainStation_JSON.json",
        section_json="MASTER_SECTION/MASTER_DeckCrBm-1Gird-Slab.json",
        out_html="deck_mixed_demo.html",
        plan_rotation_deg=30.0
    )

    print("\n=== Demo Complete ===")
    print("Generated files:")
    print("- deck_main_demo.html (MAIN data)")
    print("- deck_rcz_demo.html (RCZ_new1 data)")
    print("- deck_mixed_demo.html (Mixed MAIN/RCZ_new1)")
    print("\nThis demonstrates how objects can flexibly use different")
    print("axes and cross sections from various data sources,")
    print("with explicit control over plan (XY) orientation.")

if __name__ == "__main__":
    demo_mixed_sources()