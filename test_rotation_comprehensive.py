#!/usr/bin/env python3
"""
Comprehensive rotation testing for LinearObject classes.
Tests twist and plan rotation (XY yaw) to verify rotation is working.
"""

import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from run_deck import main as run_deck_main

def test_rotation_values():
    """Test different rotation values to verify rotation is working."""

    print("=== Comprehensive Rotation Testing ===\n")

    base_args = {
        "axis_json": "GIT/MAIN/_Axis_JSON.json",
        "cross_json": "GIT/MAIN/_CrossSection_JSON.json",
        "obj_json": "GIT/MAIN/_DeckObject_JSON.json",
        "mainstation_json": "GIT/MAIN/_MainStation_JSON.json",
        "section_json": "MASTER_SECTION/MASTER_DeckCrBm-1Gird-Slab.json",
    }

    # Test different twist_deg values (creates helical twisting along bridge)
    twist_tests = [
        (0.0, "deck_twist_000.html", "0° twist (baseline)"),
        (45.0, "deck_twist_045.html", "45° twist"),
        (90.0, "deck_twist_090.html", "90° twist"),
        (135.0, "deck_twist_135.html", "135° twist"),
        (180.0, "deck_twist_180.html", "180° twist"),
        (270.0, "deck_twist_270.html", "270° twist"),
    ]

    print("Testing different twist_deg values (helical twisting along bridge):")
    for twist_deg, filename, description in twist_tests:
        print(f"\n{description} -> {filename}")
        run_deck_main(
            axis_json=base_args["axis_json"],
            cross_json=base_args["cross_json"],
            obj_json=base_args["obj_json"],
            mainstation_json=base_args["mainstation_json"],
            section_json=base_args["section_json"],
            out_html=filename,
            twist_deg=twist_deg
        )

    # Test different plan_rotation_deg values (rotates cross-section orientation in XY plane only)
    plan_rotation_tests = [
        (0.0, "deck_plan_rot_000.html", "0° plan rotation (baseline)"),
        (30.0, "deck_plan_rot_030.html", "30° plan rotation (yaw)"),
        (60.0, "deck_plan_rot_060.html", "60° plan rotation (yaw)"),
        (90.0, "deck_plan_rot_090.html", "90° plan rotation (yaw)"),
        (120.0, "deck_plan_rot_120.html", "120° plan rotation (yaw)"),
        (150.0, "deck_plan_rot_150.html", "150° plan rotation (yaw)"),
    ]

    print("\nTesting different plan_rotation_deg values (rotate cross-section orientation in XY plane only):")
    for plan_rot, filename, description in plan_rotation_tests:
        print(f"\n{description} -> {filename}")
        run_deck_main(
            axis_json=base_args["axis_json"],
            cross_json=base_args["cross_json"],
            obj_json=base_args["obj_json"],
            mainstation_json=base_args["mainstation_json"],
            section_json=base_args["section_json"],
            out_html=filename,
            plan_rotation_deg=plan_rot
        )

    print("\n=== Rotation Testing Complete ===")
    print("Generated files:")
    print("Twist tests (helical twisting):")
    for _, filename, desc in twist_tests:
        print(f"  - {filename}: {desc}")
    print("Plan rotation tests (XY yaw only):")
    for _, filename, desc in plan_rotation_tests:
        print(f"  - {filename}: {desc}")

    print("\n=== Analysis Instructions ===")
    print("1. Twist files: Cross-sections should rotate progressively along bridge length")
    print("   (helical twisting effect)")
    print("2. Plan rotation files: Orientation should yaw in XY, vertical alignment unchanged")
    print("3. If all cross sections look identical, rotation is NOT working!")
    print("4. Plan rotation should only change yaw (XY orientation), not introduce extra twist")

if __name__ == "__main__":
    test_rotation_values()