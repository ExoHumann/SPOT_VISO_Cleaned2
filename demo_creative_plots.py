#!/usr/bin/env python3
"""
Creates a set of creative deck plots:
- Baseline (no rotations)
- Plan rotation sweep (XY yaw)
- Twist sweep (helical, in-plane YZ)
- Combined plan + twist
Across MAIN, RCZ_new1, and MIXED (MAIN axis + RCZ cross sections).
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from run_deck import main as run_deck_main


def _run_case(*, name: str,
              axis_json: str, cross_json: str, obj_json: str, mainstation_json: str,
              section_json: str = "MASTER_SECTION/MASTER_DeckCrBm-1Gird-Slab.json",
              plan_rotation_deg: float = 0.0, twist_deg: float = 0.0,
              out_dir: str = ".",
              max_stations: int = 400, loop_stride: int = 20, long_stride: int = 50) -> None:
    out_html = os.path.join(out_dir, f"{name}.html")
    print(f"- Generating: {out_html} (plan={plan_rotation_deg}°, twist={twist_deg}°)")
    run_deck_main(
        axis_json=axis_json,
        cross_json=cross_json,
        obj_json=obj_json,
        mainstation_json=mainstation_json,
        section_json=section_json,
        out_html=out_html,
        max_stations=max_stations,
        loop_stride=loop_stride,
        long_stride=long_stride,
        twist_deg=twist_deg,
        plan_rotation_deg=plan_rotation_deg,
    )


def main():
    # Sources
    main_src = dict(
        axis_json="GIT/MAIN/_Axis_JSON.json",
        cross_json="GIT/MAIN/_CrossSection_JSON.json",
        obj_json="GIT/MAIN/_DeckObject_JSON.json",
        mainstation_json="GIT/MAIN/_MainStation_JSON.json",
    )
    rcz_src = dict(
        axis_json="GIT/RCZ_new1/_Axis_JSON.json",
        cross_json="GIT/RCZ_new1/_CrossSection_JSON.json",
        obj_json="GIT/RCZ_new1/_DeckObject_JSON.json",
        mainstation_json="GIT/RCZ_new1/_MainStation_JSON.json",
    )
    mixed_src = dict(
        axis_json="GIT/MAIN/_Axis_JSON.json",                 # MAIN axis
        cross_json="GIT/RCZ_new1/_CrossSection_JSON.json",    # RCZ cross-sections
        obj_json="GIT/MAIN/_DeckObject_JSON.json",
        mainstation_json="GIT/MAIN/_MainStation_JSON.json",
    )

    # Output folder
    out_dir = "."
    print("=== Creative Deck Plots ===")

    # 1) Baselines
    _run_case(name="deck_main_baseline", **main_src, plan_rotation_deg=0, twist_deg=0, out_dir=out_dir)
    _run_case(name="deck_rcz_baseline", **rcz_src, plan_rotation_deg=0, twist_deg=0, out_dir=out_dir)
    _run_case(name="deck_mixed_baseline", **mixed_src, plan_rotation_deg=0, twist_deg=0, out_dir=out_dir)

    # 2) Plan rotation sweeps (XY yaw, no twist)
    for ang in (15, 30, 45, 60, 90):
        _run_case(name=f"deck_main_plan_{ang:03d}", **main_src, plan_rotation_deg=ang, twist_deg=0, out_dir=out_dir)
    for ang in (15, 30, 45, 60, 90):
        _run_case(name=f"deck_rcz_plan_{ang:03d}", **rcz_src, plan_rotation_deg=ang, twist_deg=0, out_dir=out_dir)
    for ang in (15, 30, 45, 60, 90):
        _run_case(name=f"deck_mixed_plan_{ang:03d}", **mixed_src, plan_rotation_deg=ang, twist_deg=0, out_dir=out_dir)

    # 3) Twist sweeps (helical, in-plane YZ, no plan yaw)
    for ang in (15, 30, 45):
        _run_case(name=f"deck_main_twist_{ang:03d}", **main_src, plan_rotation_deg=0, twist_deg=ang, out_dir=out_dir)
    for ang in (15, 30, 45):
        _run_case(name=f"deck_rcz_twist_{ang:03d}", **rcz_src, plan_rotation_deg=0, twist_deg=ang, out_dir=out_dir)
    for ang in (15, 30, 45):
        _run_case(name=f"deck_mixed_twist_{ang:03d}", **mixed_src, plan_rotation_deg=0, twist_deg=ang, out_dir=out_dir)

    # 4) Combined plan + twist (few expressive combos)
    combos = [
        (30, 15),
        (45, 30),
        (-30, 45),
    ]
    for plan, twist in combos:
        _run_case(name=f"deck_main_plan_{plan:+03d}_twist_{twist:03d}".replace("+", "p").replace("-", "m"),
                  **main_src, plan_rotation_deg=plan, twist_deg=twist, out_dir=out_dir)
        _run_case(name=f"deck_rcz_plan_{plan:+03d}_twist_{twist:03d}".replace("+", "p").replace("-", "m"),
                  **rcz_src, plan_rotation_deg=plan, twist_deg=twist, out_dir=out_dir)
        _run_case(name=f"deck_mixed_plan_{plan:+03d}_twist_{twist:03d}".replace("+", "p").replace("-", "m"),
                  **mixed_src, plan_rotation_deg=plan, twist_deg=twist, out_dir=out_dir)

    print("\n=== Done ===")
    print("Open any of the generated HTML files to compare:")
    print("- Baselines: deck_*_baseline.html")
    print("- Plan yaw: deck_*_plan_XXX.html")
    print("- Twist:    deck_*_twist_XXX.html")
    print("- Combined: deck_*_plan_±XXX_twist_YYY.html")


if __name__ == "__main__":
    main()
