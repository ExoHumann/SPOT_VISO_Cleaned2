# run_foundation.py - Test script for FoundationObject with different data sources
from __future__ import annotations
import json, argparse, plotly.graph_objects as go
import os
from models.base import (
    load_axis_from_rows,
    index_cross_sections_by_ncs, load_section_for_ncs,
    cs_steps_from_deck_row,
    from_dict,
)
from models.main_station import load_mainstations_from_rows
from models.foundation_object import FoundationObject
from models.mapping import mapping
from main import get_plot_traces_matrix

# Import streamlined builder
from linear_object_builder import LinearObjectBuilder

def main(axis_json, cross_json, obj_json, mainstation_json, section_json, out_html,
         obj_type="FoundationObject", max_stations=400, loop_stride=20, long_stride=50,
         twist_deg=0.0, plan_rotation_deg: float = 0.0):

    # Use streamlined LinearObjectBuilder approach
    builder = LinearObjectBuilder(verbose=True)
    builder.load_from_files(axis_json, cross_json, obj_json, mainstation_json, section_json)
    
    try:
        # Try to create foundation using streamlined workflow
        obj = builder.create_object(obj_type)
        
        # Build geometry using standardized method
        res = builder.build_geometry(
            obj,
            stations_m=None,
            twist_deg=float(twist_deg),
            plan_rotation_deg=float(plan_rotation_deg), 
            station_cap=max_stations,
            frame_mode="symmetric"
        )
        
    except Exception as e:
        print(f"Streamlined approach failed ({e}), falling back to legacy method...")
        
        # Fallback to legacy method for compatibility
        # Load JSON data
        axis_rows = json.load(open(axis_json, "r", encoding="utf-8"))
        cross_rows = json.load(open(cross_json, "r", encoding="utf-8"))
        obj_rows  = json.load(open(obj_json,  "r", encoding="utf-8"))
        ms_rows    = json.load(open(mainstation_json, "r", encoding="utf-8"))

        obj_row = next(r for r in obj_rows if r.get("Class") == obj_type)

        # Load all available components
        available_axes = {}
        for axis_row in axis_rows:
            if axis_row.get("Class") == "Axis":
                axis_name = axis_row.get("Name", "")
                if axis_name:
                    available_axes[axis_name] = load_axis_from_rows(axis_rows, axis_name)

        available_cross_sections = {}
        by_ncs = index_cross_sections_by_ncs(cross_rows)
        for ncs in by_ncs.keys():
            available_cross_sections[ncs] = load_section_for_ncs(ncs, by_ncs, section_json)

        available_mainstations = {}
        for axis_name in available_axes.keys():
            available_mainstations[axis_name] = load_mainstations_from_rows(ms_rows, axis_name=axis_name)

        # Create object from JSON data using mapping
        obj = from_dict(FoundationObject, obj_row, mapping)

        print(f"FoundationObject created with axis_name: {obj.axis_name}")
        print(f"Available axes: {list(available_axes.keys())}")
        print(f"Available cross sections: {list(available_cross_sections.keys())}")

        # Configure object with available components (let the object decide what to use)
        obj.configure(available_axes, available_cross_sections, available_mainstations)

        print(f"Configured with axis: {obj.axis_obj}")
        print(f"Configured with {len(obj.sections_by_ncs) if obj.sections_by_ncs else 0} cross sections")

        res = obj.build(
            stations_m=None,
            twist_deg=float(twist_deg),
            plan_rotation_deg=float(plan_rotation_deg),
            station_cap=max_stations,
            frame_mode= "symmetric"
        )

    traces, *_ = get_plot_traces_matrix(
        res["axis"], res["section_json"], res["stations_mm"],
        res["ids"], res["points_world_mm"],
        X_mm=res["local_Y_mm"], Y_mm=res["local_Z_mm"],
        loops_idx=res["loops_idx"],
        overlays=res.get("overlays"),
        loops_only_from_overlays=False,
        show_points=False,
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"FoundationObject - {os.path.basename(axis_json)} / {os.path.basename(cross_json)}",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
        )
    )
    fig.write_html(out_html)
    print(f"Saved: {out_html}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Test FoundationObject with different data sources")
    ap.add_argument("--axis", default="GIT/MAIN/_Axis_JSON.json", help="Axis JSON file")
    ap.add_argument("--cross", default="GIT/MAIN/_CrossSection_JSON.json", help="Cross section JSON file")
    ap.add_argument("--obj", default="GIT/MAIN/_FoundationObject_JSON.json", help="FoundationObject JSON file")
    ap.add_argument("--main", default="GIT/MAIN/_MainStation_JSON.json", help="Main station JSON file")
    ap.add_argument("--section", default="MASTER_SECTION/MASTER_Foundation.json", help="Section JSON file")
    ap.add_argument("--out", default="foundation_test.html", help="Output HTML file")
    ap.add_argument("--max-stations", type=int, default=400)
    ap.add_argument("--loop-stride", type=int, default=20)
    ap.add_argument("--long-stride", type=int, default=50)
    ap.add_argument("--twist", type=float, default=0.0, help="Extra in-plane rotation (deg)")
    ap.add_argument("--plan-rotation", type=float, default=0.0, help="Rotate cross-section orientation in XY plane (yaw, deg)")

    args = ap.parse_args()

    main(args.axis, args.cross, args.obj, args.main, args.section, args.out,
        max_stations=args.max_stations, loop_stride=args.loop_stride, long_stride=args.long_stride,
        twist_deg=args.twist, plan_rotation_deg=args.plan_rotation)