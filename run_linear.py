# run_linear.py
from __future__ import annotations
import json, argparse, plotly.graph_objects as go
import os
from models.base import (
    load_axis_from_rows,
    index_cross_sections_by_ncs, load_section_for_ncs,
    cs_steps_from_deck_row,
    from_dict,
)
from models.deck_object import DeckObject
from models.pier_object import PierObject
from models.foundation_object import FoundationObject
from models.main_station import load_mainstations_from_rows
from models.mapping import mapping
from main import get_plot_traces_matrix  # your existing plotter

def main(axis_json, cross_json, obj_json, mainstation_json, section_json, out_html,
         *, obj_type="DeckObject", max_stations=400, loop_stride=20, long_stride=50, twist_deg=0.0, plan_rotation_deg: float = 0.0):

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

    # Determine object class and mapping based on obj_type
    if obj_type == "DeckObject":
        obj_class = DeckObject
        obj_mapping = mapping["DeckObject"]
    elif obj_type == "PierObject":
        obj_class = PierObject
        obj_mapping = mapping["PierObject"]
    elif obj_type == "FoundationObject":
        obj_class = FoundationObject
        obj_mapping = mapping["FoundationObject"]
    else:
        raise ValueError(f"Unsupported object type: {obj_type}")

    # Create object from JSON data using mapping
    obj = from_dict(obj_class, obj_row, mapping)
    
    # Configure object with available components (let the object decide what to use)
    obj.configure(available_axes, available_cross_sections, available_mainstations)

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
        overlays=res.get("overlays"),           # NEW
        loops_only_from_overlays=False,          # show only MS/NCS-start CS slices
        show_points=False,                      # we add our own blue points via overlays
        station_stride_for_loops=loop_stride,
        longitudinal_stride=long_stride,
        include_first_station_ids_in_longitudinal=False,  # NEW
        filter_longitudinal_to_loops=True,
    )


    fig = go.Figure(traces)
    fig.update_layout(
        title=f"{obj_type} — {res['name']}",
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)', aspectmode='data'),
        template='plotly_white', margin=dict(l=0, r=0, t=40, b=0)
    )
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"Saved: {out_html}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True)
    ap.add_argument("--cross", required=True)
    ap.add_argument("--obj", required=True, help="JSON file containing object data (DeckObject, PierObject, or FoundationObject)")
    ap.add_argument("--main", required=True)
    ap.add_argument("--section", required=True)
    ap.add_argument("--obj-type", default="DeckObject", choices=["DeckObject", "PierObject", "FoundationObject"], help="Type of object to create")
    ap.add_argument("--out", default="obj_plot.html")
    ap.add_argument("--max-stations", type=int, default=400)
    ap.add_argument("--loop-stride", type=int, default=20)
    ap.add_argument("--long-stride", type=int, default=50)
    ap.add_argument("--twist", type=float, default=0.0, help="Extra in-plane rotation (deg)")
    ap.add_argument("--plan-rotation", type=float, default=0.0, help="Rotate cross-section orientation in XY plane (yaw, deg)")


    args = ap.parse_args()

    main(args.axis, args.cross, args.obj, args.main, args.section, args.out,
        obj_type=args.obj_type, max_stations=args.max_stations, loop_stride=args.loop_stride, long_stride=args.long_stride,
        twist_deg=args.twist, plan_rotation_deg=args.plan_rotation)


# Example usage:
# cd C:\Users\KrzyS\OneDrive\Skrivebord\Visio\SPOT_VISO_Cleaned\SPOT_VISO

# Deck
#  py -3.12 run_linear.py `
#    --axis "GIT/RCZ_new1/_Axis_JSON.json" `
#    --cross "GIT/RCZ_new1/_CrossSection_JSON.json" `
#    --obj "GIT/RCZ_new1/_DeckObject_JSON.json" `
#    --main "GIT/RCZ_new1/_MainStation_JSON.json" `
#    --section "MASTER_SECTION/SectionData.json" `
#    --obj-type "DeckObject" `
#    --out "deck_plot.html" `
#    --max-stations 1000 `
#    --loop-stride 1 `
#    --long-stride 1 `
#    --flip90

# Pier
#  py -3.12 run_linear.py `
#    --axis "GIT/RCZ_new1/_Axis_JSON.json" `
#    --cross "GIT/RCZ_new1/_CrossSection_JSON.json" `
#    --obj "GIT/RCZ_new1/_PierObject_JSON.json" `
#    --main "GIT/RCZ_new1/_MainStation_JSON.json" `
#    --section "MASTER_SECTION/MASTER_Pier.json" `
#    --obj-type "PierObject" `
#    --out "pier_plot.html" `
#    --max-stations 100 `
#    --loop-stride 1 `
#    --long-stride 1

# Foundation
# py -3.12 run_linear.py `
#   --axis "GIT/RCZ_new1/_Axis_JSON.json" `
#   --cross "GIT/RCZ_new1/_CrossSection_JSON.json" `
#   --obj "GIT/RCZ_new1/_FoundationObject_JSON.json" `
#   --main "GIT/RCZ_new1/_MainStation_JSON.json" `
#   --section "MASTER_SECTION/MASTER_Foundation.json" `
#   --obj-type "FoundationObject" `
#   --out "foundation_plot.html" `
#   --max-stations 100 `
#   --loop-stride 1 `
#   --long-stride 1
