# run_deck.py - Test script for DeckObject with different data sources
from __future__ import annotations
import json, argparse, plotly.graph_objects as go
import numpy as np
import os
from models.base import (
    load_axis_from_rows,
    index_cross_sections_by_ncs, load_section_for_ncs,
    cs_steps_from_deck_row,
    from_dict,
)
from models.main_station import load_mainstations_from_rows
from models.deck_object import DeckObject
from models.mapping import mapping
from main import get_plot_traces_matrix
from models.plotter import Plotter, PlotConfig

def main(axis_json, cross_json, obj_json, mainstation_json, section_json, out_html,
         obj_type="DeckObject", max_stations=400, loop_stride=20, long_stride=50,
         twist_deg=0.0, plan_rotation_deg=0.0, frame_mode="symmetric", rotation_mode="additive",
         show_frames: bool = False, frame_stride: int = 1, frame_scale: float = 5.0):

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
    obj = from_dict(DeckObject, obj_row, mapping)

    print(f"DeckObject created with axis_name: {obj.axis_name}")
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
        frame_mode=frame_mode,
        rotation_mode=rotation_mode,
    )

    # Use new modular Plotter (still keep legacy for comparison if desired)
    # Do not show separate loop point markers because cross-section loop line hover already exposes point metadata
    cfg = PlotConfig(show_points=True, show_loop_points=False, show_labels=False)
    plotter = Plotter(res["axis"], obj_name=obj.name or "Deck", config=cfg)
    traces = plotter.build_traces(
        json_data=res["section_json"],
        stations_mm=res["stations_mm"],
        ids=res["ids"],
        P_mm=res["points_world_mm"],
        X_mm=res["local_Y_mm"],
        Y_mm=res["local_Z_mm"],
        loops_idx=res["loops_idx"],
        overlays=res.get("overlays"),
    )

    # Optional frame overlay (tangent, N_yaw, B_yaw) every frame_stride stations
    if show_frames and res.get("axis") is not None:
        axis_obj = res["axis"]
        stations_mm = res["stations_mm"]
        if frame_stride < 1: frame_stride = 1
        pick_idx = np.arange(0, len(stations_mm), frame_stride)
        if len(stations_mm) > 0 and pick_idx[-1] != len(stations_mm) - 1:
            pick_idx = np.append(pick_idx, len(stations_mm) - 1)
        mode_for_frames = frame_mode
        plan_arr = res.get("plan_rotation_deg_array")
        if isinstance(plan_arr, np.ndarray):
            plan_for_frames = plan_arr[pick_idx]
        elif isinstance(plan_arr, list):
            plan_for_frames = np.asarray(plan_arr, float)[pick_idx]
        else:
            plan_for_frames = 0.0
        frame_data = axis_obj.frame_at_stations(stations_mm[pick_idx], mode=mode_for_frames, plan_rotation_deg=plan_for_frames)
        P = frame_data["P"] / 1000.0
        T = frame_data["T"]
        N_y = frame_data["N_yaw"]
        B_y = frame_data["B_yaw"]
        scale = float(frame_scale)
        vec_info = [
            (T, 'Tangent T', 'red'),
            (N_y, 'Normal N(yaw)', 'green'),
            (B_y, 'Binormal B(yaw)', 'blue')
        ]
        for V, label, color in vec_info:
            x_seg=[]; y_seg=[]; z_seg=[]; meta=[]  # meta: [station_m, vec_type, dX, dY, dZ, length]
            for i in range(len(P)):
                p = P[i]; v = V[i] * scale / 1000.0  # convert scaled mm to m
                x0,y0,z0 = p
                x1,y1,z1 = p + v
                x_seg += [x0, x1, None]
                y_seg += [y0, y1, None]
                z_seg += [z0, z1, None]
                dX, dY, dZ = v
                length = float(np.linalg.norm(v))
                st_m = float(stations_mm[pick_idx[i]] / 1000.0)
                meta.append([st_m, label, float(dX), float(dY), float(dZ), length])
                meta.append([None]*6)  # separator for segment end
            traces.append(go.Scatter3d(
                x=x_seg, y=y_seg, z=z_seg,
                mode='lines',
                line=dict(color=color, width=3),
                name=label,
                customdata=meta,
                hovertemplate=(
                    '<b>'+label+'</b><br>'
                    'Station: %{customdata[0]:.3f} m<br>'
                    'dX: %{customdata[2]:.4f} m<br>'
                    'dY: %{customdata[3]:.4f} m<br>'
                    'dZ: %{customdata[4]:.4f} m<br>'
                    'Length: %{customdata[5]:.4f} m<extra></extra>'
                )
            ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"DeckObject - {os.path.basename(axis_json)} / {os.path.basename(cross_json)}",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode='data'
        )
    )
    fig.write_html(out_html)
    print(f"Saved: {out_html}")

if __name__ == "__main__":
    base="C:\\Users\\KrzyS\\OneDrive\\Skrivebord\\Visio\\SPOT_VISO_Cleaned\\SPOT_VISO\\"
    ap = argparse.ArgumentParser(description="Test DeckObject with different data sources")
    ap.add_argument("--axis", default=base+"GIT\\RCZ_new1\\_Axis_JSON.json", help="Axis JSON file")
    ap.add_argument("--cross", default=base+"GIT\\RCZ_new1\\_CrossSection_JSON.json", help="Cross section JSON file")
    ap.add_argument("--obj", default=base+"GIT\\RCZ_new1\\_DeckObject_JSON.json", help="DeckObject JSON file")
    ap.add_argument("--main", default=base+"GIT\\RCZ_new1\\_MainStation_JSON.json", help="Main station JSON file")
    ap.add_argument("--section", default=base+"MASTER_SECTION\\SectionData.json", help="Section JSON file")
    ap.add_argument("--out", default=base+"deck_test.html", help="Output HTML file")
    ap.add_argument("--max-stations", type=int, default=400)
    ap.add_argument("--loop-stride", type=int, default=20)
    ap.add_argument("--long-stride", type=int, default=50)
    ap.add_argument("--twist", type=float, default=0, help="Extra in-plane rotation (deg)")
    ap.add_argument("--plan-rotation", type=float, default=0.0, help="Rotate cross-section orientation in XY plane (yaw, deg)")
    ap.add_argument("--frame-mode", choices=["symmetric", "pt"], default="symmetric", help="Frame construction method: symmetric middle-plane or parallel transport (pt)")
    ap.add_argument("--rotation-mode", choices=["additive", "absolute"], default="additive", help="How per-station rotations combine with global twist/plan rotations")
    ap.add_argument("--show-frames", action="store_true", help="Overlay sampled frames (T,N,B) vectors for debug")
    ap.add_argument("--frame-stride", type=int, default=1, help="Stride (in station index) for frame sampling")
    ap.add_argument("--frame-scale", type=float, default=5.0, help="Vector length scale (mm) for frame arrows")

    args = ap.parse_args()

    main(
        args.axis, args.cross, args.obj, args.main, args.section, args.out,
        max_stations=args.max_stations, loop_stride=args.loop_stride, long_stride=args.long_stride,
        twist_deg=args.twist, plan_rotation_deg=args.plan_rotation,
        frame_mode=args.frame_mode, rotation_mode=args.rotation_mode,
        show_frames=args.show_frames, frame_stride=args.frame_stride, frame_scale=args.frame_scale,
    )