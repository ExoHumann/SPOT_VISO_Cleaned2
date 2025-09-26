"""Minimal pier runner.

Loads a single axis, cross-section, pier object row set and produces a 3D HTML plot.
Assumes JSON inputs already point to the desired data (no auto-discovery, no deck, no overlays).

Updated to use the streamlined LinearObjectBuilder workflow.
"""
from __future__ import annotations
import json, argparse, os, plotly.graph_objects as go, numpy as np
from models.base import load_axis_from_rows, index_cross_sections_by_ncs, load_section_for_ncs, simple_load_piers
from models.main_station import load_mainstations_from_rows
from models.pier_object import PierObject
from models.plotter import Plotter, PlotConfig

# Import streamlined builder
from linear_object_builder import LinearObjectBuilder

def run(axis_json: str, cross_json: str, obj_json: str, main_json: str, section_json: str, out_html: str,
        twist_deg: float = 0.0, plan_rotation_deg: float = 0.0, aspect_equal: bool = False,
        debug_units: bool = False, debug_embed: bool = False) -> None:
    
    # Use streamlined LinearObjectBuilder approach
    builder = LinearObjectBuilder(verbose=True)
    builder.load_from_files(axis_json, cross_json, obj_json, main_json, section_json)
    
    try:
        # Try to create pier using streamlined workflow
        pier = builder.create_object("PierObject")
        
        # Build geometry using standardized method
        build = builder.build_geometry(
            pier,
            twist_deg=twist_deg,
            plan_rotation_deg=plan_rotation_deg
        )
        
    except Exception as e:
        print(f"Streamlined approach failed ({e}), falling back to legacy method...")
        
        # Fallback to legacy method for compatibility
        axis_rows = json.load(open(axis_json, 'r', encoding='utf-8'))
        cross_rows = json.load(open(cross_json, 'r', encoding='utf-8'))
        pier_rows  = json.load(open(obj_json,  'r', encoding='utf-8'))
        ms_rows    = json.load(open(main_json, 'r', encoding='utf-8'))

        # Load components
        axes = {r.get('Name'): load_axis_from_rows(axis_rows, r.get('Name'))
                for r in axis_rows if r.get('Class') == 'Axis' and r.get('Name')}
        by_ncs = index_cross_sections_by_ncs(cross_rows)
        cross_sections = {ncs: load_section_for_ncs(ncs, by_ncs, section_json) for ncs in by_ncs.keys()}
        mainstations = {name: load_mainstations_from_rows(ms_rows, axis_name=name) for name in axes.keys()}

        piers: list[PierObject] = simple_load_piers(pier_rows)
        if not piers:
            # synthesize single pier
            p = PierObject(); p.name = 'SyntheticPier'
            if axes:
                p.axis_name = next(iter(axes.keys()))
            if cross_sections:
                key = next(iter(cross_sections.keys()))
                p.top_cross_section_ncs = key
                p.base_section = cross_sections[key]
            piers = [p]

        # Only process first pier for minimal script
        pier = piers[0]
        pier.configure(axes, cross_sections, mainstations)
        if getattr(pier, 'base_section', None) is None and cross_sections:
            pier.base_section = next(iter(cross_sections.values()))
        pier.configure_pier_axis()
        if getattr(pier, 'axis_obj', None) is None:
            raise RuntimeError('Pier axis not configured')

        build = pier.build(manual=True, deck_axis=None, vertical_slices=6,
                           station_value_m=float(getattr(pier, 'station_value', 0.0) or 0.0),
                           twist_deg=twist_deg, plan_rotation_deg=plan_rotation_deg,
                           debug_embed=debug_embed, debug_units=debug_units)

    cfg = PlotConfig(show_points=True, show_loops=True, show_loop_points=False, show_labels=False)
    plotter = Plotter(build.get('pier_axis'), obj_name=pier.name or pier.axis_name or 'Pier', config=cfg)
    traces = plotter.build_traces(
        json_data=None,
        stations_mm=build['stations_mm'],
        ids=build['ids'],
        P_mm=build['points_world_mm'],
        X_mm=build['local_Y_mm'],
        Y_mm=build['local_Z_mm'],
        loops_idx=build['loops_idx'],
        overlays=None,
    )

    # Bounds print (optional quick diagnostic)
    Pm = build['points_world_mm'] / 1000.0
    valid = ~np.isnan(Pm)
    if valid.any():
        xs = Pm[:,:,0][valid[:,:,0]]; ys = Pm[:,:,1][valid[:,:,1]]; zs = Pm[:,:,2][valid[:,:,2]]
        print(f"[bounds:m] X:[{xs.min():.3f},{xs.max():.3f}] span={xs.max()-xs.min():.3f}")
        print(f"[bounds:m] Y:[{ys.min():.3f},{ys.max():.3f}] span={ys.max()-ys.min():.3f}")
        print(f"[bounds:m] Z:[{zs.min():.3f},{zs.max():.3f}] span={zs.max()-zs.min():.3f}")

    fig = go.Figure(data=traces)
    if aspect_equal:
        scene = dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)', aspectmode='data')
    else:
        scene = dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)', aspectratio={'x':1,'y':1,'z':1})
    fig.update_layout(title=f"Pier - {os.path.basename(axis_json)}", scene=scene)
    fig.write_html(out_html)
    print(f"Saved: {out_html}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Minimal Pier Plot')
    ap.add_argument('--axis', default='GIT/RCZ_new1/_Axis_JSON.json')
    ap.add_argument('--cross', default='GIT/RCZ_new1/_CrossSection_JSON.json')
    ap.add_argument('--obj', default='GIT/RCZ_new1/_PierObject_JSON.json')
    ap.add_argument('--main', default='GIT/RCZ_new1/_MainStation_JSON.json')
    ap.add_argument('--section', default='MASTER_SECTION/MASTER_Pier.json')
    ap.add_argument('--out', default='pier_plot.html')
    ap.add_argument('--twist', type=float, default=0.0)
    ap.add_argument('--plan-rotation', type=float, default=0.0)
    ap.add_argument('--aspect-equal', action='store_true')
    ap.add_argument('--debug-units', action='store_true')
    ap.add_argument('--debug-embed', action='store_true')
    a = ap.parse_args()
    run(a.axis, a.cross, a.obj, a.main, a.section, a.out,
        twist_deg=a.twist, plan_rotation_deg=a.plan_rotation,
        aspect_equal=a.aspect_equal, debug_units=a.debug_units, debug_embed=a.debug_embed)
