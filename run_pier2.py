"""Pier runner v2 (simplified)

Key points:
    * Sequencing (multi NCS switching) is now DATA-DRIVEN only.
        The loader (`simple_load_piers`) builds `pier.ncs_steps` from the JSON
        (Top / Internal offsets / Bottom). No CLI flag (old --pier-ncs) is needed
        or supported anymore.
    * `PierObject.build()` constructs a vertical axis internally and delegates
        geometry to `LinearObject.build()`; you select only high‑level options
        (twist, plan rotation, number of vertical slices).

Usage examples (PowerShell):
    python run_pier2.py                          # default, auto sequencing
    python run_pier2.py --vertical-slices 12     # denser vertical sampling
    python run_pier2.py --twist 15               # add twist around vertical axis
    python run_pier2.py --plan-rotation 5        # rotate plan orientation
    python run_pier2.py --use-linear             # force linear path build flag
    python run_pier2.py --aspect-equal           # equal aspect camera

After the build we print the resolved `ncs_steps` (if any) so you can verify
the offsets came directly from the JSON input.
"""
from __future__ import annotations
import json, argparse, os, numpy as np, plotly.graph_objects as go
from models.base import load_axis_from_rows, index_cross_sections_by_ncs, load_section_for_ncs, simple_load_piers
from models.main_station import load_mainstations_from_rows
from models.pier_object import PierObject
from models.plotter import Plotter, PlotConfig


def run(axis_json: str, cross_json: str, obj_json: str, main_json: str, section_json: str, out_html: str,
        twist_deg: float = 0.0, plan_rotation_deg: float = 0.0, aspect_equal: bool = False,
        debug_units: bool = False, debug_embed: bool = False, vertical_slices: int = 6,
        use_linear: bool = False) -> None:
    axis_rows = json.load(open(axis_json, 'r', encoding='utf-8'))
    cross_rows = json.load(open(cross_json, 'r', encoding='utf-8'))
    pier_rows  = json.load(open(obj_json,  'r', encoding='utf-8'))
    ms_rows    = json.load(open(main_json, 'r', encoding='utf-8'))

    axes = {r.get('Name'): load_axis_from_rows(axis_rows, r.get('Name'))
            for r in axis_rows if r.get('Class') == 'Axis' and r.get('Name')}
    by_ncs = index_cross_sections_by_ncs(cross_rows)
    cross_sections = {ncs: load_section_for_ncs(ncs, by_ncs, section_json) for ncs in by_ncs.keys()}
    mainstations = {name: load_mainstations_from_rows(ms_rows, axis_name=name) for name in axes.keys()}

    piers: list[PierObject] = simple_load_piers(pier_rows)
    if not piers:
        p = PierObject(); p.name = 'SyntheticPier'
        if axes:
            p.axis_name = next(iter(axes.keys()))
        if cross_sections:
            key = next(iter(cross_sections.keys()))
            p.top_cross_section_ncs = key
            p.base_section = cross_sections[key]
        piers = [p]

    pier = piers[0]
    pier.configure(axes, cross_sections, mainstations)
    if getattr(pier, 'base_section', None) is None and cross_sections:
        pier.base_section = next(iter(cross_sections.values()))
    # Vertical axis now constructed inside PierObject.build(); ensure placement axis exists for anchor sampling if provided.
    if getattr(pier, 'placement_axis_obj', None) is None and getattr(pier, 'axis_obj', None) is None:
        raise RuntimeError('Placement axis missing before build()')

    build = pier.build(vertical_slices=vertical_slices,
                       twist_deg=twist_deg,
                       plan_rotation_deg=plan_rotation_deg,
                       use_linear=use_linear,
                       debug_embed=debug_embed,
                       debug_units=debug_units)

    # Diagnostics: show sequencing & selected cross-section codes
    meta = build.get('meta', {}) if isinstance(build, dict) else {}
    ncs_steps = meta.get('ncs_steps')
    if ncs_steps:
        print('[pier] ncs_steps (station_m -> NCS):')
        for s, n in ncs_steps:
            print(f'  {s:8.3f} m -> {n}')
    else:
        print('[pier] ncs_steps: <none> (single-section or data missing)')
    if meta.get('selected_ncs') is not None:
        print(f"[pier] selected base NCS: {meta.get('selected_ncs')}")

    # Harmonize keys: linear path returns local_Y_mm/local_Z_mm, manual path same naming
    stations_mm = build['stations_mm']
    ids = build['ids']
    P_mm = build['points_world_mm']
    X_mm = build.get('local_Y_mm')  # naming kept for plotter compatibility
    Y_mm = build.get('local_Z_mm')

    cfg = PlotConfig(show_points=True, show_loops=True, show_loop_points=False, show_labels=False)
    plotter = Plotter(build.get('axis') or build.get('pier_axis'), obj_name=pier.name or pier.axis_name or 'Pier', config=cfg)
    traces = plotter.build_traces(
        json_data=None,
        stations_mm=stations_mm,
        ids=ids,
        P_mm=P_mm,
        X_mm=X_mm,
        Y_mm=Y_mm,
        loops_idx=build.get('loops_idx'),
        overlays=build.get('overlays'),
    )

    Pm = P_mm / 1000.0 if isinstance(P_mm, np.ndarray) else np.asarray(P_mm, float) / 1000.0
    if Pm.size:
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
    title_mode = 'LinearBuild' if use_linear else 'ManualBuild'
    fig.update_layout(title=f"Pier2 ({title_mode}) - {os.path.basename(axis_json)}", scene=scene)
    fig.write_html(out_html)
    print(f"Saved: {out_html}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Pier Runner v2 (manual vs linear build)')
    ap.add_argument('--axis', default='GIT/RCZ_new1/_Axis_JSON.json')
    ap.add_argument('--cross', default='GIT/RCZ_new1/_CrossSection_JSON.json')
    ap.add_argument('--obj', default='GIT/RCZ_new1/_PierObject_JSON.json')
    ap.add_argument('--main', default='GIT/RCZ_new1/_MainStation_JSON.json')
    ap.add_argument('--section', default='MASTER_SECTION/MASTER_Pier.json')
    ap.add_argument('--out', default='pier_plot2.html')
    ap.add_argument('--twist', type=float, default=0.0)
    ap.add_argument('--plan-rotation', type=float, default=0.0)
    ap.add_argument('--vertical-slices', type=int, default=6)
    ap.add_argument('--aspect-equal', action='store_true')
    ap.add_argument('--use-linear', action='store_true')
    ap.add_argument('--debug-units', action='store_true')
    ap.add_argument('--debug-embed', action='store_true')
    a = ap.parse_args()
    run(a.axis, a.cross, a.obj, a.main, a.section, a.out,
        twist_deg=a.twist, plan_rotation_deg=a.plan_rotation,
        aspect_equal=a.aspect_equal, debug_units=a.debug_units, debug_embed=a.debug_embed,
        vertical_slices=a.vertical_slices, use_linear=a.use_linear)
