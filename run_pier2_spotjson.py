"""Pier runner v2 using SpotJsonObject (test version)

Modified to use SpotLoader and SpotJsonObject for data loading instead of direct JSON loading.
"""
from __future__ import annotations
import json, argparse, os, numpy as np, plotly.graph_objects as go
from models.base import load_axis_from_rows, index_cross_sections_by_ncs, load_section_for_ncs, simple_load_piers
from models.main_station import load_mainstations_from_rows
from models.pier_object import PierObject
from models.plotter import Plotter, PlotConfig
from spot_loader import SpotLoader
from SPOT_Filters import SpotJsonObject


def run(master_folder: str, branch: str, section_json: str, out_html: str,
        twist_deg: float = 0.0, plan_rotation_deg: float = 0.0, aspect_equal: bool = False,
        debug_units: bool = False, debug_embed: bool = False, vertical_slices: int = 6,
        use_linear: bool = False, verbose: bool = False) -> None:
    # Load all data using SpotLoader and wrap in SpotJsonObject
    loader = SpotLoader(master_folder=master_folder, branch=branch)
    loader.load_raw().group_by_class()

    # Get SpotJsonObject lists for each class from grouped data
    axis_rows = [SpotJsonObject(row) for row in loader._by_class.get('Axis', [])]
    cross_rows = [SpotJsonObject(row) for row in loader._by_class.get('CrossSection', [])]
    pier_rows = [SpotJsonObject(row) for row in loader._by_class.get('PierObject', [])]
    ms_rows = [SpotJsonObject(row) for row in loader._by_class.get('MainStation', [])]

    # Build axes, cross_sections, mainstations using the SpotJsonObject lists
    axes = {r['Name']: load_axis_from_rows([obj.to_dict() for obj in axis_rows], r['Name'])
            for r in axis_rows if r['Class'] == 'Axis' and r['Name']}
    by_ncs = index_cross_sections_by_ncs([obj.to_dict() for obj in cross_rows])
    cross_sections = {ncs: load_section_for_ncs(ncs, by_ncs, section_json) for ncs in by_ncs.keys()}
    mainstations = {name: load_mainstations_from_rows([obj.to_dict() for obj in ms_rows], axis_name=name) for name in axes.keys()}

    piers: list[PierObject] = simple_load_piers([obj.to_dict() for obj in pier_rows])

    pier = piers[0]
    pier.configure(axes, cross_sections , mainstations)

    build = pier.build(vertical_slices=vertical_slices,
                       twist_deg=twist_deg,
                       plan_rotation_deg=plan_rotation_deg,
                       use_linear=use_linear,
                       debug_embed=debug_embed,
                       debug_units=debug_units)

    # Diagnostics: show sequencing & selected cross-section codes
    meta = build.get('meta', {}) if isinstance(build, dict) else {}
    if verbose:
        print(f"[pier] full meta: {meta}")
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
    fig.update_layout(title=f"Pier2 SpotJson ({title_mode}) - {master_folder}/{branch}", scene=scene)
    fig.write_html(out_html)
    print(f"Saved: {out_html}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Pier Runner v2 using SpotJsonObject with SpotLoader')
    ap.add_argument('--master-folder', default='GIT')
    ap.add_argument('--branch', default='MAIN')
    ap.add_argument('--section', default='MASTER_SECTION/MASTER_Pier.json')
    ap.add_argument('--out', default='pier_plot2_spotjson.html')
    ap.add_argument('--twist', type=float, default=0.0)
    ap.add_argument('--plan-rotation', type=float, default=0.0)
    ap.add_argument('--vertical-slices', type=int, default=6)
    ap.add_argument('--aspect-equal', action='store_true')
    ap.add_argument('--use-linear', action='store_true')
    ap.add_argument('--debug-units', action='store_true')
    ap.add_argument('--debug-embed', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()
    run(a.master_folder, a.branch, a.section, a.out,
        twist_deg=a.twist, plan_rotation_deg=a.plan_rotation,
        aspect_equal=a.aspect_equal, debug_units=a.debug_units, debug_embed=a.debug_embed,
        vertical_slices=a.vertical_slices, use_linear=a.use_linear, verbose=a.verbose)