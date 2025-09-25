from __future__ import annotations
"""Modular plotter utilities extracted from main.get_plot_traces_matrix.

This keeps plotting concerns isolated and easier to extend (e.g. frame overlays,
rotation debug, per-object metadata).
"""
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Dict, Iterable
import numpy as np
import plotly.graph_objects as go

@dataclass
class PlotConfig:
    show_axis: bool = True
    show_points: bool = True
    show_loops: bool = True
    show_longitudinal: bool = True
    show_loop_points: bool = False              # NEW: markers on loop vertices
    show_labels: bool = True
    station_stride_for_loops: int = 1
    longitudinal_stride: int = 1
    first_station: int = 0
    compact_meta: bool = True
    loops_only_from_overlays: bool = False
    include_first_station_ids_in_longitudinal: bool = False
    filter_longitudinal_to_loops: bool = True

    # colors (can be overridden)
    colors: Dict[str, str] = field(default_factory=lambda: {
        'axis': 'black',
        'first_station_points': 'blue',
        'cross_section_loops': 'red',
        'longitudinal_lines': 'gray',
        'loop_points': '#aa4444',
    })

class Plotter:
    def __init__(self, axis, obj_name: str = "Object", *, config: PlotConfig | None = None):
        self.axis = axis
        self.obj_name = obj_name or "Object"
        self.cfg = config or PlotConfig()

    # ---------------- helpers ----------------
    def _to_m(self, arr_mm):
        a = np.asarray(arr_mm, float)
        return a / 1000.0

    # ---------------- traces builders ----------------
    def build_axis_trace(self, traces: List, *, axis):
        if not self.cfg.show_axis:
            return
        x_m = (np.asarray(axis.x_coords, float) / 1000.0).tolist()
        y_m = (np.asarray(axis.y_coords, float) / 1000.0).tolist()
        z_m = (np.asarray(axis.z_coords, float) / 1000.0).tolist()
        stations_m = (np.asarray(axis.stations, float) / 1000.0).tolist()
        # customdata: [station_m, index, x, y, z]
        cdat = [[stations_m[i], i, x_m[i], y_m[i], z_m[i]] for i in range(len(stations_m))]
        traces.append(go.Scatter3d(
            x=x_m, y=y_m, z=z_m,
            mode='lines+markers',
            line=dict(color=self.cfg.colors['axis'], width=3),
            marker=dict(size=3, color=self.cfg.colors['axis'], opacity=0.9),
            name=f'{self.obj_name} Axis',
            meta=self.obj_name,
            customdata=cdat,
            hovertemplate=(
                '<b>%{meta} Axis</b><br>'
                'Station: %{customdata[0]:.3f} m<br>'
                'Index: %{customdata[1]}<br>'
                'X: %{customdata[2]:.3f} m<br>'
                'Y: %{customdata[3]:.3f} m<br>'
                'Z: %{customdata[4]:.3f} m<extra></extra>'
            ),
        ))

    def build_overlay_traces(self, traces: List, overlays: Optional[List[Dict]]):
        if not overlays:
            return
        PRIORITY = {'var': 3, 'ms': 2, 'switch': 1}
        bundles: Dict[float, Dict[str, Any]] = {}
        for ov in overlays:
            P_ov = self._to_m(ov.get("P_mm"))
            loops = ov.get("loops_idx") or []
            st = float(ov.get("station_m", float('nan')))
            kind = str(ov.get("kind") or "")
            col_pt = ov.get("color") or self.cfg.colors['first_station_points']
            col_loop = ov.get("color") or self.cfg.colors['cross_section_loops']
            s_key = round(st, 6)
            b = bundles.setdefault(s_key, {"st": st, "items": []})
            b["items"].append({
                "P": P_ov, "loops": loops, "kind": kind,
                "prio": PRIORITY.get(kind, 0),
                "col_pt": col_pt, "col_loop": col_loop,
            })
        for s_key in sorted(bundles.keys()):
            b = bundles[s_key]; st = b["st"]; items = b["items"]
            rep = max(items, key=lambda it: it['prio'])
            P_rep = rep['P']; loop_color = rep['col_loop']; col_pt = rep['col_pt']; rep_kind = rep['kind']
            lg = f"ov@{st:.3f}"
            # overlay point metadata: [station_m, kind, x, y, z]
            overlay_meta = [[st, rep_kind or '', float(p[0]), float(p[1]), float(p[2])] for p in P_rep]
            traces.append(go.Scatter3d(
                x=P_rep[:,0], y=P_rep[:,1], z=P_rep[:,2],
                mode='markers',
                marker=dict(size=4, opacity=0.95, color=col_pt),
                name=f"{self.obj_name} {rep_kind.capitalize()} @ {st:.3f} m" if rep_kind else f"{self.obj_name} @ {st:.3f} m",
                legendgroup=lg, showlegend=True,
                meta=self.obj_name,
                customdata=overlay_meta,
                hovertemplate=(
                    '<b>%{meta} Overlay</b><br>'
                    'Station: %{customdata[0]:.3f} m<br>'
                    'Kind: %{customdata[1]}<br>'
                    'X: %{customdata[2]:.3f} m<br>'
                    'Y: %{customdata[3]:.3f} m<br>'
                    'Z: %{customdata[4]:.3f} m<extra></extra>'
                ),
            ))
            bx, by, bz = [], [], []
            b_meta: List[Any] = []  # for loop line segments: reuse station and kind, include xyz
            for it in items:
                P_ov = it['P']
                for arr in (it['loops'] or []):
                    idxs = np.asarray(arr, int)
                    ok = (idxs >= 0) & (idxs < len(P_ov))
                    idxs = idxs[ok]
                    if idxs.size >= 2:
                        if idxs.size >= 3 and idxs[0] != idxs[-1]:
                            idxs = np.append(idxs, idxs[0])
                        xs = P_ov[idxs,0].tolist(); ys = P_ov[idxs,1].tolist(); zs = P_ov[idxs,2].tolist()
                        bx.extend(xs + [None])
                        by.extend(ys + [None])
                        bz.extend(zs + [None])
                        for (xx,yy,zz) in zip(xs,ys,zs):
                            b_meta.append([st, rep_kind or '', float(xx), float(yy), float(zz)])
                        b_meta.append([None]*5)
            if bx:
                traces.append(go.Scatter3d(
                    x=bx, y=by, z=bz,
                    mode='lines',
                    line=dict(color=loop_color, width=2),
                    name=f"{self.obj_name} CS @ {st:.3f} m",
                    legendgroup=lg, showlegend=False,
                    meta=self.obj_name,
                    customdata=b_meta,
                    hovertemplate=(
                        '<b>%{meta} Overlay CS</b><br>'
                        'Station: %{customdata[0]:.3f} m<br>'
                        'Kind: %{customdata[1]}<br>'
                        'X: %{customdata[2]:.3f} m<br>'
                        'Y: %{customdata[3]:.3f} m<br>'
                        'Z: %{customdata[4]:.3f} m<extra></extra>'
                    ),
                ))

    def build_first_station_points(self, traces: List, *, P_m, X_m, Y_m, ids, stations_m):
        if not self.cfg.show_points:
            return
        S = P_m.shape[0]
        fs = int(max(0, min(self.cfg.first_station, S-1)))
        P0 = P_m[fs]
        valid = ~np.isnan(P0).any(axis=1)
        if not valid.any():
            return
        st0 = float(stations_m[fs])
        x0 = P0[valid,0].tolist(); y0 = P0[valid,1].tolist(); z0 = P0[valid,2].tolist()
        ids0 = [ids[j] for j, ok in enumerate(valid) if ok]
        if self.cfg.compact_meta:
            ly0 = (X_m[fs, valid] if X_m is not None else np.full(len(ids0), np.nan))
            lz0 = (Y_m[fs, valid] if Y_m is not None else np.full(len(ids0), np.nan))
            cdat = np.column_stack([
                np.asarray(ids0, dtype=object),
                np.full(len(ids0), st0, float),
                np.asarray(ly0, float),
                np.asarray(lz0, float),
                np.asarray(x0, float),
                np.asarray(y0, float),
                np.asarray(z0, float),
            ]).tolist()
            traces.append(go.Scatter3d(
                x=x0, y=y0, z=z0,
                mode='markers+text' if self.cfg.show_labels else 'markers',
                marker=dict(size=4, opacity=0.9, color=self.cfg.colors['first_station_points']),
                text=ids0 if self.cfg.show_labels else None,
                textposition='top center',
                name=f'{self.obj_name} Points @ {st0:.3f} m',
                meta=self.obj_name,
                customdata=cdat,
                hovertemplate=(
                    "<b>%{meta}</b><br>" \
                    "Point: %{customdata[0]}<br>" \
                    "Station: %{customdata[1]:.3f} m<br>" \
                    "Local Y: %{customdata[2]:.3f} m<br>" \
                    "Local Z: %{customdata[3]:.3f} m<br>" \
                    "X: %{customdata[4]:.3f} m<br>" \
                    "Y: %{customdata[5]:.3f} m<br>" \
                    "Z: %{customdata[6]:.3f} m<extra></extra>"
                ),
            ))
        else:
            # non-compact not reimplemented here to keep class concise
            pass

    def build_loop_traces(self, traces: List, *, P_m, ids, stations_m, loops_idx, X_m, Y_m):
        if not (self.cfg.show_loops and not self.cfg.loops_only_from_overlays):
            return
        S, N, _ = P_m.shape
        if loops_idx is None:
            return
        all_loop_x: List[float] = []; all_loop_y: List[float] = []; all_loop_z: List[float] = []
        all_loop_meta: List[Any] = []  # [station, point_id, localY, localZ, x, y, z]
        loop_point_markers_x: List[float] = []; loop_point_markers_y: List[float] = []; loop_point_markers_z: List[float] = []
        loop_point_markers_meta: List[Any] = []
        stride = max(1, int(self.cfg.station_stride_for_loops))
        for s in range(0, S, stride):
            st_m = float(stations_m[s])
            for idxs in loops_idx:
                col = P_m[s, idxs, :]
                valid = ~np.isnan(col).any(axis=1)
                if not valid.any():
                    continue
                pts = col[valid]
                xs = pts[:,0].tolist(); ys = pts[:,1].tolist(); zs = pts[:,2].tolist()
                locY = (X_m[s, idxs][valid] if X_m is not None else np.full(len(xs), np.nan))
                locZ = (Y_m[s, idxs][valid] if Y_m is not None else np.full(len(xs), np.nan))
                pid_list = [ids[j] for j, ok in zip(idxs, valid) if ok]
                closed = False
                if len(xs) >= 3 and (xs[0] != xs[-1] or ys[0] != ys[-1] or zs[0] != zs[-1]):
                    # close the loop
                    xs.append(xs[0]); ys.append(ys[0]); zs.append(zs[0])
                    locY = np.append(locY, locY[0])
                    locZ = np.append(locZ, locZ[0])
                    pid_list.append(pid_list[0])
                    closed = True
                all_loop_x.extend(xs + [None])
                all_loop_y.extend(ys + [None])
                all_loop_z.extend(zs + [None])
                for k in range(len(xs)):
                    all_loop_meta.append([
                        st_m,
                        pid_list[k],
                        float(locY[k]),
                        float(locZ[k]),
                        float(xs[k]),
                        float(ys[k]),
                        float(zs[k]),
                    ])
                all_loop_meta.append([None]*7)  # separator for None gap
                if self.cfg.show_loop_points:
                    loop_point_markers_x.extend(pts[:,0].tolist())
                    loop_point_markers_y.extend(pts[:,1].tolist())
                    loop_point_markers_z.extend(pts[:,2].tolist())
                    for k in range(len(xs) - (1 if closed else 0)):
                        loop_point_markers_meta.append([
                            st_m,
                            pid_list[k],
                            float(locY[k]),
                            float(locZ[k]),
                            float(xs[k]),
                            float(ys[k]),
                            float(zs[k]),
                        ])
        if all_loop_x:
            traces.append(go.Scatter3d(
                x=all_loop_x, y=all_loop_y, z=all_loop_z,
                mode='lines', line=dict(color=self.cfg.colors['cross_section_loops'], width=2),
                name=f'{self.obj_name} Cross Sections',
                customdata=all_loop_meta,
                hovertemplate=(
                    '<b>'+self.obj_name+' Cross-Section</b><br>'
                    'Station: %{customdata[0]:.3f} m<br>'
                    'Point: %{customdata[1]}<br>'
                    'Local Y: %{customdata[2]:.3f} m<br>'
                    'Local Z: %{customdata[3]:.3f} m<br>'
                    'X: %{customdata[4]:.3f} m<br>'
                    'Y: %{customdata[5]:.3f} m<br>'
                    'Z: %{customdata[6]:.3f} m<extra></extra>'
                )
            ))
        if loop_point_markers_x:
            traces.append(go.Scatter3d(
                x=loop_point_markers_x, y=loop_point_markers_y, z=loop_point_markers_z,
                mode='markers', marker=dict(size=2, color=self.cfg.colors['loop_points']),
                name=f'{self.obj_name} Loop Points',
                customdata=loop_point_markers_meta,
                hovertemplate=(
                    '<b>'+self.obj_name+' Loop Point</b><br>'
                    'Station: %{customdata[0]:.3f} m<br>'
                    'Point: %{customdata[1]}<br>'
                    'Local Y: %{customdata[2]:.3f} m<br>'
                    'Local Z: %{customdata[3]:.3f} m<br>'
                    'X: %{customdata[4]:.3f} m<br>'
                    'Y: %{customdata[5]:.3f} m<br>'
                    'Z: %{customdata[6]:.3f} m<extra></extra>'
                )
            ))

    def build_longitudinal_traces(self, traces: List, *, P_m, ids, X_m, Y_m, stations_m, loops_idx):
        if not self.cfg.show_longitudinal:
            return
        S, N, _ = P_m.shape
        loop_ids = set()
        if loops_idx:
            for arr in loops_idx:
                for j in arr:
                    loop_ids.add(ids[j])
        if not self.cfg.filter_longitudinal_to_loops:
            loop_ids.clear()
        stride = max(1, int(self.cfg.longitudinal_stride))
        Sm = range(0, S, stride)
        long_x=[]; long_y=[]; long_z=[]; long_meta=[]  # meta: [station_m, point_id, x, y, z]
        for j, pid in enumerate(ids):
            if loop_ids and pid not in loop_ids:
                continue
            col = P_m[:, j, :]
            valid = ~np.isnan(col).any(axis=1)
            last_ok=False
            for k in Sm:
                if valid[k]:
                    p = col[k]
                    long_x.append(float(p[0])); long_y.append(float(p[1])); long_z.append(float(p[2]))
                    long_meta.append([
                        float(stations_m[k]),
                        pid,
                        float(p[0]), float(p[1]), float(p[2])
                    ])
                elif last_ok:
                    long_x.append(None); long_y.append(None); long_z.append(None)
                    long_meta.append([None]*5)
                last_ok = bool(valid[k])
            if last_ok:
                long_x.append(None); long_y.append(None); long_z.append(None)
                long_meta.append([None]*5)
        if long_x:
            traces.append(go.Scatter3d(
                x=long_x, y=long_y, z=long_z,
                mode='lines', line=dict(color=self.cfg.colors['longitudinal_lines'], width=1),
                name=f'{self.obj_name} Longitudinal',
                customdata=long_meta,
                hovertemplate=(
                    '<b>'+self.obj_name+' Longitudinal</b><br>'
                    'Station: %{customdata[0]:.3f} m<br>'
                    'Point: %{customdata[1]}<br>'
                    'X: %{customdata[2]:.3f} m<br>'
                    'Y: %{customdata[3]:.3f} m<br>'
                    'Z: %{customdata[4]:.3f} m<extra></extra>'
                )
            ))

    # ---------------- main entry ----------------
    def build_traces(self, *, json_data, stations_mm, ids, P_mm, X_mm=None, Y_mm=None,
                     loops_idx=None, overlays=None) -> List[Any]:
        traces: List[Any] = []
        P_mm = np.asarray(P_mm, float)
        stations_mm = np.asarray(stations_mm, float)
        stations_m = stations_mm / 1000.0
        P_m = P_mm / 1000.0
        X_m = None if X_mm is None else np.asarray(X_mm, float) / 1000.0
        Y_m = None if Y_mm is None else np.asarray(Y_mm, float) / 1000.0

        self.build_axis_trace(traces, axis=self.axis)
        self.build_overlay_traces(traces, overlays)
        self.build_first_station_points(traces, P_m=P_m, X_m=X_m, Y_m=Y_m, ids=ids, stations_m=stations_m)
        if loops_idx is not None:
            self.build_loop_traces(traces, P_m=P_m, ids=ids, stations_m=stations_m, loops_idx=loops_idx, X_m=X_m, Y_m=Y_m)
        self.build_longitudinal_traces(traces, P_m=P_m, ids=ids, X_m=X_m, Y_m=Y_m, stations_m=stations_m, loops_idx=loops_idx)
        return traces
