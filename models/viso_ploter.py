# import numpy as np
# import plotly.graph_objects as go
# from typing import List, Sequence

# def plot_local_points(
#     ids: Sequence[str],
#     X_mm: np.ndarray,
#     Y_mm: np.ndarray,
#     loops_idx: List[np.ndarray],
#     *,
#     station: int = 0,
#     close_loops: bool = True,
#     mm_to_m: bool = False,
#     loop_color: str = "blue",
#     loop_point_color: str = "green",
#     other_point_color: str = "red",
#     point_size: int = 8,
#     loop_line_width: int = 2,
# ):
#     """
#     Plot 2D Plotly figure from compute_local_points output.

#     Parameters
#     ----------
#     ids : sequence of str
#         List of point IDs of length N.
#     X_mm, Y_mm : np.ndarray
#         Arrays of shape (S, N) with coordinates in millimeters.
#     loops_idx : list of np.ndarray
#         List of index arrays (referring to columns N) defining loops.
#     station : int
#         Index of the station (0..S-1) to plot.
#     close_loops : bool
#         If True, loops are closed by repeating the first point at the end.
#     mm_to_m : bool
#         If True, convert mm to meters.
#     loop_color : str
#         Line color for loops.
#     loop_point_color : str
#         Marker color for points belonging to loops.
#     other_point_color : str
#         Marker color for points outside loops.
#     point_size : int
#         Marker size for points.
#     loop_line_width : int
#         Line width for loops.

#     Returns
#     -------
#     fig : plotly.graph_objects.Figure
#         The resulting 2D Plotly figure.
#     """
#     ids = list(ids)
#     X = np.asarray(X_mm)
#     Y = np.asarray(Y_mm)
#     S, N = X.shape
#     if not (0 <= station < S):
#         raise IndexError(f"station out of range: 0..{S-1}")

#     scale = 0.001 if mm_to_m else 1.0
#     unit = "m" if mm_to_m else "mm"

#     sX = X[station, :] * scale
#     sY = Y[station, :] * scale

#     loop_index_set = set()
#     xs_all, ys_all = [], []

#     for idxs in (loops_idx or []):
#         idxs = np.asarray(idxs, dtype=int)
#         idxs = idxs[(idxs >= 0) & (idxs < N)]
#         if idxs.size == 0:
#             continue

#         xs = sX[idxs]
#         ys = sY[idxs]
#         valid = ~np.isnan(xs) & ~np.isnan(ys)
#         xs, ys, idxs = xs[valid], ys[valid], idxs[valid]
#         if xs.size < 2:
#             continue

#         if close_loops and not (np.isclose(xs[0], xs[-1]) and np.isclose(ys[0], ys[-1])):
#             xs = np.concatenate([xs, xs[:1]])
#             ys = np.concatenate([ys, ys[:1]])

#         xs_all.extend(xs.tolist() + [None])
#         ys_all.extend(ys.tolist() + [None])

#         loop_index_set.update(idxs.tolist())

#     # Separate loop points and other points
#     loop_pts_idx = sorted(loop_index_set)
#     other_pts_idx = [i for i in range(N) if i not in loop_index_set]

#     traces = []

#     # 1. Loops as lines
#     if xs_all:
#         traces.append(go.Scatter(
#             x=xs_all, y=ys_all,
#             mode="lines",
#             name="Loops",
#             line=dict(color=loop_color, width=loop_line_width),
#             hoverinfo="skip"
#         ))

#     # 2. Points belonging to loops
#     if loop_pts_idx:
#         lx = sX[loop_pts_idx]
#         ly = sY[loop_pts_idx]
#         llabels = [ids[i] for i in loop_pts_idx]
#         traces.append(go.Scatter(
#             x=lx, y=ly,
#             mode="markers+text",
#             text=llabels,
#             textposition="top center",
#             name="Loop points",
#             marker=dict(color=loop_point_color, size=point_size, symbol="circle"),
#             hovertemplate="<b>ID: %{text}</b><br>X: %{x:.3f} "+unit+"<br>Y: %{y:.3f} "+unit+"<br>Station: "+str(station)+"<extra></extra>"
#         ))

#     # 3. Points outside loops
#     if other_pts_idx:
#         px = sX[other_pts_idx]
#         py = sY[other_pts_idx]
#         plabels = [ids[i] for i in other_pts_idx]
#         traces.append(go.Scatter(
#             x=px, y=py,
#             mode="markers+text",
#             text=plabels,
#             textposition="top center",
#             name="Other points",
#             marker=dict(color=other_point_color, size=point_size, symbol="diamond"),
#             hovertemplate="<b>ID: %{text}</b><br>X: %{x:.3f} "+unit+"<br>Y: %{y:.3f} "+unit+"<br>Station: "+str(station)+"<extra></extra>"
#         ))

#     fig = go.Figure(traces)
#     # fig.update_layout(
#     #     title=f"Station {station}",
#     #     xaxis=dict(title=f"X [{unit}]"),
#     #     yaxis=dict(title=f"Y [{unit}]", scaleanchor="x", scaleratio=1,autorange= "reversed" ),
#     #     template="plotly_white",
#     #     legend=dict(orientation="h")
#     # )
#     # return fig


#     fig.update_layout(
#         title=f"Station {station}",
#         xaxis=dict(title=f"X [{unit}]"),
#         yaxis=dict(
#             title=f"Y [{unit}]",
#             scaleanchor="x",
#             scaleratio=1,
#             range=[y_max, y_min]   # <- reversed
#         ),
#         template="plotly_white",
#         legend=dict(orientation="h")
#     )
#     return fig
# import numpy as np
# import json
# import webbrowser
# import os

# def plot_local_points_with_sidebar(ids, X_mm, Y_mm, loops_idx, station=0, mm_to_m=True, filename="local_points.html"):
#     """
#     Create interactive HTML plot of local points with loops and sidebar listing all points.
#     """
#     # pick station
#     X = X_mm[station].astype(float)
#     Y = Y_mm[station].astype(float)

#     # unit conversion
#     if mm_to_m:
#         X = X / 1000.0
#         Y = Y / 1000.0

#     # build traces
#     point_trace = {
#         "x": X.tolist(),
#         "y": Y.tolist(),
#         "mode": "markers+text",
#         "type": "scatter",
#         "name": "Points",
#         "text": ids,
#         "textposition": "top center",
#         "hovertemplate": "ID: %{text}<br>X: %{x:.2f} m<br>Y: %{y:.2f} m<extra></extra>",
#         "marker": {"size": 8, "color": "red"},
#     }

#     loop_traces = []
#     for i, idx in enumerate(loops_idx):
#         loopX = [X[j] for j in idx] + [X[idx[0]]]
#         loopY = [Y[j] for j in idx] + [Y[idx[0]]]
#         loop_traces.append({
#             "x": loopX,
#             "y": loopY,
#             "mode": "lines",
#             "type": "scatter",
#             "name": f"Loop {i+1}",
#             "hoverinfo": "skip",
#             "line": {"color": "blue"},
#         })

#     layout = {
#         "title": "Local Points Plot",
#         "xaxis": {"title": "X [m]", "scaleanchor": "y"},
#         "yaxis": {"title": "Y [m]"},
#         "legend": {"orientation": "h"}
#     }

#     # --- serialize to JSON ---
#     point_trace_json = json.dumps(point_trace)
#     loop_traces_json = json.dumps(loop_traces)
#     layout_json = json.dumps(layout)
#     ids_json = json.dumps(ids)
#     X_json = json.dumps(X.tolist())
#     Y_json = json.dumps(Y.tolist())

#     # --- HTML wrapper with sidebar ---
#     html_template = f"""
# <!DOCTYPE html>
# <html lang="en">
# <head>
#   <meta charset="UTF-8">
#   <title>Local Points Plot</title>
#   <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
#   <style>
#     body {{
#       display: flex;
#       font-family: Arial, sans-serif;
#       margin: 0;
#       height: 100vh;
#     }}
#     #plot {{
#       flex: 3;
#     }}
#     #sidebar {{
#       flex: 1;
#       border-left: 1px solid #ccc;
#       padding: 10px;
#       overflow-y: auto;
#     }}
#     h2 {{
#       font-size: 18px;
#       margin-top: 0;
#     }}
#     ul {{
#       list-style-type: none;
#       padding: 0;
#     }}
#     li {{
#       padding: 4px 0;
#     }}
#   </style>
# </head>
# <body>

# <div id="plot"></div>
# <div id="sidebar">
#   <h2>Points</h2>
#   <ul id="points-list"></ul>
# </div>

# <script>
#   const pointTrace = {point_trace_json};
#   const loopTraces = {loop_traces_json};
#   const layout = {layout_json};

#   Plotly.newPlot('plot', [pointTrace, ...loopTraces], layout);

#   // sidebar
#   const ids = {ids_json};
#   const X = {X_json};
#   const Y = {Y_json};
#   const list = document.getElementById('points-list');
#   ids.forEach((id, i) => {{
#     const li = document.createElement('li');
#     li.textContent = `${{id}}: X=${{X[i].toFixed(2)}} m, Y=${{Y[i].toFixed(2)}} m`;
#     list.appendChild(li);
#   }});
# </script>

# </body>
# </html>
#     """

#     # write to file
#     with open(filename, "w", encoding="utf-8") as f:
#         f.write(html_template)

#     # open in browser
#     webbrowser.open("file://" + os.path.abspath(filename))

# # import numpy as np

# # # --- sztuczne dane jakby z compute_local_points ---
# # ids = ["P1","P2","P3","P4","P5","P6"]
# # S = 1   # jedna stacja
# # N = len(ids)

# # # współrzędne w mm (kształt (S,N))
# # X_mm = np.array([[0, 1000, 1000, 0, 2000, 2500]])
# # Y_mm = np.array([[0, 0, 1000, 1000,  500, 1500]])

# # # loopy: jeden kwadrat z pierwszych 4 punktów
# # loops_idx = [np.array([0,1,2,3])]

# # # importujemy wcześniej zdefiniowaną funkcję
# # fig = plot_local_points(ids, X_mm, Y_mm, loops_idx, station=0, mm_to_m=True)
# # fig.show()
# ids = ["P1","P2","P3","P4","P5","P6"]
# X_mm = np.array([[0, 1000, 1000, 0, 2000, 2500]])
# Y_mm = np.array([[0, 0, 1000, 1000, 500, 1500]])
# loops_idx = [np.array([0,1,2,3]), np.array([2,3,4])]

# plot_local_points_with_sidebar(ids, X_mm, Y_mm, loops_idx)



# import numpy as np
# import json
# import webbrowser
# import os

# def plot_local_points_with_sidebar(ids, X_mm, Y_mm, loops_idx, station=0, mm_to_m=True, filename="local_points.html"):
#     """
#     Interactive 2D Plotly plot of points and loops with sidebar listing points.
#     Y-axis is flipped and aspect ratio is 1:1.
#     """
#     X = np.array(X_mm[station], dtype=float)
#     Y = np.array(Y_mm[station], dtype=float)

#     if mm_to_m:
#         X /= 1000.0
#         Y /= 1000.0

#     # --- loop traces ---
#     loop_traces = []
#     for i, idx in enumerate(loops_idx):
#         idx = np.array(idx, dtype=int)
#         loopX = X[idx].tolist() + [X[idx[0]]]
#         loopY = Y[idx].tolist() + [Y[idx[0]]]
#         loop_traces.append({
#             "x": loopX,
#             "y": loopY,
#             "mode": "lines",
#             "type": "scatter",
#             "name": f"Loop {i+1}",
#             "hoverinfo": "skip",
#             "line": {"color": "blue", "width": 2}
#         })

#     # --- points trace ---
#     point_trace = {
#         "x": X.tolist(),
#         "y": Y.tolist(),
#         "mode": "markers+text",
#         "type": "scatter",
#         "name": "Points",
#         "text": ids,
#         "textposition": "top center",
#         "marker": {"size": 8, "color": "red"},
#         "hovertemplate": "ID: %{text}<br>X: %{x:.2f} m<br>Y: %{y:.2f} m<extra></extra>"
#     }

#     layout = {
#         "title": "Local Points Plot",
#         "xaxis": {
#             "title": "Y [m]",          # rename X → Y
#             "showline": True,
#             "linecolor": "#cccccc",    # light gray
#             "linewidth": 1,
#             "mirror": True,
#             "showgrid": True,
#             "gridcolor": "#eeeeee",
#             "ticks": "outside",
#             "showticklabels": True,
#             "zeroline": False,
#             "line_dash": "dash"        # dashed line
#         },
#         "yaxis": {
#             "title": "Z [m]",          # rename Y → Z
#             "scaleanchor": "x",
#             "scaleratio": 1,
#             "autorange": "reversed",
#             "showline": True,
#             "linecolor": "#cccccc",
#             "linewidth": 1,
#             "mirror": True,
#             "showgrid": True,
#             "gridcolor": "#eeeeee",
#             "ticks": "outside",
#             "showticklabels": True,
#             "zeroline": False,
#             "line_dash": "dash"
#         },
#         "legend": {"orientation": "h"},
#         "template": "plotly_white"
#     }


#     # --- JSON ---
#     html_template = f"""
# <!DOCTYPE html>
# <html lang="en">
# <head>
# <meta charset="UTF-8">
# <title>Local Points Plot</title>
# <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
# <style>
# body {{ display:flex; font-family:Arial,sans-serif; margin:0; height:100vh; }}
# #plot {{ flex:3; }}
# #sidebar {{ flex:1; border-left:1px solid #ccc; padding:10px; overflow-y:auto; }}
# h2 {{ font-size:18px; margin-top:0; }}
# ul {{ list-style:none; padding:0; }}
# li {{ padding:4px 0; }}
# </style>
# </head>
# <body>
# <div id="plot"></div>
# <div id="sidebar">
# <h2>Points</h2>
# <ul id="points-list"></ul>
# </div>
# <script>
# const data = [{json.dumps(point_trace)}, ...{json.dumps(loop_traces)}];
# const layout = {json.dumps(layout)};
# Plotly.newPlot('plot', data, layout);

# // sidebar list
# const ids = {json.dumps(ids)};
# const X = {json.dumps(X.tolist())};
# const Y = {json.dumps(Y.tolist())};
# const list = document.getElementById('points-list');
# ids.forEach((id,i) => {{
#     const li = document.createElement('li');
#     li.textContent = `${{id}}: X=${{X[i].toFixed(2)}} m, Y=${{Y[i].toFixed(2)}} m`;
#     list.appendChild(li);
# }});
# </script>
# </body>
# </html>
# """

#     # save and open
#     with open(filename, "w", encoding="utf-8") as f:
#         f.write(html_template)
#     webbrowser.open("file://" + os.path.abspath(filename))


# import numpy as np
# import json
# import webbrowser
# import os

# def plot_local_points_with_sidebar_table(ids, X_mm, Y_mm, loops_idx, station=0, mm_to_m=True, filename="local_points.html"):
#     """
#     Interactive 2D Plotly plot with loops and points, sidebar table of points (sorted).
#     Axes as light gray dashed lines, X→Y, Y→Z.
#     """
#     X = np.array(X_mm[station], dtype=float)
#     Y = np.array(Y_mm[station], dtype=float)

#     if mm_to_m:
#         X /= 1000.0
#         Y /= 1000.0

#     # loop traces
#     loop_traces = []
#     for i, idx in enumerate(loops_idx):
#         idx = np.array(idx, dtype=int)
#         loopX = X[idx].tolist() + [X[idx[0]]]
#         loopY = Y[idx].tolist() + [Y[idx[0]]]
#         loop_traces.append({
#             "x": loopX,
#             "y": loopY,
#             "mode": "lines",
#             "type": "scatter",
#             "name": f"Loop {i+1}",
#             "hoverinfo": "skip",
#             "line": {"color": "blue", "width": 2}
#         })

#     # points trace
#     point_trace = {
#         "x": X.tolist(),
#         "y": Y.tolist(),
#         "mode": "markers+text",
#         "type": "scatter",
#         "name": "Points",
#         "text": ids,
#         "textposition": "top center",
#         "marker": {"size": 8, "color": "red"},
#         "hovertemplate": "ID: %{text}<br>Y: %{x:.2f} m<br>Z: %{y:.2f} m<extra></extra>"
#     }

#     layout = {
#         "title": "Local Points Plot",
#         "xaxis": {
#             "title": {"text": "Y [m]", "font": {"color": "#888888"}},
#             "showline": False,   # ukryć normalną linię osi
#             "showgrid": False
#         },
#         "yaxis": {
#             "title": {"text": "Z [m]", "font": {"color": "#888888"}},
#             "scaleanchor": "x",
#             "scaleratio": 1,
#             "autorange": "reversed",
#             "showline": False,
#             "showgrid": False
#         },
#         "shapes": [
#             # X axis dashed line
#             {"type":"line","x0":min(X),"x1":max(X),"y0":min(Y),"y1":min(Y),
#             "line":{"color":"#cccccc","width":1,"dash":"dash"}},
#             # Y axis dashed line
#             {"type":"line","x0":min(X),"x1":min(X),"y0":min(Y),"y1":max(Y),
#             "line":{"color":"#cccccc","width":1,"dash":"dash"}}
#         ],
#         "legend":{"orientation":"h"},
#         "template":"plotly_white"
#     }


#     # prepare sorted table
#     sorted_points = sorted(zip(ids, X, Y), key=lambda t: t[0])  # sort by ID
#     table_html = "<table border='1' style='border-collapse:collapse;width:100%;'><tr><th>ID</th><th>Y [m]</th><th>Z [m]</th></tr>"
#     for pid, x, y in sorted_points:
#         table_html += f"<tr><td>{pid}</td><td>{x:.2f}</td><td>{y:.2f}</td></tr>"
#     table_html += "</table>"

#     # HTML
#     html_template = f"""
# <!DOCTYPE html>
# <html lang="en">
# <head>
# <meta charset="UTF-8">
# <title>Local Points Plot</title>
# <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
# <style>
# body {{ display:flex; font-family:Arial,sans-serif; margin:0; height:100vh; }}
# #plot {{ flex:3; }}
# #sidebar {{ flex:1; border-left:1px solid #ccc; padding:10px; overflow-y:auto; }}
# h2 {{ font-size:18px; margin-top:0; }}
# table {{ width:100%; border-collapse:collapse; }}
# th, td {{ padding:4px; text-align:center; border:1px solid #ccc; }}
# th {{ background-color:#f2f2f2; }}
# </style>
# </head>
# <body>
# <div id="plot"></div>
# <div id="sidebar">
# <h2>Points Table</h2>
# {table_html}
# </div>
# <script>
# const data = [{json.dumps(point_trace)}, ...{json.dumps(loop_traces)}];
# const layout = {json.dumps(layout)};
# Plotly.newPlot('plot', data, layout);
# </script>
# </body>
# </html>
# """

#     with open(filename, "w", encoding="utf-8") as f:
#         f.write(html_template)
#     webbrowser.open("file://" + os.path.abspath(filename))


# import numpy as np
# import json
# import webbrowser
# import os

# def plot_local_points_with_axes_table(ids, X_mm, Y_mm, loops_idx, station=0, mm_to_m=True, filename="local_points.html"):
#     """
#     Interactive 2D Plotly plot with loops, points, axes as dashed lines, and sidebar table.
#     Axes as separate traces that can be toggled.
#     """
#     X = np.array(X_mm[station], dtype=float)
#     Y = np.array(Y_mm[station], dtype=float)

#     if mm_to_m:
#         X /= 1000.0
#         Y /= 1000.0

#     # loop traces
#     loop_traces = []
#     for i, idx in enumerate(loops_idx):
#         idx = np.array(idx, dtype=int)
#         loopX = X[idx].tolist() + [X[idx[0]]]
#         loopY = Y[idx].tolist() + [Y[idx[0]]]
#         loop_traces.append({
#             "x": loopX,
#             "y": loopY,
#             "mode": "lines",
#             "type": "scatter",
#             "name": f"Loop {i+1}",
#             "hoverinfo": "skip",
#             "line": {"color": "blue", "width": 2}
#         })

#     # points trace
#     point_trace = {
#         "x": X.tolist(),
#         "y": Y.tolist(),
#         "mode": "markers+text",
#         "type": "scatter",
#         "name": "Points",
#         "text": ids,
#         "textposition": "top center",
#         "marker": {"size": 8, "color": "red"},
#         "hovertemplate": "ID: %{text}<br>Y: %{x:.2f} m<br>Z: %{y:.2f} m<extra></extra>"
#     }

#     # axis traces as separate dashed lines
#     axis_traces = []
#     # X axis (Y-axis in name)
#     axis_traces.append({
#         "x": [min(X), max(X)],
#         "y": [min(Y), min(Y)],
#         "mode": "lines",
#         "type": "scatter",
#         "name": "X Axis (Y)",
#         "line": {"color": "#cccccc", "width": 1, "dash": "dash"},
#         "hoverinfo": "skip",
#     })
#     # Y axis (Z-axis in name)
#     axis_traces.append({
#         "x": [min(X), min(X)],
#         "y": [min(Y), max(Y)],
#         "mode": "lines",
#         "type": "scatter",
#         "name": "Y Axis (Z)",
#         "line": {"color": "#cccccc", "width": 1, "dash": "dash"},
#         "hoverinfo": "skip",
#     })

#     # layout
#     layout = {
#         "title": "Local Points Plot",
#         "xaxis": {"title": "Y [m]"},
#         "yaxis": {"title": "Z [m]", "scaleanchor": "x", "scaleratio": 1, "autorange": "reversed"},
#         "legend": {"orientation": "h"},
#         "template": "plotly_white"
#     }

#     # prepare sorted table
#     sorted_points = sorted(zip(ids, X, Y), key=lambda t: t[0])
#     table_html = "<table border='1' style='border-collapse:collapse;width:100%;'><tr><th>ID</th><th>Y [m]</th><th>Z [m]</th></tr>"
#     for pid, x, y in sorted_points:
#         table_html += f"<tr><td>{pid}</td><td>{x:.2f}</td><td>{y:.2f}</td></tr>"
#     table_html += "</table>"

#     # HTML
#     html_template = f"""
# <!DOCTYPE html>
# <html lang="en">
# <head>
# <meta charset="UTF-8">
# <title>Local Points Plot</title>
# <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
# <style>
# body {{ display:flex; font-family:Arial,sans-serif; margin:0; height:100vh; }}
# #plot {{ flex:3; }}
# #sidebar {{ flex:1; border-left:1px solid #ccc; padding:10px; overflow-y:auto; }}
# h2 {{ font-size:18px; margin-top:0; }}
# table {{ width:100%; border-collapse:collapse; }}
# th, td {{ padding:4px; text-align:center; border:1px solid #ccc; }}
# th {{ background-color:#f2f2f2; }}
# </style>
# </head>
# <body>
# <div id="plot"></div>
# <div id="sidebar">
# <h2>Points Table</h2>
# {table_html}
# </div>
# <script>
# const data = [...{json.dumps(axis_traces)}, {json.dumps(point_trace)}, ...{json.dumps(loop_traces)}];
# const layout = {json.dumps(layout)};
# Plotly.newPlot('plot', data, layout);
# </script>
# </body>
# </html>
# """

#     with open(filename, "w", encoding="utf-8") as f:
#         f.write(html_template)
#     webbrowser.open("file://" + os.path.abspath(filename))


import numpy as np
import json
import webbrowser
import os

def plot_local_points_with_axes_table(ids, X_mm, Y_mm, loops_idx, station=0, mm_to_m=True, filename="local_points.html"):
    X = np.array(X_mm[station], dtype=float)
    Y = np.array(Y_mm[station], dtype=float)

    if mm_to_m:
        X /= 1000.0
        Y /= 1000.0

    # loop traces
    loop_traces = []
    for i, idx in enumerate(loops_idx):
        idx = np.array(idx, dtype=int)
        loopX = X[idx].tolist() + [X[idx[0]]]
        loopY = Y[idx].tolist() + [Y[idx[0]]]
        loop_traces.append({
            "x": loopX,
            "y": loopY,
            "mode": "lines",
            "type": "scatter",
            "name": f"Loop {i+1}",
            "hoverinfo": "skip",
            "line": {"color": "blue", "width": 2}
        })

    # points trace
    point_trace = {
        "x": X.tolist(),
        "y": Y.tolist(),
        "mode": "markers+text",
        "type": "scatter",
        "name": "Points",
        "text": ids,
        "textposition": "top center",
        "marker": {"size": 8, "color": "red"},
        "hovertemplate": "ID: %{text}<br>Y: %{x:.2f} m<br>Z: %{y:.2f} m<extra></extra>"
    }

    # axes passing through (0,0)
    x_min, x_max = min(X), max(X)
    y_min, y_max = min(Y), max(Y)

    # rozszerzenie zakresu o 20%
    x_range = x_max - x_min
    y_range = y_max - y_min

    x_min_plot = x_min - 0.2 * x_range
    x_max_plot = x_max + 0.2 * x_range
    y_min_plot = y_min - 1.0 * y_range
    y_max_plot = y_max + 1.0 * y_range

    axis_traces = [
        # horizontal axis through Y=0
        {"x": [x_min_plot, x_max_plot], "y": [0, 0], "mode": "lines", "type": "scatter",
        "name": "X axis", "line": {"color": "#888888", "width": 1, "dash": "dash"}, "hoverinfo": "skip"},
        # vertical axis through X=0
        {"x": [0, 0], "y": [y_min_plot, y_max_plot], "mode": "lines", "type": "scatter",
        "name": "Y axis", "line": {"color": "#888888", "width": 1, "dash": "dash"}, "hoverinfo": "skip"}
    ]

    # w layout dodajemy rozszerzone granice
    layout = {
        "title": "Local Points Plot",
        "xaxis": {"title": "Y [m]", "zeroline": False, "showline": False, "range": [x_min_plot, x_max_plot]},
        "yaxis": {"title": "Z [m]", "scaleanchor": "x", "scaleratio": 1, "autorange": False,
                "zeroline": False, "showline": False, "range": [y_max_plot, y_min_plot]},  # reversed
        "legend": {"orientation": "h"},
        "template": "plotly_white"
    }


    # sorted table
    sorted_points = sorted(zip(ids, X, Y), key=lambda t: t[0])
    table_html = "<table border='1' style='border-collapse:collapse;width:100%;'><tr><th>ID</th><th>Y [m]</th><th>Z [m]</th></tr>"
    for pid, x, y in sorted_points:
        table_html += f"<tr><td>{pid}</td><td>{x:.2f}</td><td>{y:.2f}</td></tr>"
    table_html += "</table>"

    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Local Points Plot</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
body {{ display:flex; font-family:Arial,sans-serif; margin:0; height:100vh; }}
#plot {{ flex:3; }}
#sidebar {{ flex:1; border-left:1px solid #ccc; padding:10px; overflow-y:auto; }}
h2 {{ font-size:18px; margin-top:0; }}
table {{ width:100%; border-collapse:collapse; }}
th, td {{ padding:4px; text-align:center; border:1px solid #ccc; }}
th {{ background-color:#f2f2f2; }}
</style>
</head>
<body>
<div id="plot"></div>
<div id="sidebar">
<h2>Points Table</h2>
{table_html}
</div>
<script>
const data = [...{json.dumps(axis_traces)}, {json.dumps(point_trace)}, ...{json.dumps(loop_traces)}];
const layout = {json.dumps(layout)};
Plotly.newPlot('plot', data, layout);
</script>
</body>
</html>
"""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_template)
    webbrowser.open("file://" + os.path.abspath(filename))
