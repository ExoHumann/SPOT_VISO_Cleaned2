

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
