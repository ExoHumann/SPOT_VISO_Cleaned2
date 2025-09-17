# run_linear.py
from __future__ import annotations
import os, json, argparse, plotly.graph_objects as go
from models.base import load_axis_from_rows, choose_section_json_path, load_section                 
from models.main_station import load_mainstations_from_rows, resolve_sections_for_object
from models.base import LinearObject                         
from main import get_plot_traces_matrix                     
import os

def main(axis_json, cross_json, deck_json, mainstation_json, section_json, out_html,
         *, max_stations=400, loop_stride=20, long_stride=50, twist_deg=0.0, flip_90=False):
    axis_rows = json.load(open(axis_json, "r", encoding="utf-8"))
    cross_rows = json.load(open(cross_json, "r", encoding="utf-8"))
    deck_rows  = json.load(open(deck_json,  "r", encoding="utf-8"))
    ms_rows    = json.load(open(mainstation_json, "r", encoding="utf-8"))

    # naive deck-row pick (same as before)
    deck_row = next(r for r in deck_rows if r.get("Class") == "DeckObject")

    axis = load_axis_from_rows(axis_rows, axis_name=str(deck_row.get("Axis@Name") or "RA"))
    section_path = choose_section_json_path(deck_row, cross_rows, fallback_path=section_json)
    base_section = load_section(section_path, name=(deck_row.get("Name") or "Section"))

    # MainStations for this axis
    ms_list = load_mainstations_from_rows(ms_rows, axis_name=str(deck_row.get("Axis@Name") or "RA"))

    # collect all CS candidates that may appear (deck's + those in mainstations)
    cs_names = []
    cs_names.extend(deck_row.get("CrossSection@Ncs") or [])
    cs_names.extend([ms.cs_name for ms in ms_list if ms.cs_name])
    sections_by_name = resolve_sections_for_object(cross_rows, cs_names, fallback_section_path=section_path)

    # Build minimal orchestrator
    obj = LinearObject(
        name=str(deck_row.get("Name") or "LinearObject"),
        axis=axis,
        base_section=base_section,
        axis_variables_rows=(deck_row.get("AxisVariables") or []) if isinstance(deck_row.get("AxisVariables"), list) else [],
        mainstations=ms_list,
        sections_by_name=sections_by_name,
    )

    res = obj.build(
        stations_m=None,                 # full axis by default
        twist_deg=float(twist_deg),
        station_cap=max_stations,
        rotation_override_deg=(90.0 if flip_90 else None),
    )

    traces, *_ = get_plot_traces_matrix(
        res["axis"], res["section_json"], res["stations_mm"],
        res["ids"], res["points_world_mm"],
        X_mm=res["local_Y_mm"], Y_mm=res["local_Z_mm"],
        loops_idx=res["loops_idx"],
        station_stride_for_loops=loop_stride,
        longitudinal_stride=long_stride,
    )

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"Linear Object — {res['name']}",
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)', aspectmode='data'),
        template='plotly_white', margin=dict(l=0, r=0, t=40, b=0)
    )
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"Saved: {out_html}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True)
    ap.add_argument("--cross", required=True)
    ap.add_argument("--deck", required=True)
    ap.add_argument("--main", required=True)         # NEW
    ap.add_argument("--section", required=True)
    ap.add_argument("--out", default="deck_plot.html")
    ap.add_argument("--max-stations", type=int, default=400)
    ap.add_argument("--loop-stride", type=int, default=20)
    ap.add_argument("--long-stride", type=int, default=50)
    ap.add_argument("--twist", type=float, default=0.0, help="Extra in-plane rotation (deg)")
    ap.add_argument("--flip90", action="store_true", help="Apply +90° flip in local YZ")


    ROOT = os.getcwd()  # Current working directory
    GIT = os.path.join(ROOT, "GIT", "MAIN")
    MS = os.path.join(ROOT, "MASTER_SECTION")

    args = {
        "axis": GIT+"\\_Axis_JSON.json",
        "cross": GIT+"\\_CrossSection_JSON.json",
        "deck": GIT+"\\_DeckObject_JSON.json",
        "main": GIT+"\\_MainStation_JSON.json",
        "section": MS+"\\SectionData.json",
        "out": ROOT+"\\deck_plot.html",
        "max_stations": 1000,
        "loop_stride": 1,
        "long_stride": 1,
        "flip_90": True,
        "twist": 0.0
    }

    main(args["axis"], args["cross"], args["deck"], args["main"], args["section"], args["out"],
         max_stations=args["max_stations"], loop_stride=args["loop_stride"], long_stride=args["long_stride"],
         twist_deg=args["twist"], flip_90=args["flip_90"])


# # Example usage:
# cd C:\Git\SPOT_VISO_krzys\SPOT_VISO

# $ROOT = "$PWD"
# $GIT  = "$ROOT\GIT\MAIN"
# $MS   = "$ROOT\MASTER_SECTION"

# Deck
# python .\run_linear.py `
#   --axis    "$GIT\_Axis_JSON.json" `
#   --cross   "$GIT\_CrossSection_JSON.json" `
#   --deck    "$GIT\_DeckObject_JSON.json" `
#   --section "$MS\SectionData.json" `
#   --out     "$ROOT\deck_plot.html" `
#   --max-stations 1000 `
#   --loop-stride 1 `
#   --long-stride 1 `
#   --flip90

# Pier
# python .\run_linear.py `
#   --axis    "$GIT\_Axis_JSON.json" `
#   --cross   "$GIT\_CrossSection_JSON.json" `
#   --deck    "$GIT\_PierObject_JSON.json" `
#   --section "$ROOT\MASTER_SECTION\MASTER_Pier.json" `
#   --out     "$ROOT\pier_plot.html" `
#   --max-stations 600 `
#   --loop-stride 2 `
#   --long-stride 2

# Foundation
# python .\run_linear.py `
#   --axis    "$GIT\_Axis_JSON.json" `
#   --cross   "$GIT\_CrossSection_JSON.json" `
#   --deck    "$GIT\_PierObject_JSON.json" `
#   --section "$ROOT\MASTER_SECTION\MASTER_Pier.json" `
#   --out     "$ROOT\pier_plot.html" `
#   --max-stations 600 `
#   --loop-stride 2 `
#   --long-stride 2
