# models/base.py  (append)

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os, json, numpy as np
from models.cross_section import CrossSection

@dataclass
class MainStationRef:
    station_m: float
    rotation_deg: float = 0.0           # plane rotation at/after this station
    cs_name: Optional[str] = None       # optional cross-section name switch
    placement_id: Optional[str] = None  # optional identifier from the row
    desc: Optional[str] = None

def _to_list(v):
    if v is None: return []
    if isinstance(v, (list, tuple)): return list(v)
    return [v]

def load_mainstations_from_rows(rows: List[dict], *, axis_name: str) -> List[MainStationRef]:
    out: List[MainStationRef] = []
    for r in rows or []:
        if str(r.get("Class")) != "MainStation": 
            continue
        ax = str(r.get("Axis@Name") or r.get("AxisName") or "").strip()
        if ax and ax != axis_name:
            continue

        # tolerant extraction
        stations = _to_list(r.get("StationValue") or r.get("StaionValue") or r.get("Station") or [])
        rotations = _to_list(r.get("AxisRotation") or r.get("Rotation") or r.get("AxisRotationDeg") or 0.0)
        cs_names  = _to_list(r.get("CrossSection@Name") or r.get("CrossSectionName") or [])
        place_ids = _to_list(r.get("PlacementId") or r.get("PlacementID") or [])
        descs     = _to_list(r.get("PlacementDescription") or r.get("Description") or [])

        n = max(len(stations), len(rotations), len(cs_names), len(place_ids), len(descs))
        def _pick(lst, i, default=None):
            if i < len(lst): return lst[i]
            return default

        for i in range(n):
            try:
                s_m = float(_pick(stations, i, 0.0))
            except Exception:
                continue
            rot = _pick(rotations, i, 0.0)
            try:
                rot = float(rot)
            except Exception:
                rot = 0.0

            out.append(MainStationRef(
                station_m=s_m,
                rotation_deg=rot,
                cs_name=(str(_pick(cs_names,  i, "")).strip() or None),
                placement_id=str(_pick(place_ids, i, "")).strip() or None,
                desc=str(_pick(descs, i, "")).strip() or None,
            ))
    out.sort(key=lambda ms: (ms.station_m, ms.placement_id or "", ms.cs_name or ""))
    return out

def resolve_sections_for_object(
    cross_rows: List[dict],
    requested_cs_names: List[str],
    fallback_section_path: str,
) -> Dict[str, CrossSection]:
    """
    Build CrossSection objects for all names we might switch to. If a CS row
    has no JSON_name, fall back to 'fallback_section_path' (usually MASTER_SECTION\SectionData.json).
    """
    # index cross rows by Name
    by_name = {str(r.get("Name")): r for r in cross_rows if str(r.get("Class")) == "CrossSection"}
    sections: Dict[str, CrossSection] = {}

    all_names = [n for n in requested_cs_names if n]  # keep order
    # ensure uniqueness
    seen = set()
    uniq_names = []
    for n in all_names:
        if n not in seen:
            seen.add(n); uniq_names.append(n)

    for name in uniq_names or ["__BASE__"]:
        row = by_name.get(name)
        if row:
            jnames = row.get("JSON_name") or []
            json_path = None
            if isinstance(jnames, list) and jnames:
                cand = jnames[0].replace("\\", "/")
                json_path = cand
            else:
                json_path = fallback_section_path  # fallback

        else:
            # unknown name -> fallback
            json_path = fallback_section_path

        # make path absolute if needed (we only change directory piece)
        if not os.path.isabs(json_path):
            base_dir = os.path.dirname(fallback_section_path) or "."
            json_path = os.path.join(base_dir, os.path.basename(json_path))

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        sections[name] = CrossSection(data=data, name=name)
    return sections
