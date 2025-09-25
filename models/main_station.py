# models/main_station.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os, json, numpy as np
from models.cross_section import CrossSection

@dataclass
class MainStationRef:
    station_m: float
    # Backward-compatible generic rotation (legacy meaning: in-plane twist)
    rotation_deg: float = 0.0
    # New explicit per-axis rotations (additive semantics by default):
    station_rotation_x_deg: float = 0.0  # twist around tangent (in-plane rotation of section)
    station_rotation_z_deg: float = 0.0  # plan yaw about global Z
    cs_name: Optional[str] = None
    placement_id: Optional[str] = None
    desc: Optional[str] = None

# Alias for backward compatibility
MainStation = MainStationRef

def _to_list(v):
    if v is None: return []
    if isinstance(v, (list, tuple)): return list(v)
    return [v]

def load_mainstations_from_rows(rows: List[dict], *, axis_name: str) -> List[MainStationRef]:
    """Parse MainStation rows including distinct StationRotationX / StationRotationZ.

    Accepts either scalar values or arrays. Fallback to legacy rotation fields
    (AxisRotation / Rotation) when explicit X/Z rotations are absent.
    """
    out: List[MainStationRef] = []
    for r in rows or []:
        if str(r.get("Class")) != "MainStation":
            continue
        ax = str(r.get("Axis@Name") or r.get("AxisName") or "").strip()
        if ax and ax != axis_name:
            continue

        stations = _to_list(r.get("StationValue") or r.get("StaionValue") or r.get("Station") or [])
        rot_legacy = _to_list(r.get("AxisRotation") or r.get("Rotation") or r.get("AxisRotationDeg") or 0.0)
        rot_x = _to_list(r.get("StationRotationX") or [])
        rot_z = _to_list(r.get("StationRotationZ") or [])
        cs_names  = _to_list(r.get("CrossSection@Name") or r.get("CrossSectionName") or [])
        place_ids = _to_list(r.get("PlacementId") or r.get("PlacementID") or [])
        descs     = _to_list(r.get("PlacementDescription") or r.get("Description") or [])

        n = max(len(stations), len(rot_legacy), len(rot_x), len(rot_z), len(cs_names), len(place_ids), len(descs))

        def _pick(lst, i, default=None):
            return lst[i] if i < len(lst) else default

        for i in range(n):
            try:
                s_m = float(_pick(stations, i, 0.0))
            except Exception:
                continue
            # per-axis rotations
            val_x = _pick(rot_x, i, None)
            val_z = _pick(rot_z, i, None)
            def _to_float(v, default=0.0):
                if v in (None, "", 'null'): return default
                try:
                    return float(v)
                except Exception:
                    return default
            x_deg = _to_float(val_x)
            z_deg = _to_float(val_z)
            legacy = _to_float(_pick(rot_legacy, i, 0.0))

            out.append(MainStationRef(
                station_m=s_m,
                rotation_deg=legacy,
                station_rotation_x_deg=x_deg if val_x is not None else legacy,
                station_rotation_z_deg=z_deg if val_z is not None else 0.0,
                cs_name=(str(_pick(cs_names, i, "")).strip() or None),
                placement_id=str(_pick(place_ids, i, "")).strip() or None,
                desc=str(_pick(descs, i, "")).strip() or None,
            ))

    out.sort(key=lambda ms: (ms.station_m, ms.placement_id or "", ms.cs_name or ""))
    return out

## Deprecated: resolve_sections_for_object is no longer required since
## cross-sections are resolved via existing NCS loading utilities.
## Keeping stub commented for reference; can be removed after verification.
# def resolve_sections_for_object(...):
#     raise NotImplementedError("Deprecated – use dedicated NCS loaders instead")

# Alias for backward compatibility
MainStation = MainStationRef
