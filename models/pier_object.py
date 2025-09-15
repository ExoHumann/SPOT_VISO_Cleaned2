from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import BaseObject
from .cross_section import CrossSection
from .axis import Axis
from .axis_variable import AxisVariable


from dataclasses import dataclass, field
from typing import List, Dict, Optional
from .base import BaseObject

@dataclass(kw_only=True)
class PierObject(BaseObject):
    # identity / metadata
    no: str = ""
    class_name: str = "PierObject"
    type: str = ""
    description: str = ""
    name: str = ""
    inactive: str = "0"

    # axis
    object_axis_name: str = ""                 # leave if you also use axis_name elsewhere
    axis_variables: List[Dict] = field(default_factory=list)

    # cross-section references (DEFAULTS ADDED)
    bot_cross_section_name: str = ""           # <- default added
    top_cross_section_name: str = ""           # <- default added
    internal_cross_section_ncs: List[int] = field(default_factory=list)

    # geometry controls (DEFAULT ADDED)
    slices: int = 0                             # <- default added (0 => use axis stations)
    top_zoffset: float = 0.0
    bot_pier_elevation: float = 0.0
    rotation_angle: float = 0.0

    # other fields you already had (give safe defaults)
    grp: int = 0
    fixation: str = ""
    internal_placement_id: List[str] = field(default_factory=list)
    internal_ref_placement_id: List[str] = field(default_factory=list)
    internal_ref_station_offset: List[float] = field(default_factory=list)
    grp_offset: List[float] = field(default_factory=list)

    # optional helper names used elsewhere
    top_cross_section_points_name: str = ""
    bot_cross_section_points_name: str = ""
    internal_station_value: List[float] = field(default_factory=list)

    # ------------------------------------------------------------
    # New: simple metadata formatter (optional; preserves your style)
    # ------------------------------------------------------------
    def get_object_metadata(self) -> Dict:
        data = asdict(self)
        # compress noisy fields like base does
        data['axis_variables'] = f"<{len(self.axis_variables_obj)} axis variables>"
        data['axis_variables_obj'] = f"<{len(self.axis_variables_obj)} axis variable objects>"
        data['axis_obj'] = "<Axis object>" if getattr(self, "axis_obj", None) is not None else None
        # remove UI-only noise
        data.pop('axis_variables_obj', None)
        data.pop('axis_obj', None)
        return data

    # ------------------------------------------------------------
    # Ownership: pier-specific geometry
    # ------------------------------------------------------------
    def compute_length_mm(self) -> float:
        """
        Pier length in mm.
        Uses bottom pier elevation and top_zoffset. Assumes inputs in meters unless clearly mm.
        """
        bot = float(self.bot_pier_elevation or 0.0)
        top = float(self.top_zoffset or 0.0)
        # Heuristic: values < 1000 are likely meters -> convert to mm
        if abs(bot) < 1000.0: bot *= 1000.0
        if abs(top) < 1000.0: top *= 1000.0
        return float(top - bot)

    def _derive_stations_mm(self, *, axis: Axis, slices: Optional[int] = None) -> np.ndarray:
        """
        Build stations along the axis for this pier:
        - if user_stations set on the object -> use those
        - else if slices set -> interpolate evenly between visible axis range
        - else -> use full axis stations (dense path)
        """
        if getattr(self, "user_stations", None):
            s_mm = np.asarray(self.user_stations, float) * 1000.0
            return s_mm

        if slices is None:
            slices = int(getattr(self, "slices", 0) or 0)

        axS = np.asarray(getattr(axis, "stations", []), float)
        if axS.size == 0:
            return np.zeros((0,), float)

        smin, smax = float(np.min(axS)), float(np.max(axS))
        if slices and slices > 1:
            return np.linspace(smin, smax, num=int(slices), dtype=float)

        # fallback: use original axis stations
        return axS.copy()

    # ------------------------------------------------------------
    # Main: compute and embed geometry
    # ------------------------------------------------------------
    def compute_geometry(
        self,
        *,
        ctx,
        stations_m: Optional[List[float]] = None,   # unused for piers
        slices: Optional[int] = None,               # vertical slices (>=2)
        twist_deg: float = 0.0,                     # not used for the body (no twist)
        negate_x: bool = False,                     # we place manually; local X untouched
        default_drop_mm: float = 12000.0,
    ) -> Dict[str, object]:
        """
        Minimal port of your main.py 'build_pier_matrices_for_plot' logic.
        Orientation: pier axis = Z_dir of the DECK frame at this pier's station.
        """

        # ---------- quick helpers (keep in class or adapt to your existing ones) ----------
        def _clamp_station_m(axis: Axis, station_m: float) -> float:
            Smm = np.asarray(axis.stations, float)
            if Smm.size == 0 or not np.isfinite(Smm).any():
                return float(station_m)
            s_mm = float(station_m) * 1000.0
            s_eff = float(np.clip(s_mm, np.nanmin(Smm), np.nanmax(Smm)))
            return s_eff / 1000.0

        def _interp_curve_and_tangent(ax: Axis, s_mm: float) -> Tuple[np.ndarray, np.ndarray]:
            S = np.asarray(ax.stations, float)
            X = np.asarray(ax.x_coords, float)
            Y = np.asarray(ax.y_coords, float)
            Z = np.asarray(ax.z_coords, float)
            x = np.interp(s_mm, S, X); y = np.interp(s_mm, S, Y); z = np.interp(s_mm, S, Z)
            ds = max(1.0, 0.001 * (S[-1] - S[0]))
            s0 = float(np.clip(s_mm - ds, S[0], S[-1]))
            s1 = float(np.clip(s_mm + ds, S[0], S[-1]))
            p0 = np.array([np.interp(s0, S, X), np.interp(s0, S, Y), np.interp(s0, S, Z)], float)
            p1 = np.array([np.interp(s1, S, X), np.interp(s1, S, Y), np.interp(s1, S, Z)], float)
            T = p1 - p0
            n = float(np.linalg.norm(T))
            if n == 0.0: T[:] = (1.0, 0.0, 0.0)
            else: T /= n
            return np.array([x, y, z], float), T

        def _deck_frame(deck_axis: Axis, station_m: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Return right-handed frame at deck station: (Tdeck, Y_dir, Z_dir)."""
            Zg = np.array([0.0, 0.0, 1.0], float)
            _, Tdeck = _interp_curve_and_tangent(deck_axis, station_m * 1000.0)
            Y_dir = np.cross(Tdeck, Zg); nY = float(np.linalg.norm(Y_dir))
            if nY < 1e-9:
                Xg = np.array([1.0, 0.0, 0.0], float)
                Y_dir = np.cross(Tdeck, Xg); nY = float(np.linalg.norm(Y_dir))
                if nY < 1e-9:
                    Yg = np.array([0.0, 1.0, 0.0], float)
                    Y_dir = np.cross(Tdeck, Yg); nY = float(np.linalg.norm(Y_dir))
            Y_dir /= max(nY, 1e-9)
            Z_dir = np.cross(Y_dir, Tdeck); Z_dir /= max(float(np.linalg.norm(Z_dir)), 1e-9)
            if np.dot(Z_dir, Zg) < 0.0:
                Z_dir = -Z_dir; Y_dir = -Y_dir
            return Tdeck, Y_dir, Z_dir

        def _idx_or_default(ids: List[str], wanted: Optional[str], *, alt: Optional[str] = None) -> int:
            try:
                if wanted and wanted in ids: return ids.index(wanted)
                if alt and alt in ids: return ids.index(alt)
            except Exception:
                pass
            return 0

        # ---------- inputs & quick exits ----------
        axis: Optional[Axis] = getattr(self, "axis_obj", None)
        if axis is None:
            return {"ids": [], "stations_mm": np.array([], float),
                    "points_mm": np.zeros((0, 0, 3), float),
                    "local_Y_mm": np.zeros((0, 0), float),
                    "local_Z_mm": np.zeros((0, 0), float),
                    "loops_idx": [], "length_mm": 0.0}

        # Find a deck object (prefer type == "Deck")
        deck_obj = None
        try:
            cand = list(getattr(ctx, "objects_by_class", {}).get("DeckObject", []))
            if not cand:
                cand = list(getattr(ctx, "decks_by_name", {}).values())
            for o in cand:
                if getattr(o, "type", "") == "Deck":
                    deck_obj = o; break
            if deck_obj is None and cand:
                deck_obj = cand[0]
        except Exception:
            pass
        deck_axis = getattr(deck_obj, "axis_obj", None) or axis

        # Station to use (simple & faithful to your old main: StationValue → clamp to *deck* axis)
        st_raw = float(getattr(self, "station_value", 0.0) or 0.0)
        st_m   = _clamp_station_m(deck_axis, st_raw)

        # Names & offsets
        top_name = getattr(self, "top_cross_section_points_name", "") or ""
        bot_name = getattr(self, "bot_cross_section_points_name", "") or top_name
        top_y_off_mm = float(getattr(self, "top_yoffset", 0.0))
        top_z_off_mm = float(getattr(self, "top_zoffset", 0.0))
        bot_y_off_mm = float(getattr(self, "bot_yoffset", 0.0))
        bot_z_off_mm = float(getattr(self, "bot_zoffset", 0.0))
        pier_elev_mm = float(getattr(self, "bot_pier_elevation", 0.0) or 0.0)
        if abs(pier_elev_mm) < 1000.0: pier_elev_mm *= 1000.0  # m → mm

        # ---------- frame at deck, pier axis = Z_dir ----------
        Tdeck, Y_dir, Z_dir = _deck_frame(deck_axis, st_m)
        Tpier = Z_dir  # your requirement

        # ---------- anchor on the DECK section point (fallback: curve point) ----------
        # Get the deck slice at st_m so we can pick a named point
        top_anchor_mm: np.ndarray
        Pd0 = None; deck_ids: List[str] = []
        if deck_obj is not None:
            try:
                dgeo = deck_obj.compute_geometry(ctx=ctx, stations_m=[st_m], twist_deg=90.0, negate_x=True)
                deck_ids = dgeo.get("ids") or []
                Pdeck = np.asarray(dgeo.get("points_mm"), float)
                if Pdeck.ndim == 3 and Pdeck.shape[0] >= 1:
                    Pd0 = Pdeck[0]  # (N,3)
            except Exception:
                Pd0 = None

        if Pd0 is not None and deck_ids:
            j_top = _idx_or_default(deck_ids, top_name, alt="RA") if top_name else 0
            top_anchor_mm = Pd0[j_top].astype(float)
        else:
            # fallback: point on the deck curve
            S = np.asarray(deck_axis.stations, float)
            X = np.asarray(deck_axis.x_coords, float)
            Y = np.asarray(deck_axis.y_coords, float)
            Z = np.asarray(deck_axis.z_coords, float)
            s_mm = st_m * 1000.0
            top_anchor_mm = np.array([np.interp(s_mm, S, X),
                                    np.interp(s_mm, S, Y),
                                    np.interp(s_mm, S, Z)], float)

        # In-plane basis (U,V) in the section plane ⟂ Tpier
        U = Y_dir          # lateral
        V = Tdeck          # longitudinal

        # Apply top offsets (in-plane)
        top_anchor_mm = top_anchor_mm + top_y_off_mm * U + top_z_off_mm * V

        # Length / bottom anchor (downward from top along -Z_dir; axis itself is Z_dir)
        drop_mm = pier_elev_mm + bot_z_off_mm
        if drop_mm < 1000.0: drop_mm = float(default_drop_mm)
        bottom_anchor_mm = top_anchor_mm + (-drop_mm) * Tpier + bot_y_off_mm * U

        # ---------- evaluate the pier's 2D section ONCE (local arrays only) ----------
        # pick a section: prefer named top/bot, else by NCS, else empty
        top_cs = None
        if getattr(self, "top_cross_section_name", None):
            top_cs = getattr(ctx, "crosssec_by_name", {}).get(self.top_cross_section_name)
        if top_cs is None:
            ncs_list = getattr(self, "internal_cross_section_ncs", []) or []
            if ncs_list:
                top_cs = getattr(ctx, "crosssec_by_ncs", {}).get(int(ncs_list[-1]))
        if top_cs is None:
            return {"ids": [], "stations_mm": np.array([], float),
                    "points_mm": np.zeros((0, 0, 3), float),
                    "local_Y_mm": np.zeros((0, 0), float),
                    "local_Z_mm": np.zeros((0, 0), float),
                    "loops_idx": [], "length_mm": float(drop_mm)}

        # Axis variables at this station (meters in, engine handles units)
        var_rows = AxisVariable.evaluate_at_stations_cached(getattr(self, "axis_variables_obj", None), [st_m]) or [{}]
        ids, X_mm, Y_mm, loops_idx = top_cs.compute_local_points(axis_var_results=var_rows, negate_x=False)
        if not ids:
            return {"ids": [], "stations_mm": np.array([], float),
                    "points_mm": np.zeros((0, 0, 3), float),
                    "local_Y_mm": np.zeros((0, 0), float),
                    "local_Z_mm": np.zeros((0, 0), float),
                    "loops_idx": [], "length_mm": float(drop_mm)}

        # Engine convention: local Y == X_mm, local Z == Y_mm
        y_mm = np.asarray(X_mm[0], float)  # (N,)
        z_mm = np.asarray(Y_mm[0], float)  # (N,)

        # ---------- build S slices along the pier axis; sections live in plane (U,V) ----------
        S = max(2, int(slices or getattr(self, "slices", 0) or 2))
        alphas = np.linspace(0.0, 1.0, S)[:, None]                # (S,1)
        line_vec = (bottom_anchor_mm - top_anchor_mm)[None, :]    # (1,3)
        anchors = top_anchor_mm[None, :] + alphas * line_vec      # (S,3)

        offs = (y_mm[:, None] * U[None, :]) + (z_mm[:, None] * V[None, :])  # (N,3)
        P_mm = anchors[:, None, :] + offs[None, :, :]                         # (S,N,3)

        # repeat local coords for hover/meta
        X_out = np.tile(y_mm[None, :], (S, 1))
        Y_out = np.tile(z_mm[None, :], (S, 1))
        length_mm = float(np.linalg.norm(line_vec))
        stations_mm = np.linspace(0.0, length_mm, S)

        # ---------- per-pier Axis (NOT RA) for plotting ----------
        pier_axis = Axis(
            stations=[0.0, length_mm / 1000.0],
            x_coords=[float(top_anchor_mm[0]) / 1000.0, float(bottom_anchor_mm[0]) / 1000.0],
            y_coords=[float(top_anchor_mm[1]) / 1000.0, float(bottom_anchor_mm[1]) / 1000.0],
            z_coords=[float(top_anchor_mm[2]) / 1000.0, float(bottom_anchor_mm[2]) / 1000.0],
            units="m",
        )
        try:
            pier_axis.name = f"{getattr(self,'name','Pier')} Axis"
        except Exception:
            pass
        self._generated_axis_obj = pier_axis  # handy for plotting

        return {
            "ids": ids,
            "stations_mm": stations_mm,
            "points_mm": P_mm,
            "local_Y_mm": X_out,
            "local_Z_mm": Y_out,
            "loops_idx": loops_idx,
            "length_mm": length_mm,
            "pier_axis": pier_axis,
        }
