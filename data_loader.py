# spot_loader.py
from __future__ import annotations
import os, json
from typing import Dict, List, Tuple, Optional, Any

# Very fast JSON if present
try:
    import orjson as _fastjson
except Exception:
    _fastjson = None

# Reuse helpers from SPOT_Filters (unchanged)
from SPOT_Filters import get_json_files, load_json_objects

# Updated imports from models folder
from mapping import mapping
from cross_section import CrossSection
from deck_object import DeckObject
from viso_context import VisoContext
from base import BaseSpotObject
from main_station import MainStation
from axis import Axis as AxisCls

# Pier/Foundation optional
try:
    from pier_object import PierObject
    from foundation_object import FoundationObject
    _has_pier_found = True
except Exception:
    PierObject = FoundationObject = None
    _has_pier_found = False

from axis_variable import AxisVariable  # used to map axis variables

def _is_axis_type(t):
    try:
        return issubclass(t, AxisCls)
    except Exception:
        # fallback: name-only match if duplicated modules
        return getattr(t, '__name__', '') == 'Axis'

def _as_dicts(rows):
    """Accept dicts or SpotJsonObject; return list[dict]."""
    out = []
    for r in rows or []:
        if hasattr(r, "to_dict"):
            out.append(r.to_dict())
        elif isinstance(r, dict):
            out.append(r)
    return out


class SpotLoader:
    """
    Central loader that scans a branch folder (e.g. ...\GIT\MAIN),
    loads all '_*.json' files, splits by Class, materializes domain
    objects, and assembles final vis input (with JSON caching).
    """

    def __init__(
        self,
        master_folder: str,
        branch: str = "MAIN",
        *,
        cond_pier_foundation: bool = False,
        verbose: bool = True,
    ):
        self.master_folder = master_folder
        self.branch = branch
        self.cond_pier_foundation = cond_pier_foundation
        self.verbose = verbose

        self.branch_folder = os.path.join(master_folder, branch)
        if not os.path.isdir(self.branch_folder):
            raise FileNotFoundError(f"Branch folder not found: {self.branch_folder}")

        self._raw_rows: List[dict] = []
        self._by_class: Dict[str, List[dict]] = {}

    # -------- raw read ---------------------------------------------------------

    def load_raw(self) -> "SpotLoader":
        """
        Load & flatten all JSON rows from _*.json files in the branch folder.
        Uses SPOT_Filters.load_json_objects (already fast).
        """
        json_files = get_json_files(self.branch_folder)  # only files starting with "_"
        self._raw_rows = load_json_objects(json_files)   # flattened list
        if self.verbose:
            print(f"[SpotLoader] Loaded {len(self._raw_rows)} rows from {len(json_files)} files.")
        return self

    # -------- split by Class ---------------------------------------------------

    def group_by_class(self) -> "SpotLoader":
        """
        Group rows by their 'Class' key, skipping ClassInfo & CDDMapping.
        """
        rows = []
        for r in self._raw_rows:
            cls = str(r.get("Class", "")).strip()
            if cls in ("ClassInfo", "CDDMapping") or not cls:
                continue
            rows.append(r)

        by = {}
        for r in rows:
            by.setdefault(r["Class"], []).append(r)

        self._by_class = by
        if self.verbose:
            summary = {k: len(v) for k, v in sorted(by.items())}
            print(f"[SpotLoader] by Class: {summary}")
        return self

    # -------- materialize domain objects --------------------------------------

    def context(self) -> VisoContext:
        """
        Build a reusable context (Axes + CrossSections [+ MainStations later]).
        """
        axis_rows = self._by_class.get("Axis", [])
        cross_rows = self._by_class.get("CrossSection", [])
        cross_sections = CrossSection.load_from_json(cross_rows, mapping, axis_data=axis_rows)
        return VisoContext.from_json(axis_rows, cross_sections, mainstations=None)

    def build_domain_objects(self):
        """
        Materialize domain objects using classmethods.
        (Axis rows are left as dicts; objects use load_from_json.)
        """
        axis_rows = self._by_class.get("Axis", [])
        cs_rows   = self._by_class.get("CrossSection", [])
        deck_rows = self._by_class.get("DeckObject", [])
        pier_rows = self._by_class.get("PierObject", [])
        fnd_rows  = self._by_class.get("FoundationObject", [])

        cross_sections = CrossSection.load_from_json(cs_rows, mapping, axis_data=axis_rows)

        deck_objects = DeckObject.load_from_json(deck_rows, mapping, axis_data=axis_rows)

        pier_objects = foundation_objects = []
        if _has_pier_found:
            if pier_rows:
                pier_objects = PierObject.load_from_json(pier_rows, mapping, axis_data=axis_rows)
            if fnd_rows:
                foundation_objects = FoundationObject.load_from_json(fnd_rows, mapping, axis_data=axis_rows)

        return axis_rows, cross_sections, deck_objects, pier_objects, foundation_objects

    def build_vis_components(
        self,
        *,
        attach_section_json: bool = True,
        verbose: bool = True,
        axis_override=None,             
        mainstations_override=None,     
        cross_sections_override=None    
    ):
        """
        Component-driven build:
        1) Create a VisoContext from Axis/CrossSection/MainStation collections.
        2) Build Deck, Pier, Foundation objects by injecting the context
            (so you can swap Axis/MainStations/CrossSections if desired).
        3) Populate self.vis_data and self.vis_objs in the same shape as the old build_vis().

        Notes:
        - We DO NOT write or mutate any JSON files; when attach_section_json is True,
            we just read the CrossSection JSON once and attach it to the vis row in memory.
        - axis_override is expected to be a list of raw JSON rows (dicts). If not provided,
            we use the Axis rows we already loaded from CSV/JSON.
        """
        if not self._by_class:
            self.group_by_class()

        # ---- 1) Collect core component rows ----
        axis_rows = self._by_class.get('Axis', []) or []
        cs_rows   = self._by_class.get('CrossSection', []) or []
        ms_rows   = self._by_class.get('MainStation', []) or []

        # If an override is provided, prefer it when it looks like raw dict rows
        axis_rows_for_ctx = axis_override if isinstance(axis_override, list) and axis_override and isinstance(axis_override[0], dict) else axis_rows

        # Parse CrossSections/MainStations into objects (keep raw axis rows; VisoContext will build Axis objects)
        if cross_sections_override is None:
            cs_objs = CrossSection.load_from_json(cs_rows, mapping, axis_data=axis_rows_for_ctx)
        else:
            # Allow passing either objects or raw rows
            if isinstance(cross_sections_override, list) and cross_sections_override and isinstance(cross_sections_override[0], dict):
                cs_objs = CrossSection.load_from_json(cross_sections_override, mapping, axis_data=axis_rows_for_ctx)
            else:
                cs_objs = cross_sections_override

        if mainstations_override is None:
            ms_objs = MainStation.load_from_json(ms_rows, mapping, axis_data=axis_rows_for_ctx)
        else:
            if isinstance(mainstations_override, list) and mainstations_override and isinstance(mainstations_override[0], dict):
                ms_objs = MainStation.load_from_json(mainstations_override, mapping, axis_data=axis_rows_for_ctx)
            else:
                ms_objs = mainstations_override

        # ---- 2) Build a context (indexes by name/NCS) from RAW axis rows + parsed CS/MS ----
        ctx = VisoContext.from_json(
            axis_rows_for_ctx,
            cs_objs,
            ms_objs,
        )

        # ---- 3) Parse Viso objects (Deck / Pier / Foundation) into dataclasses ----
        deck_rows = self._by_class.get('DeckObject', []) or []
        pier_rows = self._by_class.get('PierObject', []) or []
        fnd_rows  = self._by_class.get('FoundationObject', []) or []

        deck_objs = DeckObject.load_from_json(deck_rows, mapping, axis_data=axis_rows_for_ctx)
        pier_objs = PierObject.load_from_json(pier_rows, mapping, axis_data=axis_rows_for_ctx) if _has_pier_found else []
        fnd_objs  = FoundationObject.load_from_json(fnd_rows, mapping, axis_data=axis_rows_for_ctx) if _has_pier_found else []

        # ---- 4) Optional utility: attach CrossSection JSON once per vis row ----
        def _attach_json_data(row: dict, branch_root: str):
            jf = (row.get("json_file") or "").strip()
            if not jf or jf.lower().startswith("no master"):
                row["json_data"] = None
                return row

            candidate = jf
            if not os.path.isabs(candidate):
                candidate = os.path.normpath(os.path.join(branch_root, candidate))

            if os.path.exists(candidate):
                with open(candidate, "r", encoding="utf-8") as f:
                    row["json_data"] = json.load(f)
            else:
                row["json_data"] = None
            return row

        # ---- 5) Build vis rows via dependency injection ----
        vis_data_all: list[dict] = []
        vis_objs_all:  list      = []

        # Helper to pick an axis for an object: named axis from ctx, else any/default
        def _axis_for(obj):
            name = getattr(obj, "axis_name", None)
            if name and isinstance(name, str):
                hit = ctx.axes_by_name.get(name.strip())
                if hit is not None:
                    return hit
            return next(iter(ctx.axes_by_name.values()), None)

        # Keep ordering: Decks → Piers → Foundation
        for obj in deck_objs:
            vis_row = BaseSpotObject.build_viso_object(
                obj,
                ctx,
                axis=_axis_for(obj),
                mainstations=ctx.mainstations_by_name,
                cross_sections_override=list(ctx.crosssec_by_ncs.values()),
            )
            if attach_section_json:
                vis_row = _attach_json_data(vis_row, self.branch_folder)  # Use branch_folder for relative paths
            vis_data_all.append(vis_row)
            vis_objs_all.append(obj)

        for obj in pier_objs:
            vis_row = BaseSpotObject.build_viso_object(
                obj,
                ctx,
                axis=_axis_for(obj),
                mainstations=ctx.mainstations_by_name,
                cross_sections_override=list(ctx.crosssec_by_ncs.values()),
            )
            if attach_section_json:
                vis_row = _attach_json_data(vis_row, self.branch_folder)
            vis_data_all.append(vis_row)
            vis_objs_all.append(obj)

        for obj in fnd_objs:
            vis_row = BaseSpotObject.build_viso_object(
                obj,
                ctx,
                axis=_axis_for(obj),
                mainstations=ctx.mainstations_by_name,
                cross_sections_override=list(ctx.crosssec_by_ncs.values()),
            )
            if attach_section_json:
                vis_row = _attach_json_data(vis_row, self.branch_folder)
            vis_data_all.append(vis_row)
            vis_objs_all.append(obj)

        # ---- 6) Preserve fields expected by main.py ----
        self._deck_objects       = deck_objs
        self._pier_objects       = pier_objs
        self._foundation_objects = fnd_objs

        self.vis_data = vis_data_all
        self.vis_objs = vis_objs_all

        if verbose:
            print(f"[Vis] rows: {len(self.vis_data)}")
            for r in self.vis_data:
                print(f"  - {r.get('name')} json_file= {r.get('json_file')} json_data= {bool(r.get('json_data'))}")

        return self