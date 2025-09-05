# spot_loader.py
from __future__ import annotations
import os, json
from typing import Dict, List, Tuple, Optional, Any



# Very fast JSON if present
try:
    import orjson as _fastjson
except Exception:
    _fastjson = None

# Reuse your helpers & models
from SPOT_Filters import get_json_files, load_json_objects   # folder scan + bulk load  :contentReference[oaicite:6]{index=6}
from models import Axis, MainStation, CrossSection, DeckObject, PierObject, FoundationObject, VisoContext, AxisVariable, load_from_json, build_viso_object
from models.mapping import mapping as MAP


# from AddClasses import (
#     load_from_json, mapping,
#     CrossSection, DeckObject,
#     VisoContext, build_viso_object, BaseObject, MainStation, from_dict
# )


def _is_axis_type(t):
    try:
        return issubclass(t, Axis)
    except Exception:
        # fallback: name-only match if duplicated modules
        return getattr(t, '__name__', '') == 'Axis'
# Pier/Foundation optional
# try:
#     from AddClasses import PierObject, FoundationObject
#     _has_pier_found = True
# except Exception:
#     PierObject = FoundationObject = None
#     _has_pier_found = False
_has_pier_found = True
#from AxisVariables import AxisVariable  # used to map axis variables  :contentReference[oaicite:7]{index=7}


def _read_json(path: str):
    """
    Small helper: read JSON file as Python object.
    Uses orjson if available (2–5x faster on large files).
    """
    if _fastjson:
        with open(path, "rb") as f:
            return _fastjson.loads(f.read())
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
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
    objects (with your existing AddClasses loader), and assembles
    final vis input (with JSON caching).
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
        Uses SPOT_Filters.load_json_objects (already fast).  :contentReference[oaicite:8]{index=8}
        """
        json_files = get_json_files(self.branch_folder)  # only files starting with "_"  :contentReference[oaicite:9]{index=9}
        # If you prefer orjson per-file speed here, you can re-implement
        # a tiny loop, but SPOT_Filters.load_json_objects is fine:
        self._raw_rows = load_json_objects(json_files)   # flattened list  :contentReference[oaicite:10]{index=10}
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
        cross_sections, _ = load_from_json(CrossSection, cross_rows, MAP, axis_data=axis_rows)
        return VisoContext.from_json(axis_rows, cross_sections, mainstations=None, mapping_cfg=MAP)
    
    def get_ctx(self) -> VisoContext:
        return getattr(self, "ctx", None)

    def get_axis(self, name: str):
        return self.ctx.get_axis(name) if self.get_ctx() else None

    def get_cross_section(self, ncs: int):
        return self.ctx.get_cross_section(ncs) if self.get_ctx() else None

    def get_mainstation(self, name: str):
        return self.ctx.get_mainstation(name) if self.get_ctx() else None

    def objects(self, clsname: str | None = None):
        return self.ctx.all_objects(clsname) if self.get_ctx() else []

    def objects_for_axis(self, axis_name: str, clsname: str | None = None):
        return self.ctx.objects_for_axis(axis_name, clsname) if self.get_ctx() else []

    def get_object_by_name(self, name: str):
        return self.ctx.get_object_by_name(name) if self.get_ctx() else None

    def get_object_by_id(self, oid):
        return self.ctx.get_object_by_id(oid) if self.get_ctx() else None

    def build_domain_objects(self):
        """
        Your current path that materializes domain objects.
        (Axis rows are left as dicts; objects use load_from_json.)
        """
        axis_rows = self._by_class.get("Axis", [])
        cs_rows   = self._by_class.get("CrossSection", [])
        deck_rows = self._by_class.get("DeckObject", [])
        pier_rows = self._by_class.get("PierObject", [])
        fnd_rows  = self._by_class.get("FoundationObject", [])

        cross_sections, _ = load_from_json(CrossSection, cs_rows, MAP, axis_data=axis_rows)

        deck_objects, deck_filtered = load_from_json(DeckObject, deck_rows, MAP, axis_data=axis_rows)
        for i, dobj in enumerate(deck_objects or []):
            if hasattr(dobj, "axis_variables"):
                dobj.set_axis_variables(deck_filtered[i].get("AxisVariables", []), MAP.get(AxisVariable, {}))
            if hasattr(dobj, "axis_name") and axis_rows:
                dobj.set_axis(dobj.axis_name, axis_rows)

        pier_objects = foundation_objects = []
        if _has_pier_found:
            if pier_rows:
                pier_objects, pier_filtered = load_from_json(PierObject, pier_rows, MAP, axis_data=axis_rows)
                for i, pobj in enumerate(pier_objects or []):
                    if hasattr(pobj, "axis_variables"):
                        pobj.set_axis_variables(pier_filtered[i].get("AxisVariables", []), MAP.get(AxisVariable, {}))
                    if hasattr(pobj, "axis_name") and axis_rows:
                        pobj.set_axis(pobj.axis_name, axis_rows)
            if fnd_rows:
                foundation_objects, found_filtered = load_from_json(FoundationObject, fnd_rows, MAP, axis_data=axis_rows)
                for i, fobj in enumerate(foundation_objects or []):
                    if hasattr(fobj, "axis_variables"):
                        fobj.set_axis_variables(found_filtered[i].get("AxisVariables", []), MAP.get(AxisVariable, {}))
                    if hasattr(fobj, "axis_name") and axis_rows:
                        fobj.set_axis(fobj.axis_name, axis_rows)

        return axis_rows, cross_sections, deck_objects, deck_filtered, pier_objects, foundation_objects

    def build_vis_components(
        self,
        *,
        attach_section_json: bool = True,
        verbose: bool = True,
        axis_override=None,             # optional: list[dict] raw Axis rows to use instead of CSV rows
        mainstations_override=None,     # optional: list[MainStation] objects (or raw rows; see below)
        cross_sections_override=None    # optional: list[CrossSection] objects (or raw rows; see below)
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
            cs_objs, _ = load_from_json(CrossSection, cs_rows, MAP, axis_data=axis_rows_for_ctx)
        else:
            # Allow passing either objects or raw rows
            if isinstance(cross_sections_override, list) and cross_sections_override and isinstance(cross_sections_override[0], dict):
                cs_objs, _ = load_from_json(CrossSection, cross_sections_override, MAP, axis_data=axis_rows_for_ctx)
            else:
                cs_objs = cross_sections_override

        if mainstations_override is None:
            ms_objs, _ = load_from_json(MainStation, ms_rows, MAP, axis_data=axis_rows_for_ctx)
        else:
            if isinstance(mainstations_override, list) and mainstations_override and isinstance(mainstations_override[0], dict):
                ms_objs, _ = load_from_json(MainStation, mainstations_override, MAP, axis_data=axis_rows_for_ctx)
            else:
                ms_objs = mainstations_override

        # ---- 2) Build a context (indexes by name/NCS) from RAW axis rows + parsed CS/MS ----
        # VisoContext.from_json expects raw Axis/MainStation/CrossSection rows OR lists of objects,
        # but it is robust in your codebase to build the lookups we need.
        ctx = VisoContext.from_json(
            axis_rows_for_ctx,
            cs_objs,
            ms_objs,
            mapping_cfg=MAP
        )

        # ---- 3) Parse Viso objects (Deck / Pier / Foundation) into dataclasses ----
        deck_rows = self._by_class.get('DeckObject', []) or []
        pier_rows = self._by_class.get('PierObject', []) or []
        fnd_rows  = self._by_class.get('FoundationObject', []) or []

        deck_objs, _ = load_from_json(DeckObject, deck_rows, MAP, axis_data=axis_rows_for_ctx)
        pier_objs, _ = load_from_json(PierObject,  pier_rows, MAP, axis_data=axis_rows_for_ctx)
        fnd_objs,  _ = load_from_json(FoundationObject, fnd_rows, MAP, axis_data=axis_rows_for_ctx)

        axis_var_map = MAP.get(AxisVariable, {})
        ctx.add_objects(deck_objs, pier_objs, fnd_objs, axis_var_map=axis_var_map)
        self.ctx = ctx


        # ---- 4) Optional utility: attach CrossSection JSON once per vis row ----
        # def _attach_json_data(vis_row: dict) -> dict:
        #     jf = vis_row.get('json_file')
        #     if not jf:
        #         return vis_row
        #     try:
        #         with open(jf, 'r', encoding='utf-8') as f:
        #             vis_row['json_data'] = json.load(f)
        #     except Exception as e:
        #         if verbose:
        #             print(f"[SpotLoader] WARN: cannot open section JSON: {jf} ({e})")
        #     return vis_row

        def _attach_json_data(row: dict, branch_root: str):
            jf = (row.get("json_file") or "").strip()
            if not jf or jf.lower().startswith("no master"):
                row["json_data"] = None
                return row

            import os, json
            # if relative, join with branch
            candidate = jf
            if not os.path.isabs(candidate):
                candidate = os.path.normpath(os.path.join(branch_root, candidate))

            if os.path.exists(candidate):
                with open(candidate, "r", encoding="utf-8") as f:
                    row["json_data"] = json.load(f)
            else:
                # don't drop the row if json is missing; just leave json_data=None
                row["json_data"] = None
            return row

        # ---- 5) Build vis rows via dependency injection ----
        vis_data_all: list[dict] = []
        vis_objs_all:  list      = []

        # Helper to pick an axis for an object: named axis from ctx, else any/default
        def _axis_for(obj):
            # Try common attribute names; VisoContext.get_axis() handles case/spacing
            name = getattr(obj, "axis_name", None) or getattr(obj, "object_axis_name", None)
            ax = ctx.get_axis(name) if name else None
            if ax is not None:
                return ax
            # last-resort fallback: first axis in the context
            return next(iter(ctx.axes_by_name.values()), None)

        # Keep ordering: Decks → Piers → Foundation (like your old build)
        for obj in deck_objs:
            vis_row = build_viso_object(
                obj,
                ctx,
                axis=_axis_for(obj),
                mainstations=ctx.mainstations_by_name,
                cross_sections_override=list(ctx.crosssec_by_ncs.values()),  # lets create_input_for_visualisation resolve by NCS
                mapping_cfg=MAP
            )
            if attach_section_json:
                vis_row = _attach_json_data(vis_row, "MASTER_SECTION")
            vis_data_all.append(vis_row)
            vis_objs_all.append(obj)

        for obj in pier_objs:
            vis_row = build_viso_object(
                obj,
                ctx,
                axis=_axis_for(obj),
                mainstations=ctx.mainstations_by_name,
                cross_sections_override=list(ctx.crosssec_by_ncs.values()),
                mapping_cfg=MAP
            )
            if attach_section_json:
                vis_row = _attach_json_data(vis_row, "MASTER_SECTION")
            vis_data_all.append(vis_row)
            vis_objs_all.append(obj)

        for obj in fnd_objs:
            vis_row = build_viso_object(
                obj,
                ctx,
                axis=_axis_for(obj),
                mainstations=ctx.mainstations_by_name,
                cross_sections_override=list(ctx.crosssec_by_ncs.values()),
                mapping_cfg=MAP
            )
            if attach_section_json:
                vis_row = _attach_json_data(vis_row, "MASTER_SECTION")
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
