from __future__ import annotations
import os, json
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

# Fast JSON library
try:
    import orjson as _fastjson
except Exception:
    _fastjson = None

# Reuse helpers and models
from SPOT_Filters import get_json_files, load_json_objects
from Objects import (
    load_from_json, mapping, build_viso_object,
    CrossSection, DeckObject, PierObject, FoundationObject, MainStation, BearingArticulation, VisoContext, from_dict
)
from Axis import Axis as AxisCls
from AxisVariables import AxisVariable

def _is_axis_type(t):
    try:
        return issubclass(t, AxisCls)
    except Exception:
        return getattr(t, '__name__', '') == 'Axis'

def _read_json(path: str):
    if _fastjson:
        with open(path, "rb") as f:
            return _fastjson.loads(f.read())
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _as_dicts(rows):
    out = []
    for r in rows or []:
        if hasattr(r, "to_dict"):
            out.append(r.to_dict())
        elif isinstance(r, dict):
            out.append(r)
    return out

class SpotLoader:
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

    def load_raw(self) -> "SpotLoader":
        json_files = get_json_files(self.branch_folder)
        self._raw_rows = load_json_objects(json_files)
        if self.verbose:
            logger.info(f"[SpotLoader] Loaded {len(self._raw_rows)} rows from {len(json_files)} files.")
        return self

    def group_by_class(self) -> "SpotLoader":
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
            logger.info(f"[SpotLoader] by Class: {summary}")
        return self

    def context(self) -> VisoContext:
        axis_rows = self._by_class.get("Axis", [])
        cross_rows = self._by_class.get("CrossSection", [])
        ms_rows = self._by_class.get("MainStation", [])
        cross_sections, _ = load_from_json(CrossSection, cross_rows, mapping, axis_data=axis_rows)
        main_stations, _ = load_from_json(MainStation, ms_rows, mapping, axis_data=axis_rows)
        return VisoContext.from_json(axis_rows, cross_sections, main_stations, mapping_cfg=mapping)

    def build_domain_objects(self):
        axis_rows = self._by_class.get("Axis", [])
        cs_rows = self._by_class.get("CrossSection", [])
        deck_rows = self._by_class.get("DeckObject", [])
        pier_rows = self._by_class.get("PierObject", [])
        fnd_rows = self._by_class.get("FoundationObject", [])
        ms_rows = self._by_class.get("MainStation", [])
        ba_rows = self._by_class.get("BearingArticulation", [])

        cross_sections, _ = load_from_json(CrossSection, cs_rows, mapping, axis_data=axis_rows)
        deck_objects, deck_filtered = load_from_json(DeckObject, deck_rows, mapping, axis_data=axis_rows)
        pier_objects, pier_filtered = load_from_json(PierObject, pier_rows, mapping, axis_data=axis_rows)
        foundation_objects, found_filtered = load_from_json(FoundationObject, fnd_rows, mapping, axis_data=axis_rows)
        main_stations, ms_filtered = load_from_json(MainStation, ms_rows, mapping, axis_data=axis_rows)
        bearing_articulations, ba_filtered = load_from_json(BearingArticulation, ba_rows, mapping, axis_data=axis_rows)

        for i, obj in enumerate(deck_objects or []):
            if hasattr(obj, "axis_variables"):
                obj.set_axis_variables(deck_filtered[i].get("AxisVariables", []), mapping.get(AxisVariable, {}))
            if hasattr(obj, "axis_name") and axis_rows:
                obj.set_axis(obj.axis_name, axis_rows)

        for i, obj in enumerate(pier_objects or []):
            if hasattr(obj, "axis_variables"):
                obj.set_axis_variables(pier_filtered[i].get("AxisVariables", []), mapping.get(AxisVariable, {}))
            if hasattr(obj, "axis_name") and axis_rows:
                obj.set_axis(obj.axis_name, axis_rows)

        for i, obj in enumerate(foundation_objects or []):
            if hasattr(obj, "axis_variables"):
                obj.set_axis_variables(found_filtered[i].get("AxisVariables", []), mapping.get(AxisVariable, {}))
            if hasattr(obj, "axis_name") and axis_rows:
                obj.set_axis(obj.axis_name, axis_rows)

        for i, obj in enumerate(main_stations or []):
            if hasattr(obj, "axis_variables"):
                obj.set_axis_variables(ms_filtered[i].get("AxisVariables", []), mapping.get(AxisVariable, {}))
            if hasattr(obj, "axis_name") and axis_rows:
                obj.set_axis(obj.axis_name, axis_rows)

        for i, obj in enumerate(bearing_articulations or []):
            if hasattr(obj, "axis_variables"):
                obj.set_axis_variables(ba_filtered[i].get("AxisVariables", []), mapping.get(AxisVariable, {}))
            if hasattr(obj, "axis_name") and axis_rows:
                obj.set_axis(obj.axis_name, axis_rows)

        return axis_rows, cross_sections, deck_objects, pier_objects, foundation_objects, main_stations, bearing_articulations

    def build_vis_components(
        self,
        *,
        attach_section_json: bool = True,
        verbose: bool = True,
        axis_override=None,
        mainstations_override=None,
        cross_sections_override=None
    ):
        if not self._by_class:
            self.group_by_class()

        axis_rows = self._by_class.get('Axis', []) or []
        cs_rows = self._by_class.get('CrossSection', []) or []
        ms_rows = self._by_class.get('MainStation', []) or []

        axis_rows_for_ctx = axis_override if isinstance(axis_override, list) and axis_override and isinstance(axis_override[0], dict) else axis_rows
        if cross_sections_override is None:
            cs_objs, _ = load_from_json(CrossSection, cs_rows, mapping, axis_data=axis_rows_for_ctx)
        else:
            if isinstance(cross_sections_override, list) and cross_sections_override and isinstance(cross_sections_override[0], dict):
                cs_objs, _ = load_from_json(CrossSection, cross_sections_override, mapping, axis_data=axis_rows_for_ctx)
            else:
                cs_objs = cross_sections_override
        if mainstations_override is None:
            ms_objs, _ = load_from_json(MainStation, ms_rows, mapping, axis_data=axis_rows_for_ctx)
        else:
            if isinstance(mainstations_override, list) and mainstations_override and isinstance(mainstations_override[0], dict):
                ms_objs, _ = load_from_json(MainStation, mainstations_override, mapping, axis_data=axis_rows_for_ctx)
            else:
                ms_objs = mainstations_override

        ctx = VisoContext.from_json(
            axis_rows_for_ctx,
            cs_objs,
            ms_objs,
            mapping_cfg=mapping
        )

        deck_rows = self._by_class.get('DeckObject', []) or []
        pier_rows = self._by_class.get('PierObject', []) or []
        fnd_rows = self._by_class.get('FoundationObject', []) or []
        ba_rows = self._by_class.get('BearingArticulation', []) or []

        deck_objs, _ = load_from_json(DeckObject, deck_rows, mapping, axis_data=axis_rows_for_ctx)
        pier_objs, _ = load_from_json(PierObject, pier_rows, mapping, axis_data=axis_rows_for_ctx)
        fnd_objs, _ = load_from_json(FoundationObject, fnd_rows, mapping, axis_data=axis_rows_for_ctx)
        ba_objs, _ = load_from_json(BearingArticulation, ba_rows, mapping, axis_data=axis_rows_for_ctx)

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

        def _axis_for(obj):
            name = getattr(obj, "axis_name", None)
            if name and isinstance(name, str):
                hit = ctx.axes_by_name.get(name.strip())
                if hit is not None:
                    return hit
            return next(iter(ctx.axes_by_name.values()), None)

        vis_data_all: list[dict] = []
        vis_objs_all: list = []

        for obj in deck_objs:
            vis_row = build_viso_object(
                obj,
                ctx,
                axis=_axis_for(obj),
                mainstations=ctx.mainstations_by_name,
                cross_sections_override=list(ctx.crosssec_by_ncs.values()),
                mapping_cfg=mapping
            )
            if attach_section_json:
                vis_row = _attach_json_data(vis_row, self.branch_folder)
            vis_data_all.append(vis_row)
            vis_objs_all.append(obj)

        for obj in pier_objs:
            vis_row = build_viso_object(
                obj,
                ctx,
                axis=_axis_for(obj),
                mainstations=ctx.mainstations_by_name,
                cross_sections_override=list(ctx.crosssec_by_ncs.values()),
                mapping_cfg=mapping
            )
            if attach_section_json:
                vis_row = _attach_json_data(vis_row, self.branch_folder)
            vis_data_all.append(vis_row)
            vis_objs_all.append(obj)

        for obj in fnd_objs:
            vis_row = build_viso_object(
                obj,
                ctx,
                axis=_axis_for(obj),
                mainstations=ctx.mainstations_by_name,
                cross_sections_override=list(ctx.crosssec_by_ncs.values()),
                mapping_cfg=mapping
            )
            if attach_section_json:
                vis_row = _attach_json_data(vis_row, self.branch_folder)
            vis_data_all.append(vis_row)
            vis_objs_all.append(obj)

        for obj in ba_objs:
            vis_row = build_viso_object(
                obj,
                ctx,
                axis=_axis_for(obj),
                mainstations=ctx.mainstations_by_name,
                cross_sections_override=list(ctx.crosssec_by_ncs.values()),
                mapping_cfg=mapping
            )
            if attach_section_json:
                vis_row = _attach_json_data(vis_row, self.branch_folder)
            vis_data_all.append(vis_row)
            vis_objs_all.append(obj)

        self._deck_objects = deck_objs
        self._pier_objects = pier_objs
        self._foundation_objects = fnd_objs
        self._bearing_articulations = ba_objs
        self.vis_data = vis_data_all
        self.vis_objs = vis_objs_all

        if verbose:
            logger.info(f"[Vis] rows: {len(self.vis_data)}")
            for r in self.vis_data:
                logger.info(f"  - {r.get('name')} json_file= {r.get('json_file')} json_data= {bool(r.get('json_data'))}")

        return self