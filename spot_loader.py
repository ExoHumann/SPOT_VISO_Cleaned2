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
from AddClasses import (
    load_from_json, mapping,
    CrossSection, DeckObject, BaseObject, from_dict  # core
)
# Pier/Foundation are optional
try:
    from AddClasses import PierObject, FoundationObject
    _has_pier_found = True
except Exception:
    PierObject = FoundationObject = None
    _has_pier_found = False

from AxisVariables import AxisVariable  # used to map axis variables  :contentReference[oaicite:7]{index=7}


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

    def build_domain_objects(self) -> Tuple[
        List[dict],        # axis_data (list[dict])
        List[CrossSection],
        List[DeckObject],
        List[dict],        # filtered deck (raw)
        Optional[List[Any]], Optional[List[Any]]  # pier, foundation (if enabled/available)
    ]:
        """
        Materialize your dataclasses using your current `load_from_json`.
        """
        axis_data: List[dict] = self._by_class.get("Axis", [])
        cs_data:   List[dict] = self._by_class.get("CrossSection", [])
        deck_data: List[dict] = self._by_class.get("DeckObject", [])

        # Cross-sections
        cross_sections, _ = load_from_json(CrossSection, cs_data, mapping)

        # Decks
        deck_rows = self._by_class.get("DeckObject", [])
        deck_objects = [from_dict(DeckObject, r, mapping, axis_data) for r in deck_rows]
        deck_filtered = deck_rows  # you already filtered ClassInfo upstream

        #deck_objects, deck_filtered = load_from_json(DeckObject, deck_data, mapping, axis_data=axis_data)

        # Post-create pass: keep your current pattern
        for i, obj in enumerate(deck_objects):
            # axis vars
            if hasattr(obj, "axis_variables") and isinstance(obj.axis_variables, list):
                obj.set_axis_variables(deck_filtered[i].get("AxisVariables", []), mapping.get(AxisVariable, {}))
            # axis
            if hasattr(obj, "axis_name") and axis_data:
                obj.set_axis(obj.axis_name, axis_data)

        pier_objects = foundation_objects = None
        if self.cond_pier_foundation and _has_pier_found:
            # Load only if available & requested
            pier_data   = self._by_class.get("PierObject", [])
            found_data  = self._by_class.get("FoundationObject", [])

            if pier_data:

                # --- normalize pier rows (singular/plural & spelling variants) ---
                norm_pier_data = []
                for row in pier_data:
                    r = dict(row)

                    # singular vs plural “Point(s)@Name”
                    if "Top-CrossSection_Point@Name" in r and "Top-CrossSection_Points@Name" not in r:
                        r["Top-CrossSection_Points@Name"] = r["Top-CrossSection_Point@Name"]
                    if "Bot-CrossSection_Point@Name" in r and "Bot-CrossSection_Points@Name" not in r:
                        r["Bot-CrossSection_Points@Name"]  = r["Bot-CrossSection_Point@Name"]

                    # StationValue (with t) -> StaionValue (current mapping)
                    if "Internal@StationValue" in r and "Internal@StaionValue" not in r:
                        r["Internal@StaionValue"] = r["Internal@StationValue"]

                    norm_pier_data.append(r)

                # pier_objects, pier_filtered = load_from_json(
                #     PierObject, norm_pier_data, mapping, axis_data=axis_data
                # )
                pier_objects, pier_filtered = load_from_json(PierObject, pier_data, mapping, axis_data=axis_data)
                for i, pobj in enumerate(pier_objects or []):
                    if hasattr(pobj, "axis_variables") and isinstance(pobj.axis_variables, list):
                        pobj.set_axis_variables(pier_filtered[i].get("AxisVariables", []), mapping.get(AxisVariable, {}))
                    if hasattr(pobj, "axis_name") and axis_data:
                        pobj.set_axis(pobj.axis_name, axis_data)

            if found_data:
                foundation_objects, found_filtered = load_from_json(FoundationObject, found_data, mapping, axis_data=axis_data)
                for i, fobj in enumerate(foundation_objects or []):
                    if hasattr(fobj, "axis_variables") and isinstance(fobj.axis_variables, list):
                        fobj.set_axis_variables(found_filtered[i].get("AxisVariables", []), mapping.get(AxisVariable, {}))
                    if hasattr(fobj, "axis_name") and axis_data:
                        fobj.set_axis(fobj.axis_name, axis_data)

        return axis_data, cross_sections, deck_objects, deck_filtered, pier_objects, foundation_objects

    # -------- assemble vis -----------------------------------------------------

    def build_vis(
        self,
        *,
        attach_section_json: bool = True,
    ) -> Tuple[List[dict], List[object]]:
        """
        Returns (vis_data_all, vis_objs_all).
        Will attach parsed `json_data` to each vis (cached by file) if enabled.
        """
        axis_data, cross_sections, deck_objects, deck_filtered, pier_objects, foundation_objects = \
            self.build_domain_objects()

        # index CrossSection by NCS for fast json file lookup
        cs_by_ncs: Dict[str, CrossSection] = {}
        for cs in cross_sections:
            ncs = getattr(cs, "ncs", None)
            if ncs is not None:
                cs_by_ncs[str(ncs)] = cs

        vis_data_all: List[dict] = []
        vis_objs_all: List[object] = []
        json_cache: Dict[str, Optional[dict]] = {}

        def _attach_json_data(vis_row: dict):
            jf = vis_row.get("json_file")
            if not (attach_section_json and jf):
                vis_row["json_data"] = None
                return
            jkey = os.path.abspath(os.path.join(self.branch_folder, jf)) if not os.path.isabs(jf) else jf
            if jkey not in json_cache:
                try:
                    json_cache[jkey] = _read_json(jkey)
                except Exception as e:
                    print(f"[SpotLoader] warn: cannot read {jkey}: {e}")
                    json_cache[jkey] = None
            vis_row["json_data"] = json_cache[jkey]

        # ---- Decks
        for i, obj in enumerate(deck_objects):
            try:
                vis = obj.get_input_for_visualisation(
                    axis_data=axis_data,
                    cross_section_objects=cross_sections,
                    json_file_path=None,  # let the object find via NCS->CrossSection.JSON_name
                )
                _attach_json_data(vis)
                vis_data_all.append(vis)
                vis_objs_all.append(obj)
            except Exception as e:
                print(f"[SpotLoader] skip Deck {getattr(obj,'name','?')}: {e}")

        # ---- Optional: Piers / Foundations
        if self.cond_pier_foundation and _has_pier_found:
            for coll in (pier_objects or []), (foundation_objects or []):
                for obj in coll:
                    try:
                        vis = obj.get_input_for_visualisation(
                            axis_data=axis_data,
                            cross_section_objects=cross_sections,
                            json_file_path=None,
                        )
                        _attach_json_data(vis)
                        vis_data_all.append(vis)
                        vis_objs_all.append(obj)
                    except Exception as e:
                        print(f"[SpotLoader] skip {obj.__class__.__name__} {getattr(obj,'name','?')}: {e}")

        return vis_data_all, vis_objs_all
