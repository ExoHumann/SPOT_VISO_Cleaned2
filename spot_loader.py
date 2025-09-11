# spot_loader.py
from __future__ import annotations
import logging
import os
import inspect
import importlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, TYPE_CHECKING

# --- IO helpers (your SPOT_Filters) -----------------------------------------
try:
    # package-relative import
    from .SPOT_Filters import get_subfolders, get_json_files, load_json_objects
except Exception:
    # flat import fallback
    from SPOT_Filters import get_subfolders, get_json_files, load_json_objects

# --- Mapping + Context -------------------------------------------------------
try:
    from models.mapping import mapping as MAP
except Exception:
    from mapping import mapping as MAP  # fallback if flat

try:
    from models.viso_context import VisoContext
except Exception:
    from viso_context import VisoContext  # fallback if flat

# --- TYPE CHECKING ONLY (no runtime deps required) ---------------------------
if TYPE_CHECKING:
    from models.axis import Axis
    from models.cross_section import CrossSection
    from models.main_station import MainStation
    from models.deck_object import DeckObject
    from models.pier_object import PierObject
    from models.foundation_object import FoundationObject
    from models.bearing_articulation import BearingArticulation
    from models.secondary_object import SecondaryObject
    #from models.materials import Materials
    #from models.global_variable import GlobalVariable
    from models.axis_variable import AxisVariable

import json, os
from collections import defaultdict

# --------------------------------------------------------------------------- #
# SpotLoader
# --------------------------------------------------------------------------- #
@dataclass
class SpotLoader:
    master_folder: str
    branch: Optional[str] = None
    verbose: bool = False

    # ------------------------------- internals ------------------------------ #
    def _dbg(self, *parts: object) -> None:
        if self.verbose:
            print(*("[SpotLoader]", *parts))

    @property
    def _logger(self) -> logging.Logger:
        # Lazy to avoid configuring global logging for callers
        return logging.getLogger(__name__)

    # raw + grouped
    _raw_rows: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _by_class: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict, init=False)

    # context
    ctx: Optional[VisoContext] = field(default=None, init=False)

    # typed collections (populated by build_all_with_context)
    _deck_objects: List[Any] = field(default_factory=list, init=False)
    _pier_objects: List[Any] = field(default_factory=list, init=False)
    _foundation_objects: List[Any] = field(default_factory=list, init=False)
    _bearing_objects: List[Any] = field(default_factory=list, init=False)
    _secondary_objects: List[Any] = field(default_factory=list, init=False)
    _materials: List[Any] = field(default_factory=list, init=False)
    _globals: List[Any] = field(default_factory=list, init=False)

    # convenience for UI / vis
    vis_objs: List[Any] = field(default_factory=list, init=False)

    # cache for loaded section library files
    _section_payload_cache: Dict[str, dict] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
    # ...your existing init...
        self.repo_root = os.path.abspath(os.path.join(self.master_folder, os.pardir))
        self.master_section_dir = os.path.join(self.repo_root, "MASTER_SECTION")

    def _resolve_candidate_path(self, candidate: str) -> Optional[str]:
        """
        Resolve to .../MASTER_SECTION/<candidate>.
        Accepts either a basename ('PierMaster.json') or a path ('subdir/xyz.json').
        """
        if not candidate:
            return None

        # Normalize: keep both original and basename
        names = [candidate.replace("\\", os.sep).replace("/", os.sep),
                os.path.basename(candidate)]

        for name in names:
            p = os.path.join(self.master_section_dir, name)
            if os.path.exists(p):
                return p
        return None

    def _load_section_payload(self, candidate: str) -> Optional[Dict]:
        """
        Load the geometry JSON once; cached on path.
        """
        path = self._resolve_candidate_path(candidate)
        if not path:
            self._logger.debug("[SpotLoader] geometry not found in MASTER_SECTION: %r", candidate)
            return None

        cache = self._section_payload_cache  # class-level cache
        if path in cache:
            return cache[path]

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cache[path] = data
            self._logger.debug("[SpotLoader] loaded geometry JSON: %s  Points=%d Loops=%d",
                        os.path.basename(path),
                        len(data.get("Points") or []),
                        len(data.get("Loops") or []))
            return data
        except Exception as e:
            self._logger.warning("[SpotLoader] failed reading %s: %s", path, e)
            return None


    def _pick_default_library_for_consumers(self, consumers: set[str]) -> list[str]:
        """
        Order of fallbacks based on who uses the CrossSection.
        """
        out: list[str] = []
        if "DeckObject" in consumers:
            out.append("MASTER_DeckMain-1Gird-Slab.json")
        if "PierObject" in consumers:
            out.append("PierMaster.json")
        if "FoundationObject" in consumers:
            out.append("MASTER_Foundation.json")
        # last resort: generic shapes
        out.append("SectionData.json")
        # dedupe preserving order
        seen = set(); res=[]
        for x in out:
            if x not in seen:
                seen.add(x); res.append(x)
        return res

    def _enrich_cross_sections_with_geometry(self, ctx) -> None:
        """
        Attach Loops/Points/Variables from external section JSONs to every CrossSection.
        Priority:
        1) explicit JSON_name on the cross section row (first that exists),
        2) defaults inferred from which object types reference this NCS,
        3) SectionData.json as a final fallback.
        """
        # Build: NCS -> set of consumer class names
        consumers = defaultdict(set)

        def _acc_ncs(obj, *attrs):
            for a in attrs:
                v = getattr(obj, a, None)
                if v is None:
                    continue
                if isinstance(v, (list, tuple)):
                    for n in v:
                        try: consumers[int(n)].add(obj.__class__.__name__)
                        except Exception: pass
                else:
                    try: consumers[int(v)].add(obj.__class__.__name__)
                    except Exception: pass

        # Deck
        for o in getattr(ctx, "deck_objects", []):
            _acc_ncs(o, "cross_section_ncs")

        # Pier (top/bot/internal)
        for o in getattr(ctx, "pier_objects", []):
            _acc_ncs(o, "top_cross_section_ncs", "bot_cross_section_ncs", "internal_cross_section_ncs")

        # Foundation
        for o in getattr(ctx, "foundation_objects", []):
            _acc_ncs(o, "cross_section_ncs", "cross_section_ncs2", "internal_cross_section_ncs")

        # Secondary (beg/end)
        for o in getattr(ctx, "secondary_objects", []):
            _acc_ncs(o, "beg_ncs", "end_ncs")

        # Enrich every cross section
        for ncs, cs in (getattr(ctx, "crosssec_by_ncs", {}) or {}).items():
            # 1) explicit hints (JSON_name can be str or list)
            cand: list[str] = []
            jn = getattr(cs, "json_name", None)
            if isinstance(jn, str) and jn.strip():
                cand.append(jn)
            elif isinstance(jn, (list, tuple)):
                cand += [s for s in jn if isinstance(s, str) and s.strip()]

            # 2) fallbacks from consumers
            cand += self._pick_default_library_for_consumers(consumers.get(ncs, set()))

            payload = None
            used = None
            for c in cand:
                payload = self._load_section_payload(c)
                if payload:
                    used = c
                    break

            if not payload:
                self._logger.debug("[SpotLoader] no geometry for CrossSection ncs=%s name=%r  candidates=%s",
                            ncs, getattr(cs, "name", ""), cand)
                continue

            # Copy over geometry
            if not getattr(cs, "points", None):
                cs.points = payload.get("Points") or []
            cs.loops = payload.get("Loops") or getattr(cs, "loops", []) or []
            if not getattr(cs, "variables", None):
                cs.variables = payload.get("Variables") or {}

            # keep the whole blob if you want to debug later
            cs.json_data = payload

            self._logger.debug("[SpotLoader] wired geometry for ncs=%s name=%r via %s  -> points=%d loops=%d",
                        ncs, getattr(cs, "name", ""),
                        used,
                        len(cs.points or []),
                        len(cs.loops or []))


    # ------------------------------- public API ------------------------------ #
    def set_branch(self, branch: str) -> "SpotLoader":
        self.branch = branch
        return self

    def load_raw(self) -> "SpotLoader":
        """
        Scan the target branch folder, load all JSON (files starting with "_"),
        flatten to a single list of rows.
        """
        if not os.path.isdir(self.master_folder):
            raise ValueError(f"Invalid master folder: {self.master_folder}")
        
        sub = get_subfolders(self.master_folder)
        if not sub:
            raise ValueError(f"No subfolders found in {self.master_folder}")

        # default to MAIN if present, otherwise first folder
        selected = self.branch or ("MAIN" if "MAIN" in sub else sub[0])
        folder_path = os.path.join(self.master_folder, selected)
        if not os.path.isdir(folder_path):
            raise ValueError(f"Selected branch not found: {folder_path}")

        self._dbg("Master folder:", self.master_folder)
        self._dbg("Subfolders found:", sub)
        self._dbg("Selected branch:", selected, "->", folder_path)

        json_files = get_json_files(folder_path)
        rows = load_json_objects(json_files)  # list[dict] (flattened)

        self._raw_rows = rows or []
        self._dbg(f"Loaded {len(self._raw_rows)} rows from {len(json_files)} files.")
        if self.verbose and json_files:
            self._dbg("Files:", json_files[:5], ("... ({} more)".format(max(0, len(json_files)-5)) if len(json_files) > 5 else ""))

        return self  # <-- important for method chaining

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

        by: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            by.setdefault(r["Class"], []).append(r)

        self._by_class = by
        if self.verbose:
            summary = {k: len(v) for k, v in sorted(by.items())}
            self._dbg("Grouped by Class:", summary)
            self._dbg("Total rows (excluding ClassInfo/CDDMapping):", len(rows))
            self._dbg("Classes found:", list(summary.keys()))
            # Deep dive (very verbose)
            for class_name, class_rows in sorted(by.items()):
                self._dbg(f"  {class_name}: {len(class_rows)} rows")
                for i, row in enumerate(class_rows, 1):
                    self._dbg(f"    Row {i}: keys={list(row.keys())}")

        return self

    def build_all_with_context(self, *, verbose: bool = True) -> "SpotLoader":
        """
        Materialize typed objects for all known classes and build a VisoContext
        that indexes + wires them.
        """
        if not self._by_class:
            self.group_by_class()

        axis_rows = self._by_class.get("Axis", []) or []
        cs_rows   = self._by_class.get("CrossSection", []) or []
        ms_rows   = self._by_class.get("MainStation", []) or []

        # --- typed core: CrossSection & MainStation
        CrossSection = self._maybe_cls("cross_section", "CrossSection")
        MainStation  = self._maybe_cls("main_station", "MainStation")
        if self.verbose:
            self._dbg("Resolved classes:",
                      "CrossSection" if CrossSection else "CrossSection[missing]",
                      "MainStation" if MainStation else "MainStation[missing]")

        cross_sections, _ = self._load_typed(CrossSection, cs_rows, axis_rows=axis_rows)
        mainstations,  _  = self._load_typed(MainStation,  ms_rows, axis_rows=axis_rows)

        # --- context (axes come from raw 'Axis' rows)
        self.ctx = VisoContext.from_json(axis_rows, cross_sections, mainstations, mapping_cfg=MAP, verbose=False)
        if self.verbose:
            self._dbg("Context created:",
                      f"axes={len(self.ctx.axes_by_name)}",
                      f"cross_sections={len(self.ctx.crosssec_by_ncs)}|{len(self.ctx.crosssec_by_name)}",
                      f"mainstations={len(self.ctx.mainstations_by_name)}")

        # --- remaining families
        DeckObject          = self._maybe_cls("deck_object", "DeckObject")
        PierObject          = self._maybe_cls("pier_object", "PierObject")
        FoundationObject    = self._maybe_cls("foundation_object", "FoundationObject")
        BearingArticulation = self._maybe_cls("bearing_articulation", "BearingArticulation")
        SecondaryObject     = self._maybe_cls("secondary_object", "SecondaryObject")
        Materials           = self._maybe_cls("materials", "Materials")
        GlobalVariable      = self._maybe_cls("global_variable", "GlobalVariable")
        AxisVariable        = self._maybe_cls("axis_variable", "AxisVariable")
        if self.verbose:
            self._dbg("Resolved optional classes:",
                      "Deck" if DeckObject else "Deck[missing]",
                      "Pier" if PierObject else "Pier[missing]",
                      "Fnd" if FoundationObject else "Fnd[missing]",
                      "Bearing" if BearingArticulation else "Bearing[missing]",
                      "Secondary" if SecondaryObject else "Secondary[missing]",
                      "Materials" if Materials else "Materials[missing]",
                      "Globals" if GlobalVariable else "Globals[missing]",
                      "AxisVariable" if AxisVariable else "AxisVariable[missing]")
            
        self._deck_objects, deck_raw = self._load_typed(DeckObject, self._by_class.get("DeckObject", []) or [], axis_rows=axis_rows)
        self._pier_objects, pier_raw = self._load_typed(PierObject, self._by_class.get("PierObject", []) or [], axis_rows=axis_rows)
        self._foundation_objects, fnd_raw = self._load_typed(FoundationObject, self._by_class.get("FoundationObject", []) or [], axis_rows=axis_rows)
        self._bearing_objects, _ = self._load_typed(BearingArticulation, self._by_class.get("BearingArticulation", []) or [], axis_rows=axis_rows)
        self._secondary_objects, _ = self._load_typed(SecondaryObject, self._by_class.get("SecondaryObject", []) or [], axis_rows=axis_rows)
        self._materials, _ = self._load_typed(Materials, self._by_class.get("Materials", []) or [], axis_rows=axis_rows)
        self._globals, _ = self._load_typed(GlobalVariable, self._by_class.get("GlobalVariable", []) or [], axis_rows=axis_rows)

        # Keep raw AxisVariables list on objects that expose .axis_variables (VisoContext will build objects)
        self._maybe_set_raw_axisvars(self._deck_objects, deck_raw)
        self._maybe_set_raw_axisvars(self._pier_objects, pier_raw)
        self._maybe_set_raw_axisvars(self._foundation_objects, fnd_raw)

        # --- wire + register everything
        # AxisVariable mapping (support both class-key and string-key)
        axis_var_map: Dict[str, Any] = {}
        try:
            if AxisVariable and (AxisVariable in MAP):
                axis_var_map = MAP.get(AxisVariable, {}) or {}
            else:
                axis_var_map = MAP.get("AxisVariable", {}) or {}
        except Exception:
            axis_var_map = MAP.get("AxisVariable", {}) or {}
        if self.verbose:
            self._dbg("AxisVariable mapping keys:", list(axis_var_map.keys()) if isinstance(axis_var_map, dict) else type(axis_var_map))

        # spot_loader.py – after you’ve built ctx and registered objects
        self._enrich_cross_sections_with_geometry(self.ctx)

        self.ctx.add_objects(
            self._deck_objects, self._pier_objects,
            axis_var_map=axis_var_map,
        )
        self.ctx.add_objects(self._foundation_objects, self._bearing_objects, self._secondary_objects, self._materials, self._globals, axis_var_map=axis_var_map)

        # --- convenience array useful for visualisation/UIs
        self.vis_objs = list(self._deck_objects) + list(self._pier_objects) + list(self._foundation_objects) + list(self._bearing_objects) + list(self._secondary_objects)

        if verbose or self.verbose:
            def _n(x): return len(x or [])
            self._dbg("Objects registered:",
                f"Deck={_n(self._deck_objects)} Pier={_n(self._pier_objects)}",
                f"Fnd={_n(self._foundation_objects)} Bearing={_n(self._bearing_objects)}",
                f"Sec={_n(self._secondary_objects)} Mat={_n(self._materials)} GVar={_n(self._globals)}")

        return self

    # ------------------------------- helpers -------------------------------- #
    def _maybe_cls(self, module_name: str, cls_name: str):
        """
        Resolve a class with several import strategies in priority order:
          1) models.<module_name>
          2) <module_name> (flat)
          3) .<module_name> relative to this package (if available)
          4) models package re-exports (models.__getattr__)
        """
        errs: List[str] = []

        # 1) Preferred: fully-qualified under 'models'
        try:
            mod = importlib.import_module(f"models.{module_name}")
            return getattr(mod, cls_name)
        except Exception as e:
            errs.append(f"models.{module_name}: {e!r}")

        # 2) Flat module name (legacy)
        try:
            mod = importlib.import_module(module_name)
            return getattr(mod, cls_name)
        except Exception as e:
            errs.append(f"{module_name}: {e!r}")

        # 3) Relative to this package if __package__ is defined
        try:
            if __package__:
                mod = importlib.import_module(f".{module_name}", package=__package__)
                return getattr(mod, cls_name)
        except Exception as e:
            errs.append(f".{module_name}@{__package__}: {e!r}")

        # 4) As a last resort, let the 'models' package re-export provide it
        try:
            pkg = importlib.import_module("models")
            return getattr(pkg, cls_name)
        except Exception as e:
            errs.append(f"models re-export {cls_name}: {e!r}")

        if self.verbose:
            self._dbg(f"Unable to import {cls_name} from {module_name}:", " / ".join(errs))
        return None

    def _maybe_set_raw_axisvars(self, objs: List[Any], raw_rows: List[Dict[str, Any]]) -> None:
        if not objs or not raw_rows:
            return
        for inst, row in zip(objs, raw_rows):
            if hasattr(inst, "axis_variables"):
                inst.axis_variables = (row or {}).get("AxisVariables", []) or getattr(inst, "axis_variables", [])

    # -- typed materialization ------------------------------------------------ #
    def _load_typed(
        self,
        cls: Optional[Type],
        rows: List[Dict[str, Any]],
        *,
        axis_rows: List[Dict[str, Any]],
    ) -> Tuple[List[Any], List[Dict[str, Any]]]:
        """
        Best-effort materialization of rows into typed objects.

        Strategy (per row):
          1) call a class factory if available:
               - from_dict(row, mapping_cfg=MAP, axis_data=axis_rows)
               - from_json(row, mapping_cfg=MAP, axis_data=axis_rows)
               - from_raw(row, mapping_cfg=MAP, axis_data=axis_rows)
          2) else build kwargs using mapping (MAP[cls] or MAP[cls.__name__]) and call cls(**kwargs)
          3) on failure, return None and skip the row (but keep raw_rows for debugging)
        """
        if not cls:
            return [], rows

        factory_names = ("from_dict", "from_json", "from_raw")
        mapping_for_cls = MAP.get(cls, None) or MAP.get(getattr(cls, "__name__", ""), {}) or {}

        out: List[Any] = []
        kept_raw: List[Dict[str, Any]] = []

        for idx, r in enumerate(rows or []):
            inst = None
            kwargs: Dict[str, Any] = {}

            # 1) Try factories with several signatures
            for fname in factory_names:
                fn = getattr(cls, fname, None)
                if not callable(fn):
                    continue
                try:
                    # Prefer keyword style — most robust to user-defined factories
                    inst = fn(r, mapping_cfg=MAP, axis_data=axis_rows)
                    if self.verbose:
                        self._dbg(f"{cls.__name__}[{idx}] factory used:", fname)
                except TypeError:
                    # Try a few fallbacks
                    try:
                        inst = fn(r, MAP, axis_rows)
                    except TypeError:
                        try:
                            inst = fn(r, MAP)
                        except TypeError:
                            try:
                                inst = fn(r)
                            except Exception:
                                inst = None
                except Exception as e:
                    if self.verbose:
                        self._dbg(f"{cls.__name__}[{idx}] factory {fname} raised:", repr(e))
                    inst = None
                if inst is not None:
                    break

            # 2) No factory worked — try dataclass-style ctor with mapped kwargs
            if inst is None:
                try:
                    kwargs = self._build_kwargs_from_mapping(r, cls, mapping_for_cls)
                    kwargs = _filter_kwargs_for_ctor(
                        cls, kwargs,
                        drop_none=True,
                        verbose=self.verbose,
                        dbg=self._dbg if self.verbose else None,
                        tag=f"[{idx}]"
                    )
                    inst = cls(**kwargs)
                except Exception as e:
                    if self.verbose:
                        cname = getattr(cls, "__name__", str(cls))
                        self._dbg(f"{cname}[{idx}] ctor failed:", repr(e))
                        self._dbg("  kwargs=", kwargs)
                        self._dbg("  row_keys=", list(r.keys()))
                    inst = None

            if inst is not None:
                out.append(inst)
                kept_raw.append(r)
            else:
                if self.verbose:
                    cname = getattr(cls, "__name__", str(cls))
                    self._dbg(f"Warning: failed to instantiate {cname} from row idx={idx}")
                # Very specific legacy fallback: only try DeckObject when we are *already* loading DeckObject
                try:
                    if getattr(cls, "__name__", "") == "DeckObject" and kwargs:
                        if self.verbose:
                            self._dbg(f"DeckObject[{idx}] trying legacy ctor fallback with kwargs keys:", list(kwargs.keys()))
                        obj = cls(**kwargs)  # same as instantiation above; keep behavior symmetrical
                        out.append(obj)
                        kept_raw.append(r)
                except Exception as e:
                    # Use logger to avoid crashing; this was previously using an invalid 'log.error'
                    self._logger.error("Legacy DeckObject fallback failed for row %s: %s", idx, e)

        return out, kept_raw

    def _build_kwargs_from_mapping(self, row: Dict[str, Any], cls: Type, mp: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a raw row -> kwargs for a dataclass constructor using the mapping.

        Accepts either:
        - python_field -> json_key (what the loader originally expects), or
        - json_key -> python_field (your current mapping style).

        Also supports multiple source keys for one field (list/tuple), taking the first present.
        """
        kwargs: Dict[str, Any] = {}

        # If no mapping, best-effort 1:1 (your original fallback)
        if not mp:
            dataclass_fields = getattr(cls, "__dataclass_fields__", {}) or {}
            for f_name in dataclass_fields.keys():
                if f_name in row:
                    kwargs[f_name] = row[f_name]
                else:
                    cap = f_name[:1].upper() + f_name[1:]
                    if cap in row:
                        kwargs[f_name] = row[cap]
            return kwargs

        # Detect orientation
        dc_fields = set((getattr(cls, "__dataclass_fields__", {}) or {}).keys())
        # how many mapping KEYS look like python fields?
        hits_keys_as_py = sum(1 for k in mp.keys() if k in dc_fields)
        # how many mapping VALUES look like python fields?
        hits_vals_as_py = sum(1 for v in mp.values() if isinstance(v, str) and v in dc_fields)
        mapping_is_json_to_py = hits_keys_as_py == 0 and hits_vals_as_py > 0
        if self.verbose:
            orient = "json->py" if mapping_is_json_to_py else "py->json"
            self._dbg(f"{getattr(cls, '__name__', cls)} mapping orientation:", orient)

        # Normalize to python_field -> json_key(s)
        if mapping_is_json_to_py:
            # Merge duplicates: multiple json keys may map to the same python field
            norm: Dict[str, Any] = {}
            for json_key, py_field in mp.items():
                if py_field in norm:
                    prev = norm[py_field]
                    if isinstance(prev, (list, tuple)):
                        norm[py_field] = list(prev) + [json_key]
                    else:
                        norm[py_field] = [prev, json_key]
                else:
                    norm[py_field] = json_key
            mp_norm = norm
        else:
            mp_norm = mp  # already python_field -> json_key

        # Build kwargs
        for py_field, js_spec in mp_norm.items():
            if callable(js_spec):
                val = js_spec(row)
            else:
                # js_spec can be a single key or a list/tuple of alternatives
                if isinstance(js_spec, (list, tuple)):
                    val = None
                    for k in js_spec:
                        if k in row:
                            val = row.get(k)
                            if val is not None:
                                break
                else:
                    val = row.get(js_spec)
            kwargs[py_field] = val
            
        if self.verbose:
            missing = [f for f, v in kwargs.items() if v is None]
            if missing:
                self._dbg(f"{getattr(cls, '__name__', cls)} missing fields from row:", missing)

        return kwargs

    # ------------------------------- convenience ----------------------------- #
    @property
    def deck_objects(self) -> List[Any]:
        return list(self._deck_objects)

    @property
    def pier_objects(self) -> List[Any]:
        return list(self._pier_objects)

    @property
    def foundation_objects(self) -> List[Any]:
        return list(self._foundation_objects)

    @property
    def bearing_objects(self) -> List[Any]:
        return list(self._bearing_objects)

    @property
    def secondary_objects(self) -> List[Any]:
        return list(self._secondary_objects)

    @property
    def materials(self) -> List[Any]:
        return list(self._materials)

    @property
    def global_variables(self) -> List[Any]:
        return list(self._globals)


# imports needed at top of the file if not present
import inspect
from inspect import Parameter

def _accepted_ctor_args(cls):
    """
    Return the set of accepted keyword names for cls's constructor and whether it
    accepts **kwargs. Falls back to dataclass fields.
    """
    try:
        sig = inspect.signature(cls)
        accepts_var_kw = any(p.kind == Parameter.VAR_KEYWORD for p in sig.parameters.values())
        names = {p.name for p in sig.parameters.values()
                 if p.name != "self" and p.kind in (Parameter.POSITIONAL_OR_KEYWORD,
                                                    Parameter.KEYWORD_ONLY)}
        return names, accepts_var_kw
    except (TypeError, ValueError):
        # Dataclass or simple class without inspectable signature
        dc_fields = getattr(cls, "__dataclass_fields__", {}) or {}
        return set(dc_fields.keys()), False

def _filter_kwargs_for_ctor(cls, kwargs, *, drop_none=True, verbose=False, dbg=None, tag=""):
    """
    Filter kwargs to only those that the class constructor will accept.
    Optionally drop keys whose value is None.
    """
    accepted, has_var_kw = _accepted_ctor_args(cls)
    if drop_none:
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if has_var_kw:
        # ctor has **kwargs; keep everything (after None drop)
        kept = kwargs
        dropped = []
    else:
        kept = {k: v for k, v in kwargs.items() if k in accepted}
        dropped = [k for k in kwargs.keys() if k not in kept]
    if verbose and dropped and dbg:
        cname = getattr(cls, "__name__", str(cls))
        dbg(f"{cname}{tag} dropped unknown kwargs:", dropped)
    return kept



