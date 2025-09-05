from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Any, Iterable
from collections import defaultdict
from .mapping import mapping

if TYPE_CHECKING:
    from .axis import Axis
    from .cross_section import CrossSection
    from .main_station import MainStation

def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()

@dataclass(slots=True)
class VisoContext:
    # Core components
    axes_by_name: Dict[str, "Axis"] = field(default_factory=dict)
    crosssec_by_ncs: Dict[int, "CrossSection"] = field(default_factory=dict)
    mainstations_by_name: Dict[str, "MainStation"] = field(default_factory=dict)

    # Registries
    objects_by_class: Dict[str, List[Any]] = field(default_factory=lambda: defaultdict(list))
    objects_by_axis: Dict[str, List[Any]] = field(default_factory=lambda: defaultdict(list))
    objects_by_name: Dict[str, Any] = field(default_factory=dict)
    objects_by_id: Dict[Any, Any] = field(default_factory=dict)

    # NEW: geometry engine
    engine: Any = field(default=None, repr=False)

    @classmethod
    def from_json(
        cls,
        axis_data: List[dict],
        cross_sections: List["CrossSection"],
        mainstations: Optional[List["MainStation"]] = None,
        *,
        mapping_cfg=None,
    ) -> "VisoContext":
        from .axis import Axis  # avoid cycles
        m = mapping_cfg or mapping
        amap = (m.get("Axis", None) or m.get(Axis, {})) or {}
        name_k = amap.get("name", "Name")
        sta_k  = amap.get("stations", "StaionValue")
        x_k    = amap.get("x_coords", "CurvCoorX")
        y_k    = amap.get("y_coords", "CurvCoorY")
        z_k    = amap.get("z_coords", "CurvCoorZ")
        cls_k  = amap.get("class", "Class")

        axes: Dict[str, Axis] = {}
        for row in (axis_data or []):
            if str(row.get(cls_k, "Axis")) == "Axis":
                nm = str(row.get(name_k, "")).strip()
                if nm:
                    axes[_norm(nm)] = Axis(
                        stations=[float(s) for s in row.get(sta_k, [])],
                        x_coords=[float(v) for v in row.get(x_k, [])],
                        y_coords=[float(v) for v in row.get(y_k, [])],
                        z_coords=[float(v) for v in row.get(z_k, [])],
                    )

        cs_by_ncs = {int(cs.ncs): cs for cs in (cross_sections or [])}
        ms_by_name = {_norm(getattr(ms, "name", "")): ms for ms in (mainstations or []) if getattr(ms, "name", "").strip()}

        ctx = cls(
            axes_by_name=axes,
            crosssec_by_ncs=cs_by_ncs,
            mainstations_by_name=ms_by_name,
        )
        if ctx.engine is None:
            from section_engine import SectionGeometryEngine
            ctx.engine = SectionGeometryEngine()
        return ctx

    # Convenience entry point so callers never import the engine:
    def compute_section(
        self,
        *,
        section_json: dict,
        axis: "Axis",
        axis_var_results: List[dict],
        stations_m: List[float],
        twist_deg: float = 0.0,
        negate_x: bool = True,
    ):
        return self.engine.compute(
            section_json=section_json,
            axis=axis,
            axis_var_results=axis_var_results,
            stations_m=stations_m,
            twist_deg=twist_deg,
            negate_x=negate_x,
        )

    @classmethod
    def from_collections(
        cls,
        *,
        axis_rows: List[dict],
        cross_sections: List["CrossSection"],
        mainstations: Optional[List["MainStation"]] = None,
        objects: Iterable[Any] = (),
        mapping_cfg=None,
    ) -> "VisoContext":
        """Optional convenience: build context and immediately register objects."""
        ctx = cls.from_json(axis_rows, cross_sections, mainstations, mapping_cfg=mapping_cfg)
        ctx.register_objects(objects)
        return ctx

    # ------------- Registration -------------

    def register_objects(self, objs: Iterable[Any]) -> None:
        for o in (objs or []):
            clsname = o.__class__.__name__
            self.objects_by_class[clsname].append(o)

            nm = getattr(o, "name", None)
            if nm:
                self.objects_by_name[_norm(nm)] = o

            oid = getattr(o, "no", None) or getattr(o, "id", None)
            if oid is not None:
                self.objects_by_id[oid] = o

            ax = getattr(o, "axis_name", None) or getattr(o, "object_axis_name", None)
            if ax:
                self.objects_by_axis[_norm(ax)].append(o)

    def wire_objects(self, objs: list[Any], *, axis_var_map: dict) -> None:
        for o in (objs or []):
            ax_name = getattr(o, "axis_name", None) or getattr(o, "object_axis_name", None)
            if ax_name:
                ax = self.get_axis(ax_name)
                if ax is not None and hasattr(o, "axis_obj"):
                    o.axis_obj = ax
            if hasattr(o, "axis_variables") and hasattr(o, "set_axis_variables"):
                o.set_axis_variables(o.axis_variables, axis_var_map)

    def add_objects(self, *objs_lists: list[Any], axis_var_map: dict) -> None:
        for lst in objs_lists:
            self.wire_objects(lst, axis_var_map=axis_var_map)
            self.register_objects(lst)
    # ------------- Lookups -------------

    # axes
    def get_axis(self, name: str) -> Optional["Axis"]:
        return self.axes_by_name.get(_norm(name))

    # cross sections
    def get_cross_section(self, ncs: int) -> Optional["CrossSection"]:
        try:
            return self.crosssec_by_ncs.get(int(ncs))
        except Exception:
            return None

    # mainstations
    def get_mainstation(self, name: str) -> Optional["MainStation"]:
        return self.mainstations_by_name.get(_norm(name))

    # generic object queries
    def all_objects(self, clsname: Optional[str] = None) -> List[Any]:
        if not clsname:
            out: List[Any] = []
            for arr in self.objects_by_class.values():
                out.extend(arr)
            return out
        return list(self.objects_by_class.get(clsname, []))

    def objects_for_axis(self, axis_name: str, clsname: Optional[str] = None) -> List[Any]:
        items = list(self.objects_by_axis.get(_norm(axis_name), []))
        return [o for o in items if (clsname is None or o.__class__.__name__ == clsname)]

    def get_object_by_name(self, name: str) -> Optional[Any]:
        return self.objects_by_name.get(_norm(name))

    def get_object_by_id(self, oid: Any) -> Optional[Any]:
        return self.objects_by_id.get(oid)
