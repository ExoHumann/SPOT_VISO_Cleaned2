# models/__init__.py
"""
Public API for the models package.

External code (e.g. main.py, spot_loader.py) can do:
    from models import Axis, CrossSection, BaseObject, mapping, load_from_json

Inside the package modules themselves, prefer relative imports, e.g.:
    from .base import BaseObject
to avoid circular imports during initialization.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import importlib

__all__ = [
    # core dataclasses / objects
    "Axis", "AxisVariable", "CrossSection", "DeckObject",
    "PierObject", "FoundationObject", "SecondaryObject",
    "MainStation", "VisoContext",
    # shared base + helpers
    "BaseObject",
    "create_input_for_visualisation", "load_from_json", "build_viso_object",
    "build_axis_index", "build_cross_section_index", "get_axis_cached",
    "create_axis_variables", "from_dict",
    # config
    "mapping",
]

# Map exported names -> submodule that defines them
_EXPORT_MAP = {
    # modules
    "Axis": "models.axis",
    "AxisVariable": "models.axis_variable",
    "CrossSection": "models.cross_section",
    "DeckObject": "models.deck_object",
    "PierObject": "models.pier_object",
    "FoundationObject": "models.foundation_object",
    "SecondaryObject": "models.secondary_object",
    "MainStation": "models.main_station",
    "VisoContext": "models.viso_context",

    # base + helpers
    "BaseObject": "models.base",
    "create_input_for_visualisation": "models.base",
    "load_from_json": "models.base",
    "build_viso_object": "models.base",
    "build_axis_index": "models.base",
    "build_cross_section_index": "models.base",
    "get_axis_cached": "models.base",
    "create_axis_variables": "models.base",
    "from_dict": "models.base",

    # config
    "mapping": "models.mapping",
}

def __getattr__(name: str):
    """Lazy attribute loader to avoid import-time cycles."""
    mod_name = _EXPORT_MAP.get(name)
    if not mod_name:
        raise AttributeError(f"module 'models' has no attribute {name!r}")
    mod = importlib.import_module(mod_name)
    try:
        return getattr(mod, name)
    except AttributeError:
        # Support module-level variables like `mapping`
        return getattr(mod, name)  # will still raise if truly missing

if TYPE_CHECKING:
    # For static type checkers / IDEs, do eager imports
    from .axis import (Axis, embed_points_to_global_mm, resample_axis_by_spacing)
    from .axis_variable import AxisVariable
    from .cross_section import CrossSection
    from .deck_object import DeckObject
    from .pier_object import PierObject
    from .foundation_object import FoundationObject
    from .secondary_object import SecondaryObject
    from .main_station import MainStation
    from .viso_context import (
        VisoContext, 
        _wire_objects)

    from .base import (
        BaseObject,
        create_input_for_visualisation, load_from_json, build_viso_object,
        build_axis_index, build_cross_section_index, get_axis_cached,
        create_axis_variables, from_dict,
    )
    from .mapping import mapping
