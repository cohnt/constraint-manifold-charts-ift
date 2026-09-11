# SPDX-License-Identifier: MIT
"""Material parsing for meshcat objects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class MaterialType(Enum):
    """Types of materials supported by meshcat."""

    MESH_STANDARD = "MeshStandardMaterial"
    MESH_PHONG = "MeshPhongMaterial"
    MESH_BASIC = "MeshBasicMaterial"
    MESH_LAMBERT = "MeshLambertMaterial"
    LINE_BASIC = "LineBasicMaterial"
    POINTS = "PointsMaterial"


@dataclass
class Color:
    """RGB color with values 0-1."""

    r: float
    g: float
    b: float

    @classmethod
    def from_int(cls, color_int: int) -> Color:
        """Create from integer color (0xRRGGBB)."""
        r = ((color_int >> 16) & 0xFF) / 255.0
        g = ((color_int >> 8) & 0xFF) / 255.0
        b = (color_int & 0xFF) / 255.0
        return cls(r=r, g=g, b=b)

    @classmethod
    def from_hex(cls, hex_str: str) -> Color:
        """Create from hex string (#RRGGBB or 0xRRGGBB)."""
        hex_str = hex_str.lstrip("#").lstrip("0x")
        color_int = int(hex_str, 16)
        return cls.from_int(color_int)

    def to_tuple(self) -> tuple[float, float, float]:
        """Convert to RGB tuple."""
        return (self.r, self.g, self.b)

    def to_tuple_alpha(self, alpha: float = 1.0) -> tuple[float, float, float, float]:
        """Convert to RGBA tuple."""
        return (self.r, self.g, self.b, alpha)


@dataclass
class ParsedMaterial:
    """Parsed material data from meshcat."""

    material_type: MaterialType
    color: Color = None
    opacity: float = 1.0
    transparent: bool = False

    # PBR properties (MeshStandardMaterial)
    metalness: float = 0.0
    roughness: float = 1.0

    # Phong properties
    shininess: float = 30.0
    specular: Color = None

    # Emission
    emissive: Color = None
    emissive_intensity: float = 1.0

    # Textures (UUIDs or URLs)
    map: str | None = None  # Diffuse/color map
    normal_map: str | None = None
    roughness_map: str | None = None
    metalness_map: str | None = None
    emissive_map: str | None = None

    # Rendering options
    side: str = "front"  # "front", "back", "double"
    wireframe: bool = False
    flat_shading: bool = False
    vertex_colors: bool = False

    def __post_init__(self):
        if self.color is None:
            self.color = Color(1.0, 1.0, 1.0)
        if self.emissive is None:
            self.emissive = Color(0.0, 0.0, 0.0)
        if self.specular is None:
            self.specular = Color(0.1, 0.1, 0.1)


def parse_material(mat_data: dict[str, Any]) -> ParsedMaterial | None:
    """Parse material data from a meshcat object.

    Args:
        mat_data: Material dictionary from meshcat command

    Returns:
        ParsedMaterial or None if unsupported
    """
    mat_type_str = mat_data.get("type", "")

    # Map string to enum
    type_map = {
        "MeshStandardMaterial": MaterialType.MESH_STANDARD,
        "MeshPhongMaterial": MaterialType.MESH_PHONG,
        "MeshBasicMaterial": MaterialType.MESH_BASIC,
        "MeshLambertMaterial": MaterialType.MESH_LAMBERT,
        "LineBasicMaterial": MaterialType.LINE_BASIC,
        "PointsMaterial": MaterialType.POINTS,
    }

    mat_type = type_map.get(mat_type_str)
    if mat_type is None:
        print(f"Warning: Unsupported material type: {mat_type_str}")
        return None

    # Parse color
    color = Color(1.0, 1.0, 1.0)
    if "color" in mat_data:
        color_val = mat_data["color"]
        if isinstance(color_val, int):
            color = Color.from_int(color_val)
        elif isinstance(color_val, str):
            color = Color.from_hex(color_val)

    # Parse emissive
    emissive = Color(0.0, 0.0, 0.0)
    if "emissive" in mat_data:
        emissive_val = mat_data["emissive"]
        if isinstance(emissive_val, int):
            emissive = Color.from_int(emissive_val)
        elif isinstance(emissive_val, str):
            emissive = Color.from_hex(emissive_val)

    # Parse specular
    specular = Color(0.1, 0.1, 0.1)
    if "specular" in mat_data:
        specular_val = mat_data["specular"]
        if isinstance(specular_val, int):
            specular = Color.from_int(specular_val)
        elif isinstance(specular_val, str):
            specular = Color.from_hex(specular_val)

    # Parse side
    side_val = mat_data.get("side", 0)
    if side_val == 0:
        side = "front"
    elif side_val == 1:
        side = "back"
    elif side_val == 2:
        side = "double"
    else:
        side = "front"

    # Parse texture references
    map_uuid = None
    if "map" in mat_data and mat_data["map"]:
        map_data = mat_data["map"]
        if isinstance(map_data, dict):
            map_uuid = map_data.get("uuid")
        elif isinstance(map_data, str):
            map_uuid = map_data

    return ParsedMaterial(
        material_type=mat_type,
        color=color,
        opacity=mat_data.get("opacity", 1.0),
        transparent=mat_data.get("transparent", False),
        metalness=mat_data.get("metalness", 0.0),
        roughness=mat_data.get("roughness", 1.0),
        shininess=mat_data.get("shininess", 30.0),
        specular=specular,
        emissive=emissive,
        emissive_intensity=mat_data.get("emissiveIntensity", 1.0),
        map=map_uuid,
        normal_map=mat_data.get("normalMap"),
        roughness_map=mat_data.get("roughnessMap"),
        metalness_map=mat_data.get("metalnessMap"),
        emissive_map=mat_data.get("emissiveMap"),
        side=side,
        wireframe=mat_data.get("wireframe", False),
        flat_shading=mat_data.get("flatShading", False),
        vertex_colors=mat_data.get("vertexColors", False),
    )


def shininess_to_roughness(shininess: float) -> float:
    """Convert Phong shininess to PBR roughness.

    Higher shininess = sharper specular = lower roughness.
    This is an approximation; the mapping isn't exact.

    Args:
        shininess: Phong shininess value (typically 0-1000)

    Returns:
        Roughness value (0-1)
    """
    # Common approximation: roughness = sqrt(2 / (shininess + 2))
    import math

    if shininess <= 0:
        return 1.0
    roughness = math.sqrt(2.0 / (shininess + 2.0))
    return min(1.0, max(0.0, roughness))
