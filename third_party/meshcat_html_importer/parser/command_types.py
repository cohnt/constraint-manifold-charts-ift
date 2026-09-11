# SPDX-License-Identifier: MIT
"""Command types for meshcat protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CommandType(Enum):
    """Types of commands in meshcat protocol."""

    SET_OBJECT = "set_object"
    SET_TRANSFORM = "set_transform"
    DELETE = "delete"
    SET_PROPERTY = "set_property"
    SET_ANIMATION = "set_animation"
    CAPTURE_IMAGE = "capture_image"
    SET_RENDER_CALLBACK = "set_render_callback"


@dataclass
class Command:
    """A parsed meshcat command."""

    type: CommandType
    path: str
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Command:
        """Create a Command from a decoded msgpack dictionary."""
        cmd_type_str = d.get("type", "")

        # Map string type to enum
        type_map = {
            "set_object": CommandType.SET_OBJECT,
            "set_transform": CommandType.SET_TRANSFORM,
            "delete": CommandType.DELETE,
            "set_property": CommandType.SET_PROPERTY,
            "set_animation": CommandType.SET_ANIMATION,
            "capture_image": CommandType.CAPTURE_IMAGE,
            "set_render_callback": CommandType.SET_RENDER_CALLBACK,
        }

        cmd_type = type_map.get(cmd_type_str)
        if cmd_type is None:
            raise ValueError(f"Unknown command type: {cmd_type_str}")

        return cls(
            type=cmd_type,
            path=d.get("path", ""),
            data=d,
        )


@dataclass
class GeometryData:
    """Parsed geometry data from a meshcat object."""

    geometry_type: str
    positions: list[float] | None = None
    normals: list[float] | None = None
    uvs: list[float] | None = None
    indices: list[int] | None = None
    # For primitives
    width: float | None = None
    height: float | None = None
    depth: float | None = None
    radius: float | None = None
    radial_segments: int | None = None
    height_segments: int | None = None
    # For mesh files
    mesh_format: str | None = None
    mesh_data: bytes | None = None


@dataclass
class MaterialData:
    """Parsed material data from a meshcat object."""

    material_type: str
    color: int | None = None
    opacity: float = 1.0
    transparent: bool = False
    metalness: float = 0.0
    roughness: float = 1.0
    emissive: int = 0
    shininess: float = 30.0
    # Texture references
    map_uuid: str | None = None
    # Vertex colors
    vertex_colors: bool = False


@dataclass
class TransformData:
    """Parsed transform data."""

    matrix: list[float]  # 4x4 column-major matrix (16 elements)


@dataclass
class AnimationTrack:
    """A single animation track for a property."""

    path: str
    property_name: str  # e.g., "position", "quaternion", "scale"
    times: list[float]
    values: list[float]


@dataclass
class AnimationClip:
    """A collection of animation tracks."""

    name: str
    fps: float
    tracks: list[AnimationTrack] = field(default_factory=list)
