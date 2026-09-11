# SPDX-License-Identifier: MIT
"""Build Blender materials from meshcat material data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import bpy

from ..scene.materials import (
    MaterialType,
    ParsedMaterial,
    shininess_to_roughness,
)

if TYPE_CHECKING:
    pass


def create_material(
    mat_data: ParsedMaterial,
    name: str,
) -> bpy.types.Material:
    """Create a Blender material from parsed material data.

    Args:
        mat_data: ParsedMaterial from meshcat
        name: Material name

    Returns:
        Blender material
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    nodes.clear()

    # Create output node
    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (400, 0)

    if mat_data.material_type == MaterialType.MESH_BASIC:
        # Basic material - use emission for unlit look
        shader = _create_emission_shader(nodes, links, mat_data)
    else:
        # PBR materials - use Principled BSDF
        shader = _create_principled_shader(nodes, links, mat_data)

    # Connect to output
    # Blender 5.0 changed output name from "Shader" to "BSDF"
    output_socket = shader.outputs.get("BSDF") or shader.outputs.get("Shader")
    links.new(output_socket, output.inputs["Surface"])

    # Note: blend_method/shadow_method were removed in Blender 4.0.
    # Transparency is handled via the shader node tree (Alpha input on Principled BSDF).

    # Set backface culling based on side
    if mat_data.side == "front":
        mat.use_backface_culling = True
    else:
        mat.use_backface_culling = False

    return mat


def _create_principled_shader(
    nodes: bpy.types.NodeTree,
    links: bpy.types.NodeLinks,
    mat_data: ParsedMaterial,
) -> bpy.types.ShaderNode:
    """Create a Principled BSDF shader."""
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.location = (0, 0)

    # Base color
    shader.inputs["Base Color"].default_value = mat_data.color.to_tuple_alpha()

    # Handle different material types
    if mat_data.material_type == MaterialType.MESH_STANDARD:
        # Direct PBR mapping
        shader.inputs["Metallic"].default_value = mat_data.metalness
        shader.inputs["Roughness"].default_value = mat_data.roughness

    elif mat_data.material_type == MaterialType.MESH_PHONG:
        # Convert shininess to roughness
        roughness = shininess_to_roughness(mat_data.shininess)
        shader.inputs["Roughness"].default_value = roughness
        shader.inputs["Metallic"].default_value = 0.0

        # Phong specular can be approximated with specular tint
        # Note: Blender 4.0+ changed specular handling
        if "Specular IOR Level" in shader.inputs:
            # Approximate specular intensity
            spec_intensity = (
                mat_data.specular.r + mat_data.specular.g + mat_data.specular.b
            ) / 3.0
            shader.inputs["Specular IOR Level"].default_value = spec_intensity

    elif mat_data.material_type == MaterialType.MESH_LAMBERT:
        # Lambert is diffuse-only
        shader.inputs["Roughness"].default_value = 1.0
        shader.inputs["Metallic"].default_value = 0.0

    # Emission
    if mat_data.emissive:
        emission_strength = mat_data.emissive_intensity
        emission_color = mat_data.emissive.to_tuple_alpha()
        if any(c > 0 for c in emission_color[:3]):
            shader.inputs["Emission Color"].default_value = emission_color
            shader.inputs["Emission Strength"].default_value = emission_strength

    # Alpha/transparency
    if mat_data.transparent:
        shader.inputs["Alpha"].default_value = mat_data.opacity

    return shader


def _create_emission_shader(
    nodes: bpy.types.NodeTree,
    links: bpy.types.NodeLinks,
    mat_data: ParsedMaterial,
) -> bpy.types.ShaderNode:
    """Create an emission shader for unlit materials."""
    # For MeshBasicMaterial, we want an unlit look
    # Use emission to bypass lighting calculations

    emission = nodes.new("ShaderNodeEmission")
    emission.location = (0, 0)

    # Set color
    emission.inputs["Color"].default_value = mat_data.color.to_tuple_alpha()
    emission.inputs["Strength"].default_value = 1.0

    if mat_data.transparent and mat_data.opacity < 1.0:
        # Mix with transparent shader for alpha
        transparent = nodes.new("ShaderNodeBsdfTransparent")
        transparent.location = (0, -200)

        mix = nodes.new("ShaderNodeMixShader")
        mix.location = (200, 0)

        links.new(transparent.outputs["BSDF"], mix.inputs[1])
        links.new(emission.outputs["Emission"], mix.inputs[2])
        mix.inputs["Fac"].default_value = mat_data.opacity

        return mix

    return emission


def apply_material_to_object(
    obj: bpy.types.Object,
    material: bpy.types.Material,
) -> None:
    """Apply a material to an object.

    Args:
        obj: Blender object
        material: Blender material
    """
    if obj.data is None:
        return

    # Clear existing materials
    obj.data.materials.clear()

    # Add the new material
    obj.data.materials.append(material)


def create_default_material(name: str = "DefaultMaterial") -> bpy.types.Material:
    """Create a default gray material.

    Args:
        name: Material name

    Returns:
        Blender material
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes

    # Get the principled shader (default node setup)
    principled = nodes.get("Principled BSDF")
    if principled:
        principled.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
        principled.inputs["Roughness"].default_value = 0.5

    return mat
