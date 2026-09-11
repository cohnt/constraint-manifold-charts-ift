# SPDX-License-Identifier: MIT
"""Msgpack decoder with support for meshcat's typed array extensions."""

from __future__ import annotations

from typing import Any

# Try vendored msgpack first (for Blender addon), then system package
try:
    from .. import _msgpack as msgpack
except ImportError:
    import msgpack  # type: ignore

import numpy as np

# Extension type codes used by meshcat/Three.js for typed arrays
EXT_UINT8_ARRAY = 0x12  # 18
EXT_INT32_ARRAY = 0x15  # 21
EXT_UINT32_ARRAY = 0x16  # 22
EXT_FLOAT32_ARRAY = 0x17  # 23


def decode_typed_array(code: int, data: bytes) -> np.ndarray:
    """Decode a msgpack extension type to a numpy array."""
    if code == EXT_UINT8_ARRAY:
        return np.frombuffer(data, dtype=np.uint8)
    elif code == EXT_INT32_ARRAY:
        return np.frombuffer(data, dtype=np.int32)
    elif code == EXT_UINT32_ARRAY:
        return np.frombuffer(data, dtype=np.uint32)
    elif code == EXT_FLOAT32_ARRAY:
        return np.frombuffer(data, dtype=np.float32)
    else:
        # Return raw data for unknown extension types
        return data


def ext_hook(code: int, data: bytes) -> Any:
    """Hook for handling msgpack extension types."""
    return decode_typed_array(code, data)


def decode_msgpack(data: bytes) -> Any:
    """Decode msgpack data with support for typed arrays.

    Args:
        data: Raw msgpack bytes

    Returns:
        Decoded Python object (dict, list, etc.)
    """
    return msgpack.unpackb(data, ext_hook=ext_hook, raw=False, strict_map_key=False)


def numpy_to_list(obj: Any) -> Any:
    """Recursively convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_list(item) for item in obj]
    else:
        return obj
