# SPDX-License-Identifier: Apache-2.0
"""Minimal msgpack unpacker for meshcat HTML recordings.

This is a simplified msgpack implementation that handles the subset of
msgpack features used by meshcat recordings. For full msgpack support,
install the msgpack package from PyPI.
"""

from __future__ import annotations

import struct
from typing import Any, Callable


def unpackb(
    data: bytes,
    *,
    ext_hook: Callable[[int, bytes], Any] | None = None,
    raw: bool = True,
    strict_map_key: bool = True,
) -> Any:
    """Unpack msgpack binary data.

    Args:
        data: Msgpack-encoded bytes
        ext_hook: Callback for extension types (code, data) -> value
        raw: If True, return bytes for strings; if False, decode as UTF-8
        strict_map_key: If True, require string map keys

    Returns:
        Decoded Python object
    """
    unpacker = _Unpacker(data, ext_hook=ext_hook, raw=raw)
    return unpacker.unpack()


class _Unpacker:
    """Internal msgpack unpacker."""

    def __init__(
        self,
        data: bytes,
        ext_hook: Callable[[int, bytes], Any] | None = None,
        raw: bool = True,
    ):
        self._data = data
        self._pos = 0
        self._ext_hook = ext_hook
        self._raw = raw

    def _read(self, n: int) -> bytes:
        """Read n bytes from the buffer."""
        if self._pos + n > len(self._data):
            raise ValueError("Unexpected end of data")
        result = self._data[self._pos : self._pos + n]
        self._pos += n
        return result

    def _read_byte(self) -> int:
        """Read a single byte."""
        return self._read(1)[0]

    def unpack(self) -> Any:
        """Unpack the next value."""
        b = self._read_byte()

        # Positive fixint (0x00 - 0x7f)
        if b <= 0x7F:
            return b

        # Fixmap (0x80 - 0x8f)
        if 0x80 <= b <= 0x8F:
            length = b & 0x0F
            return self._read_map(length)

        # Fixarray (0x90 - 0x9f)
        if 0x90 <= b <= 0x9F:
            length = b & 0x0F
            return self._read_array(length)

        # Fixstr (0xa0 - 0xbf)
        if 0xA0 <= b <= 0xBF:
            length = b & 0x1F
            return self._read_str(length)

        # nil
        if b == 0xC0:
            return None

        # false
        if b == 0xC2:
            return False

        # true
        if b == 0xC3:
            return True

        # bin 8
        if b == 0xC4:
            length = self._read_byte()
            return self._read(length)

        # bin 16
        if b == 0xC5:
            length = struct.unpack(">H", self._read(2))[0]
            return self._read(length)

        # bin 32
        if b == 0xC6:
            length = struct.unpack(">I", self._read(4))[0]
            return self._read(length)

        # ext 8
        if b == 0xC7:
            length = self._read_byte()
            ext_type = struct.unpack("b", self._read(1))[0]
            ext_data = self._read(length)
            return self._handle_ext(ext_type, ext_data)

        # ext 16
        if b == 0xC8:
            length = struct.unpack(">H", self._read(2))[0]
            ext_type = struct.unpack("b", self._read(1))[0]
            ext_data = self._read(length)
            return self._handle_ext(ext_type, ext_data)

        # ext 32
        if b == 0xC9:
            length = struct.unpack(">I", self._read(4))[0]
            ext_type = struct.unpack("b", self._read(1))[0]
            ext_data = self._read(length)
            return self._handle_ext(ext_type, ext_data)

        # float 32
        if b == 0xCA:
            return struct.unpack(">f", self._read(4))[0]

        # float 64
        if b == 0xCB:
            return struct.unpack(">d", self._read(8))[0]

        # uint 8
        if b == 0xCC:
            return self._read_byte()

        # uint 16
        if b == 0xCD:
            return struct.unpack(">H", self._read(2))[0]

        # uint 32
        if b == 0xCE:
            return struct.unpack(">I", self._read(4))[0]

        # uint 64
        if b == 0xCF:
            return struct.unpack(">Q", self._read(8))[0]

        # int 8
        if b == 0xD0:
            return struct.unpack("b", self._read(1))[0]

        # int 16
        if b == 0xD1:
            return struct.unpack(">h", self._read(2))[0]

        # int 32
        if b == 0xD2:
            return struct.unpack(">i", self._read(4))[0]

        # int 64
        if b == 0xD3:
            return struct.unpack(">q", self._read(8))[0]

        # fixext 1
        if b == 0xD4:
            ext_type = struct.unpack("b", self._read(1))[0]
            ext_data = self._read(1)
            return self._handle_ext(ext_type, ext_data)

        # fixext 2
        if b == 0xD5:
            ext_type = struct.unpack("b", self._read(1))[0]
            ext_data = self._read(2)
            return self._handle_ext(ext_type, ext_data)

        # fixext 4
        if b == 0xD6:
            ext_type = struct.unpack("b", self._read(1))[0]
            ext_data = self._read(4)
            return self._handle_ext(ext_type, ext_data)

        # fixext 8
        if b == 0xD7:
            ext_type = struct.unpack("b", self._read(1))[0]
            ext_data = self._read(8)
            return self._handle_ext(ext_type, ext_data)

        # fixext 16
        if b == 0xD8:
            ext_type = struct.unpack("b", self._read(1))[0]
            ext_data = self._read(16)
            return self._handle_ext(ext_type, ext_data)

        # str 8
        if b == 0xD9:
            length = self._read_byte()
            return self._read_str(length)

        # str 16
        if b == 0xDA:
            length = struct.unpack(">H", self._read(2))[0]
            return self._read_str(length)

        # str 32
        if b == 0xDB:
            length = struct.unpack(">I", self._read(4))[0]
            return self._read_str(length)

        # array 16
        if b == 0xDC:
            length = struct.unpack(">H", self._read(2))[0]
            return self._read_array(length)

        # array 32
        if b == 0xDD:
            length = struct.unpack(">I", self._read(4))[0]
            return self._read_array(length)

        # map 16
        if b == 0xDE:
            length = struct.unpack(">H", self._read(2))[0]
            return self._read_map(length)

        # map 32
        if b == 0xDF:
            length = struct.unpack(">I", self._read(4))[0]
            return self._read_map(length)

        # Negative fixint (0xe0 - 0xff)
        if b >= 0xE0:
            return b - 256

        raise ValueError(f"Unknown msgpack format: 0x{b:02x}")

    def _read_str(self, length: int) -> str | bytes:
        """Read a string of given length."""
        data = self._read(length)
        if self._raw:
            return data
        return data.decode("utf-8")

    def _read_array(self, length: int) -> list:
        """Read an array of given length."""
        return [self.unpack() for _ in range(length)]

    def _read_map(self, length: int) -> dict:
        """Read a map of given length."""
        result = {}
        for _ in range(length):
            key = self.unpack()
            value = self.unpack()
            result[key] = value
        return result

    def _handle_ext(self, ext_type: int, ext_data: bytes) -> Any:
        """Handle extension type."""
        if self._ext_hook is not None:
            return self._ext_hook(ext_type, ext_data)
        return ext_data
