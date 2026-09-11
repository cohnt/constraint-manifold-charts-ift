# SPDX-License-Identifier: MIT
"""Animation data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TrackType(Enum):
    """Types of animation tracks."""

    POSITION = "position"
    QUATERNION = "quaternion"
    SCALE = "scale"
    VISIBLE = "visible"


@dataclass
class AnimationTrack:
    """A single animation track for a property."""

    name: str
    track_type: TrackType
    times: list[float]  # Time in seconds
    values: list[float]  # Flattened values (3 for position/scale, 4 for quat)

    def get_value_at(self, index: int) -> tuple:
        """Get the value tuple at a given keyframe index."""
        if self.track_type == TrackType.QUATERNION:
            start = index * 4
            return tuple(self.values[start : start + 4])
        elif self.track_type in (TrackType.POSITION, TrackType.SCALE):
            start = index * 3
            return tuple(self.values[start : start + 3])
        elif self.track_type == TrackType.VISIBLE:
            return (self.values[index],)
        return ()

    def __len__(self) -> int:
        """Return number of keyframes."""
        return len(self.times)


@dataclass
class AnimationClip:
    """A collection of animation tracks for an object."""

    name: str
    tracks: list[AnimationTrack] = field(default_factory=list)
    fps: float = 30.0

    @property
    def duration(self) -> float:
        """Get total duration in seconds."""
        if not self.tracks:
            return 0.0
        max_time = 0.0
        for track in self.tracks:
            if track.times:
                max_time = max(max_time, track.times[-1])
        return max_time

    @property
    def frame_count(self) -> int:
        """Get total frame count."""
        return int(self.duration * self.fps) + 1

    def get_track(self, track_type: TrackType) -> AnimationTrack | None:
        """Get track by type."""
        for track in self.tracks:
            if track.track_type == track_type:
                return track
        return None

    def add_track(self, track: AnimationTrack) -> None:
        """Add a track to the clip."""
        self.tracks.append(track)


def parse_three_js_track(track_data: dict[str, Any]) -> AnimationTrack | None:
    """Parse a Three.js animation track.

    Args:
        track_data: Track dictionary from meshcat animation

    Returns:
        AnimationTrack or None if parsing fails
    """
    name = track_data.get("name", "")
    times = track_data.get("times", [])
    values = track_data.get("values", [])

    # Convert numpy arrays to lists
    import numpy as np

    if isinstance(times, np.ndarray):
        times = times.tolist()
    if isinstance(values, np.ndarray):
        values = values.tolist()

    # Determine track type from name
    # Three.js format: "object.property" or ".property"
    if ".position" in name:
        track_type = TrackType.POSITION
    elif ".quaternion" in name:
        track_type = TrackType.QUATERNION
    elif ".scale" in name:
        track_type = TrackType.SCALE
    elif ".visible" in name:
        track_type = TrackType.VISIBLE
    else:
        return None

    return AnimationTrack(
        name=name,
        track_type=track_type,
        times=times,
        values=values,
    )


def parse_animation_clip(clip_data: dict[str, Any], fps: float = 30.0) -> AnimationClip:
    """Parse a Three.js animation clip.

    Args:
        clip_data: Clip dictionary from meshcat animation
        fps: Frames per second

    Returns:
        AnimationClip object
    """
    name = clip_data.get("name", "MeshcatAnimation")
    tracks_data = clip_data.get("tracks", [])

    clip = AnimationClip(name=name, fps=fps)

    for track_data in tracks_data:
        track = parse_three_js_track(track_data)
        if track:
            clip.add_track(track)

    return clip
