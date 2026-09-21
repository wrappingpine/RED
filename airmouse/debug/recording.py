"""
Recording and Replay system for AirMouse landmark streams.
Implements §63: Recording and Replay.
Allows recording hand/face landmark streams to lightweight JSON files and replaying them
to simulate camera inputs deterministically.
"""

import json
import time
import os
from typing import List, Dict, Any, Optional
from dataclasses import asdict
from ..vision.hand_tracker import Hand, Landmark
from ..vision.face_tracker import Face

class LandmarkStreamRecorder:
    """
    Records landmark streams (hands and faces) with high-resolution timestamps
    to a portable, inspectable JSON format.
    """
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.frames: List[Dict[str, Any]] = []
        self.start_time: float = 0.0
        self.recording: bool = False

    def start(self):
        """Start recording."""
        self.frames = []
        self.start_time = time.time()
        self.recording = True

    def record_frame(self, hands: List[Hand], faces: List[Face]):
        """Record a single frame of landmarks."""
        if not self.recording:
            return
        
        timestamp = time.time() - self.start_time
        
        frame_data = {
            "timestamp": timestamp,
            "hands": [self._serialize_hand(h) for h in hands],
            "faces": [self._serialize_face(f) for f in faces]
        }
        self.frames.append(frame_data)

    def stop(self) -> str:
        """Stop recording and write to file."""
        self.recording = False
        data = {
            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": time.time() - self.start_time,
            "frame_count": len(self.frames),
            "frames": self.frames
        }
        
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        return self.output_path

    def _serialize_hand(self, hand: Hand) -> Dict[str, Any]:
        landmarks = []
        for lm in hand.landmarks:
            landmarks.append({"x": lm.x, "y": lm.y, "z": lm.z})
        return {
            "landmarks": landmarks,
            "handedness": hand.handedness,
            "confidence": hand.confidence
        }

    def _serialize_face(self, face: Face) -> Dict[str, Any]:
        landmarks = []
        for lm in face.landmarks:
            landmarks.append({"x": lm.x, "y": lm.y, "z": lm.z})
        return {
            "landmarks": landmarks,
            "confidence": face.confidence
        }


class LandmarkStreamReplayer:
    """
    Replays recorded landmark streams at precise relative timings.
    """
    def __init__(self, input_path: str):
        self.input_path = input_path
        self.data: Dict[str, Any] = {}
        self.frames: List[Dict[str, Any]] = []
        self.current_frame_idx: int = 0
        self.replay_start_time: float = 0.0
        self.loop: bool = False
        self.playing: bool = False
        self.load()

    def load(self):
        """Load recording from file."""
        with open(self.input_path, 'r') as f:
            self.data = json.load(f)
        self.frames = self.data.get("frames", [])
        self.current_frame_idx = 0
        self.playing = True

    def start(self, loop: bool = False):
        """Start replaying."""
        self.loop = loop
        self.current_frame_idx = 0
        self.replay_start_time = time.time()
        self.playing = True

    def get_next_frame(self) -> Optional[Dict[str, Any]]:
        """Get the next frame based on elapsed time, or None if finished."""
        if not self.playing:
            return None
        
        # Handle empty recording
        if not self.frames:
            self.playing = False
            return None
        
        elapsed = time.time() - self.replay_start_time
        
        # Get current frame
        frame = self.frames[self.current_frame_idx]
        if elapsed >= frame["timestamp"]:
            self.current_frame_idx += 1
            
            # Return frame if within bounds
            if self.current_frame_idx <= len(self.frames):
                return {
                    "hands": [self._deserialize_hand(h) for h in frame["hands"]],
                    "faces": [self._deserialize_face(f) for f in frame["faces"]],
                    "timestamp": frame["timestamp"]
                }
            # End of recording
            self.playing = False
            return None
        
        # Not yet time for this frame
        return None

    def get_first_frame(self) -> Optional[Dict[str, Any]]:
        """Get the first frame immediately without waiting for timestamp."""
        if not self.frames:
            return None
        frame = self.frames[0]
        return {
            "hands": [self._deserialize_hand(h) for h in frame["hands"]],
            "faces": [self._deserialize_face(f) for f in frame["faces"]],
            "timestamp": frame["timestamp"]
        }

    def _deserialize_hand(self, data: Dict[str, Any]) -> Hand:
        landmarks = [Landmark(x=lm["x"], y=lm["y"], z=lm["z"]) for lm in data["landmarks"]]
        return Hand(
            landmarks=landmarks,
            handedness=data["handedness"],
            confidence=data["confidence"]
        )

    def _deserialize_face(self, data: Dict[str, Any]) -> Face:
        landmarks = [Landmark(x=lm["x"], y=lm["y"], z=lm["z"]) for lm in data["landmarks"]]
        return Face(
            landmarks=landmarks,
            confidence=data["confidence"]
        )
