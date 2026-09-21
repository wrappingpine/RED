"""
Synthetic hand motion sequences for deterministic algorithm testing.
Implements §64: Synthetic Testing.
Generates complex geometric motion profiles, clicks, pinches, hand losses, and transitions.
"""

import math
from typing import List, Dict, Any
from ..vision.hand_tracker import Hand, Landmark, HandLandmark
from ..vision.face_tracker import Face

def create_synthetic_landmark(x: float, y: float, z: float = 0.0) -> Landmark:
    return Landmark(x=x, y=y, z=z)

def create_synthetic_hand(
    wrist_x: float,
    wrist_y: float,
    index_extended: bool = True,
    middle_extended: bool = False,
    ring_extended: bool = False,
    pinky_extended: bool = False,
    thumb_pinched: bool = False,
    handedness: str = "Right",
    confidence: float = 1.0
) -> Hand:
    """Generate a realistic mock hand skeleton layout centered at (wrist_x, wrist_y)."""
    landmarks = [create_synthetic_landmark(wrist_x, wrist_y, 0.0) for _ in range(21)]
    
    # Simple finger layout offsets relative to wrist
    # WRIST offsets
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    offsets = {
        WRIST: (0.0, 0.0),
        THUMB_CMC: (-0.05, -0.05),
        THUMB_MCP: (-0.08, -0.1),
        THUMB_IP: (-0.09, -0.15),
        THUMB_TIP: (-0.09 if thumb_pinched else -0.12, -0.18 if thumb_pinched else -0.2),
        INDEX_MCP: (-0.03, -0.15),
        INDEX_PIP: (-0.03, -0.22),
        INDEX_DIP: (-0.03, -0.27),
        INDEX_TIP: (-0.03, -0.32 if index_extended else -0.22),
        MIDDLE_MCP: (0.0, -0.16),
        MIDDLE_PIP: (0.0, -0.24),
        MIDDLE_DIP: (0.0, -0.3),
        MIDDLE_TIP: (0.0, -0.35 if middle_extended else -0.24),
        RING_MCP: (0.03, -0.15),
        RING_PIP: (0.03, -0.22),
        RING_DIP: (0.03, -0.27),
        RING_TIP: (0.03, -0.32 if ring_extended else -0.22),
        PINKY_MCP: (0.06, -0.13),
        PINKY_PIP: (0.06, -0.19),
        PINKY_DIP: (0.06, -0.23),
        PINKY_TIP: (0.06, -0.27 if pinky_extended else -0.19)
    }
    
    for idx, (dx, dy) in offsets.items():
        # Mirror x offsets if left hand
        if handedness == "Left":
            dx = -dx
        landmarks[idx] = create_synthetic_landmark(wrist_x + dx, wrist_y + dy, 0.0)
        
    return Hand(landmarks=landmarks, handedness=handedness, confidence=confidence)


class SyntheticMotionGenerator:
    """
    Generates synthetic hand landmark sequences representing specific movements or actions.
    Useful for §64: Synthetic Testing.
    """
    
    @staticmethod
    def generate_stationary(duration: float = 1.0, fps: int = 30) -> List[Dict[str, Any]]:
        """A stationary hand hover sequence."""
        frames = []
        num_frames = int(duration * fps)
        for i in range(num_frames):
            hand = create_synthetic_hand(0.5, 0.5, index_extended=True)
            frames.append({
                "timestamp": i / fps,
                "hands": [hand],
                "faces": []
            })
        return frames

    @staticmethod
    def generate_linear_movement(
        start: tuple, end: tuple, speed: float = 0.2, fps: int = 30
    ) -> List[Dict[str, Any]]:
        """Moves hand linearly from start (x,y) to end (x,y)."""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dist = (dx*dx + dy*dy) ** 0.5
        duration = dist / speed if speed > 0 else 1.0
        num_frames = int(duration * fps)
        
        frames = []
        for i in range(num_frames):
            t = i / num_frames
            curr_x = start[0] + dx * t
            curr_y = start[1] + dy * t
            hand = create_synthetic_hand(curr_x, curr_y, index_extended=True)
            frames.append({
                "timestamp": i / fps,
                "hands": [hand],
                "faces": []
            })
        return frames

    @staticmethod
    def generate_circular_movement(
        center: tuple, radius: float = 0.1, duration: float = 2.0, fps: int = 30
    ) -> List[Dict[str, Any]]:
        """Moves hand in a circular path around center."""
        num_frames = int(duration * fps)
        frames = []
        for i in range(num_frames):
            t = i / num_frames
            angle = 2 * math.pi * t
            curr_x = center[0] + radius * math.cos(angle)
            curr_y = center[1] + radius * math.sin(angle)
            hand = create_synthetic_hand(curr_x, curr_y, index_extended=True)
            frames.append({
                "timestamp": i / fps,
                "hands": [hand],
                "faces": []
            })
        return frames

    @staticmethod
    def generate_pinch_sequence(
        hold_duration: float = 0.5, fps: int = 30
    ) -> List[Dict[str, Any]]:
        """Pinch, hold pinch, and release sequence."""
        frames = []
        # Phase 1: Open hover (0.5s)
        for i in range(15):
            hand = create_synthetic_hand(0.5, 0.5, index_extended=True, thumb_pinched=False)
            frames.append({"timestamp": i / fps, "hands": [hand], "faces": []})
        
        # Phase 2: Pinch trigger (approx 0.1s transition, or direct trigger)
        # Hold pinch
        num_hold = int(hold_duration * fps)
        start_idx = len(frames)
        for i in range(num_hold):
            hand = create_synthetic_hand(0.5, 0.5, index_extended=False, thumb_pinched=True)
            frames.append({"timestamp": (start_idx + i) / fps, "hands": [hand], "faces": []})
            
        # Phase 3: Release hover (0.5s)
        start_idx = len(frames)
        for i in range(15):
            hand = create_synthetic_hand(0.5, 0.5, index_extended=True, thumb_pinched=False)
            frames.append({"timestamp": (start_idx + i) / fps, "hands": [hand], "faces": []})
            
        return frames
