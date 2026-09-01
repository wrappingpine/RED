#!/usr/bin/env python3
"""
Step 1: Observe actual desktop cursor behavior with timestamped logging.

Logs all pipeline values with timestamps to understand cursor behavior.
"""

import sys
import time
import logging
import threading
import queue
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import json

sys.path.insert(0, str(Path(__file__).parent))

from airmouse.control.main_loop import AirMouseController, AirMouseConfig
from airmouse.camera.manager import CameraSettings
from airmouse.vision.hand_tracker import HandTrackerSettings
from airmouse.vision.face_tracker import FaceTrackerSettings
from airmouse.vision.tracking_processor import TrackingConfig
from airmouse.control.cursor import CursorConfig, SensitivityMode

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineLogEntry:
    """Single log entry for the pipeline."""
    timestamp: float
    frame_idx: int

    # Hand detection
    hand_detected: bool = False
    hand_landmarks: Optional[List[Dict]] = None
    index_tip_norm: Optional[tuple] = None
    palm_center_norm: Optional[tuple] = None

    # Face/Head tracking
    face_detected: bool = False
    eye_midpoint: Optional[tuple] = None
    nose_tip: Optional[tuple] = None
    forehead: Optional[tuple] = None

    # Head coordinate system
    head_coords_valid: bool = False
    head_origin: Optional[tuple] = None
    head_forward: Optional[tuple] = None
    head_right: Optional[tuple] = None
    head_up: Optional[tuple] = None

    # Virtual plane
    plane_center_cam: Optional[tuple] = None
    plane_normal_cam: Optional[tuple] = None

    # Ray-plane intersection
    ray_origin_cam: Optional[tuple] = None
    ray_direction_cam: Optional[tuple] = None
    intersection_cam: Optional[tuple] = None
    intersection_head: Optional[tuple] = None

    # Normalized plane coordinates
    u_norm: Optional[float] = None
    v_norm: Optional[float] = None

    # Cursor controller
    rel_movement: Optional[tuple] = None
    sensitivity_mode: Optional[str] = None
    effective_sensitivity: Optional[float] = None

    # Virtual mouse
    virtual_mouse_pos: Optional[tuple] = None
    virtual_mouse_delta: Optional[tuple] = None

    # Gestures
    gestures: List[Dict] = field(default_factory=list)

    # Performance
    frame_time_ms: float = 0.0
    hand_detection_ms: float = 0.0
    tracking_ms: float = 0.0
    gesture_ms: float = 0.0
    mouse_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp,
            'frame_idx': self.frame_idx,
            'hand_detected': self.hand_detected,
            'index_tip_norm': self.index_tip_norm,
            'palm_center_norm': self.palm_center_norm,
            'face_detected': self.face_detected,
            'eye_midpoint': self.eye_midpoint,
            'nose_tip': self.nose_tip,
            'forehead': self.forehead,
            'head_coords_valid': self.head_coords_valid,
            'head_origin': self.head_origin,
            'head_forward': self.head_forward,
            'head_right': self.head_right,
            'head_up': self.head_up,
            'plane_center_cam': self.plane_center_cam,
            'plane_normal_cam': self.plane_normal_cam,
            'ray_origin_cam': self.ray_origin_cam,
            'ray_direction_cam': self.ray_direction_cam,
            'intersection_cam': self.intersection_cam,
            'intersection_head': self.intersection_head,
            'u_norm': self.u_norm,
            'v_norm': self.v_norm,
            'rel_movement': self.rel_movement,
            'sensitivity_mode': self.sensitivity_mode,
            'effective_sensitivity': self.effective_sensitivity,
            'virtual_mouse_pos': self.virtual_mouse_pos,
            'virtual_mouse_delta': self.virtual_mouse_delta,
            'gestures': self.gestures,
            'frame_time_ms': self.frame_time_ms,
            'hand_detection_ms': self.hand_detection_ms,
            'tracking_ms': self.tracking_ms,
            'gesture_ms': self.gesture_ms,
            'mouse_ms': self.mouse_ms,
        }


class CursorObserver:
    """Observes and logs the entire AirMouse pipeline."""

    def __init__(self, log_file: str = "cursor_observation.jsonl"):
        self.log_file = log_file
        self.log_entries: List[PipelineLogEntry] = []
        self.frame_count = 0
        self.running = False
        self._lock = threading.Lock()

    def log_entry(self, entry: PipelineLogEntry):
        """Add log entry and write to file."""
        with self._lock:
            self.log_entries.append(entry)
            # Write to file immediately
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(entry.to_dict()) + '\n')

    def get_stats(self) -> Dict:
        """Compute statistics from logged entries."""
        if not self.log_entries:
            return {}

        # Jitter analysis on virtual mouse position
        positions = [e.virtual_mouse_pos for e in self.log_entries if e.virtual_mouse_pos]
        if len(positions) < 2:
            return {'error': 'Not enough position data'}

        import numpy as np
        pos_array = np.array(positions)

        # Calculate deltas
        deltas = np.diff(pos_array, axis=0)
        distances = np.linalg.norm(deltas, axis=1)

        stats = {
            'total_frames': len(self.log_entries),
            'frames_with_mouse': len(positions),
            'mean_delta_pixels': float(np.mean(distances)),
            'std_delta_pixels': float(np.std(distances)),
            'max_delta_pixels': float(np.max(distances)),
            'rms_delta_pixels': float(np.sqrt(np.mean(distances**2))),
            'jump_count_10px': int(np.sum(distances > 10)),
            'jump_count_50px': int(np.sum(distances > 50)),
            'jump_count_100px': int(np.sum(distances > 100)),
        }

        # Latency analysis - track time from hand detection to mouse move
        latencies = []
        for i, entry in enumerate(self.log_entries):
            if entry.rel_movement and (entry.rel_movement[0] != 0 or entry.rel_movement[1] != 0):
                # Found a frame with movement, look at next frame's mouse delta
                if i + 1 < len(self.log_entries):
                    next_entry = self.log_entries[i + 1]
                    if next_entry.virtual_mouse_delta and (next_entry.virtual_mouse_delta[0] != 0 or next_entry.virtual_mouse_delta[1] != 0):
                        latency = (next_entry.timestamp - entry.timestamp) * 1000  # ms
                        latencies.append(latency)

        if latencies:
            latencies_np = np.array(latencies)
            stats['latency_ms'] = {
                'mean': float(np.mean(latencies_np)),
                'std': float(np.std(latencies_np)),
                'min': float(np.min(latencies_np)),
                'max': float(np.max(latencies_np)),
                'p50': float(np.percentile(latencies_np, 50)),
                'p95': float(np.percentile(latencies_np, 95)),
            }

        return stats


def make_observer_callbacks(observer: CursorObserver):
    """Create callbacks that feed data into the observer."""

    def on_hand_detected(hand):
        pass  # We'll capture in the main loop

    def on_gesture(event):
        pass  # We'll capture in the main loop

    def on_frame_processed(frame, hands):
        pass  # We'll capture in the main loop

    def on_stats_update(stats):
        pass  # We'll capture in the main loop

    return {
        'on_hand_detected': on_hand_detected,
        'on_gesture': on_gesture,
        'on_frame_processed': on_frame_processed,
        'on_stats_update': on_stats_update,
    }


def run_observation(duration_seconds: int = 30):
    """Run AirMouse and observe cursor behavior for specified duration."""

    observer = CursorObserver()

    # Configuration with debug settings
    config = AirMouseConfig()
    config.camera = CameraSettings(device_index=0, width=640, height=480, fps=60)
    config.hand_tracker = HandTrackerSettings(
        max_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    config.tracking = TrackingConfig(
        use_head_relative=True,
        min_face_confidence=0.5,
        dead_zone_radius=0.02,
        max_velocity=0.5,
        velocity_smoothing=0.3,
        one_euro_min_cutoff=1.0,
        one_euro_beta=0.007,
        one_euro_d_cutoff=1.0,
    )
    config.cursor = CursorConfig(
        dead_zone_radius=0.02,
        sensitivity_mode=SensitivityMode.NORMAL,
        base_sensitivity=1.0,
        acceleration=1.2,
        smoothing='one_euro',
    )
    config.target_fps = 60

    controller = AirMouseController(config)

    # Track additional state
    last_virtual_mouse_pos = None
    frame_idx = 0
    start_time = time.time()

    # We'll intercept the _process_frame method to log everything
    original_process_frame = controller._process_frame

    def logged_process_frame(frame):
        nonlocal last_virtual_mouse_pos, frame_idx
        frame_start = time.time()

        # Get all the pipeline data BEFORE calling original
        timestamp = time.time()

        # Hand detection
        hand_start = time.time()
        hands = controller.hand_tracker.process(frame)
        hand_detection_time = (time.time() - hand_start) * 1000

        # Face detection
        faces = []
        face_detection_time = 0.0
        if controller.face_tracker:
            face_start = time.time()
            faces = controller.face_tracker.process(frame)
            face_detection_time = (time.time() - face_start) * 1000

        # Tracking processor
        tracking_start = time.time()
        tracking_result = controller.tracking_processor.process(hands, faces) if controller.tracking_processor else None
        tracking_time = (time.time() - tracking_start) * 1000

        # Build log entry
        entry = PipelineLogEntry(
            timestamp=timestamp,
            frame_idx=frame_idx,
            hand_detection_ms=hand_detection_time,
            tracking_ms=tracking_time,
            gesture_ms=0.0,  # Will be updated after gesture recognition
        )

        # Hand data
        if hands:
            entry.hand_detected = True
            primary_hand = hands[0] if hands else None
            if primary_hand:
                if primary_hand.index_tip:
                    entry.index_tip_norm = (primary_hand.index_tip.x, primary_hand.index_tip.y, primary_hand.index_tip.z)
                if primary_hand.palm_center:
                    entry.palm_center_norm = (primary_hand.palm_center.x, primary_hand.palm_center.y, primary_hand.palm_center.z)

        # Face data
        if faces:
            entry.face_detected = True
            face = faces[0]
            if face.eye_midpoint:
                entry.eye_midpoint = (face.eye_midpoint.x, face.eye_midpoint.y, face.eye_midpoint.z)
            if face.nose_tip:
                entry.nose_tip = (face.nose_tip.x, face.nose_tip.y, face.nose_tip.z)
            if face.forehead:
                entry.forehead = (face.forehead.x, face.forehead.y, face.forehead.z)

        # Head coordinates (from tracking processor internal state)
        if controller.tracking_processor and controller.tracking_processor._head_coords:
            hc = controller.tracking_processor._head_coords
            entry.head_coords_valid = hc.is_valid()
            if hc.is_valid():
                entry.head_origin = tuple(hc.get_origin())
                entry.head_forward = tuple(hc.get_forward_vector())
                entry.head_right = tuple(hc.get_right_vector())
                entry.head_up = tuple(hc.get_up_vector())

                # Check orthonormality
                import numpy as np
                f = np.array(hc.get_forward_vector())
                r = np.array(hc.get_right_vector())
                u = np.array(hc.get_up_vector())
                print(f"  Frame {frame_idx}: Basis check - f·r={np.dot(f,r):.6f}, f·u={np.dot(f,u):.6f}, r·u={np.dot(r,u):.6f}, |f|={np.linalg.norm(f):.6f}, |r|={np.linalg.norm(r):.6f}, |u|={np.linalg.norm(u):.6f}")

        # Virtual plane
        if controller.tracking_processor and controller.tracking_processor._virtual_plane:
            plane = controller.tracking_processor._virtual_plane
            entry.plane_center_cam = tuple(plane.get_plane_center_camera()) if plane.get_plane_center_camera() is not None else None
            entry.plane_normal_cam = tuple(plane.get_plane_normal_camera()) if plane.get_plane_normal_camera() is not None else None

        # Ray-plane intersection and normalized coords
        if tracking_result and tracking_result.projection:
            proj = tracking_result.projection
            entry.u_norm = proj.u
            entry.v_norm = proj.v
            entry.ray_origin_cam = tuple(proj.ray_origin_camera) if proj.ray_origin_camera is not None else None
            entry.ray_direction_cam = tuple(proj.ray_direction_camera) if proj.ray_direction_camera is not None else None
            entry.intersection_cam = tuple(proj.intersection_camera) if proj.intersection_camera is not None else None
            entry.intersection_head = tuple(proj.intersection_head) if proj.intersection_head is not None else None

        # Call original process_frame which does the rest
        original_process_frame(frame)

        # After processing, capture cursor controller state
        if controller.cursor_controller:
            entry.sensitivity_mode = controller.cursor_controller.get_sensitivity_mode().value
            entry.effective_sensitivity = controller.cursor_controller.config.effective_sensitivity
            if controller.cursor_controller._reference_point:
                entry.rel_movement = (entry.u_norm - controller.cursor_controller._reference_point[0] if entry.u_norm else 0,
                                      entry.v_norm - controller.cursor_controller._reference_point[1] if entry.v_norm else 0)

        # Virtual mouse state
        if controller.virtual_mouse:
            entry.virtual_mouse_pos = controller.virtual_mouse.get_position()
            if last_virtual_mouse_pos:
                entry.virtual_mouse_delta = (
                    entry.virtual_mouse_pos[0] - last_virtual_mouse_pos[0],
                    entry.virtual_mouse_pos[1] - last_virtual_mouse_pos[1]
                )
            last_virtual_mouse_pos = entry.virtual_mouse_pos

        # Frame timing
        entry.frame_time_ms = (time.time() - frame_start) * 1000

        # Log the entry
        observer.log_entry(entry)
        frame_idx += 1

        # Print periodic summary
        if frame_idx % 60 == 0:
            stats = observer.get_stats()
            print(f"\n=== Frame {frame_idx} Stats ===")
            for k, v in stats.items():
                if isinstance(v, dict):
                    print(f"  {k}:")
                    for k2, v2 in v.items():
                        print(f"    {k2}: {v2}")
                else:
                    print(f"  {k}: {v}")

    controller._process_frame = logged_process_frame

    print("=" * 60)
    print("AIRMOUSE CURSOR OBSERVATION")
    print(f"Running for {duration_seconds} seconds...")
    print(f"Logging to: {observer.log_file}")
    print("=" * 60)

    if controller.initialize():
        controller.start()

        try:
            time.sleep(duration_seconds)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            controller.stop()
    else:
        print("Failed to initialize")
        return

    # Final stats
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    stats = observer.get_stats()
    for k, v in stats.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for k2, v2 in v.items():
                print(f"    {k2}: {v2}")
        else:
            print(f"  {k}: {v}")

    print(f"\nTotal log entries: {len(observer.log_entries)}")
    print(f"Log file: {observer.log_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Observe AirMouse cursor behavior")
    parser.add_argument("--duration", type=int, default=30, help="Duration in seconds")
    parser.add_argument("--log-file", type=str, default="cursor_observation.jsonl", help="Output log file")
    args = parser.parse_args()

    run_observation(args.duration)