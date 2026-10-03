"""
Cursor Control Module for Air Mouse

Maps hand landmarks to screen coordinates with:
- Dead zone filtering
- Exponential moving average (EMA) smoothing
- Configurable sensitivity and acceleration curves
- Screen boundary clamping
"""

import time
import math
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


from airmouse.control.smoothing import (
    SmoothingAlgorithm, 
    SensitivityMode, 
    CursorConfig,
    SmoothingConfig,
    SmoothingFilter,
    EmaFilter,
    OneEuroFilter,
    SmoothingFilterFactory
)


class CursorController:
    """
    Maps hand landmarks to screen cursor position with smoothing and acceleration.
    """

    def __init__(self, config: Optional[CursorConfig] = None):
        self.config = config or CursorConfig()
        self._screen_width = self.config.screen_width
        self._screen_height = self.config.screen_height
        self._camera_width = self.config.camera_width
        self._camera_height = self.config.camera_height

        # Create smoothing filters using factory - operate on [0,1] plane coordinates
        smoothing_config = SmoothingConfig(
            algorithm=self.config.smoothing,
            ema_alpha=self.config.ema_alpha,
            one_euro_min_cutoff=self.config.one_euro_min_cutoff,
            one_euro_beta=self.config.one_euro_beta,
            one_euro_d_cutoff=self.config.one_euro_d_cutoff,
        )
        self._smoother_x, self._smoother_y = SmoothingFilterFactory.create_pair(smoothing_config)

        # State
        self._last_position: Optional[Tuple[float, float]] = None
        self._last_time: Optional[float] = None
        self._is_active = False
        self._reference_point: Optional[Tuple[float, float]] = None  # Initial hand position (calibration zero)
        self._last_plane_position: Optional[Tuple[float, float]] = None  # Previous frame plane position (for frame-to-frame deltas)
        # Fractional accumulator for subpixel cursor movement (§17)
        self._accumulator_x: float = 0.0
        self._accumulator_y: float = 0.0

    def update_screen_size(self, width: int, height: int):
        """Update screen dimensions."""
        self._screen_width = width
        self._screen_height = height

    def update_camera_size(self, width: int, height: int):
        """Update camera frame dimensions."""
        self._camera_width = width
        self._camera_height = height

    def _clamp_velocity(self, dx: float, dy: float, dt: float) -> Tuple[float, float]:
        """Clamp movement to max_velocity (pixels/sec) per §24."""
        # Use a minimum dt to prevent over-clamping in rapid test succession
        dt = max(dt, 0.001)  # 1ms minimum

        # Determine max velocity based on sensitivity mode
        if self.config.sensitivity_mode == SensitivityMode.PRECISION:
            max_vel = self.config.max_velocity_precision
        else:
            max_vel = self.config.max_velocity

        max_dist = max_vel * dt  # Max pixels allowed in this frame

        distance = math.sqrt(dx * dx + dy * dy)
        if distance > max_dist:
            scale = max_dist / distance
            return (dx * scale, dy * scale)
        return (dx, dy)

    def _normalize_to_screen(self, x_norm: float, y_norm: float) -> Tuple[float, float]:
        """Convert normalized coordinates (0-1) to screen pixels."""
        # Apply inversion if needed
        if self.config.invert_x:
            x_norm = 1.0 - x_norm
        if self.config.invert_y:
            y_norm = 1.0 - y_norm

        # Map to virtual desktop coordinates (supports multi-monitor §25)
        if self.config.monitor_count > 1:
            vw = self.config.virtual_desktop_width or (self.config.screen_width * self.config.monitor_count)
            vh = self.config.virtual_desktop_height or self.config.screen_height
            screen_x = x_norm * vw
            screen_y = y_norm * vh
        else:
            screen_x = x_norm * self._screen_width
            screen_y = y_norm * self._screen_height

        # Clamp to screen bounds
        screen_x = max(0, min(self._screen_width - 1, screen_x))
        screen_y = max(0, min(self._screen_height - 1, screen_y))

        return (screen_x, screen_y)

    def _apply_dead_zone(self, dx: float, dy: float) -> Tuple[float, float]:
        """Apply dead zone - ignore small movements."""
        distance = math.sqrt(dx * dx + dy * dy)
        if distance < self.config.dead_zone_radius:
            return (0.0, 0.0)
        return (dx, dy)

    def _apply_acceleration(self, dx: float, dy: float) -> Tuple[float, float]:
        """Apply velocity-based acceleration curve to movement per spec P5.
        
        Implements velocity-adaptive acceleration for smooth cursor control:
        - Low velocity: higher acceleration for precision
        - High velocity: reduced acceleration for control
        - Sustained movements: consistent response
        """
        if self.config.acceleration == 1.0:
            return (dx, dy)

        distance = math.sqrt(dx * dx + dy * dy)
        if distance == 0:
            return (0.0, 0.0)

        # Get previous velocity for adaptive acceleration
        prev_velocity = getattr(self, '_last_velocity', 0.0)
        current_velocity = distance

        # Base acceleration factor (same as before)
        base_factor = distance ** (self.config.acceleration - 1.0)
        
        # Velocity-based adaptive adjustment
        # At low velocity: moderate acceleration boost for precision
        # At high velocity: reduced acceleration for control
        # This creates a smoother response curve across velocity ranges
        
        max_vel = self.config.max_velocity
        velocity_ratio = min(current_velocity / max_vel, 1.0)
        
        # Velocity factor: reduces acceleration at high speeds
        if velocity_ratio < 0.3:
            velocity_factor = 1.0 + (0.3 - velocity_ratio) * 0.5  # Boost at very low speed
        elif velocity_ratio > 0.7:
            velocity_factor = 1.0 - (velocity_ratio - 0.7) * 0.3  # Reduce at high speed
        else:
            velocity_factor = 1.0  # Normal at medium speed
        
        # Apply smooth transition to avoid sudden jumps
        if prev_velocity > 0 and abs(velocity_factor - 1.0) > 0.2:
            transition_factor = min(0.5, (current_velocity - prev_velocity) / max_vel)
            velocity_factor = velocity_factor * (1.0 - transition_factor) + 1.0 * transition_factor
        
        # Final acceleration factor
        adaptive_factor = base_factor * velocity_factor
        
        # Update velocity for next frame
        self._last_velocity = current_velocity
        
        return (dx * adaptive_factor, dy * adaptive_factor)

    def _apply_sensitivity(self, dx: float, dy: float) -> Tuple[float, float]:
        """Apply sensitivity multiplier."""
        return (dx * self.config.effective_sensitivity, dy * self.config.effective_sensitivity)

    def map_hand_to_cursor(self, hand) -> Optional[Tuple[int, int]]:
        """
        Map hand landmarks to screen cursor position.

        Args:
            hand: Hand object from hand_tracker with landmarks and derived properties

        Returns:
            (screen_x, screen_y) tuple or None if no valid hand
        """
        if not hand or not hand.landmarks:
            return None

        # Get the reference point (index tip or palm center)
        if self.config.use_index_tip:
            ref_point = hand.index_tip
        else:
            ref_point = hand.palm_center

        if not ref_point:
            return None

        # Use monotonic clock for runtime timing (§18)
        now = time.monotonic()

        # Convert to normalized coordinates relative to camera frame
        # Hand landmarks are already normalized (0-1)
        x_norm = ref_point.x
        y_norm = ref_point.y

        # Initialize reference point on first frame
        if self._reference_point is None:
            self._reference_point = (x_norm, y_norm)

        # Apply dead zone relative to reference point (hand's initial position)
        dx = x_norm - self._reference_point[0]
        dy = y_norm - self._reference_point[1]
        dx, dy = self._apply_dead_zone(dx, dy)

        # Apply sensitivity and acceleration
        dx, dy = self._apply_sensitivity(dx, dy)
        dx, dy = self._apply_acceleration(dx, dy)

        # Convert back to absolute normalized coordinates
        x_norm = self._reference_point[0] + dx
        y_norm = self._reference_point[1] + dy

        # Clamp to valid range
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))

        # Apply smoothing on plane coordinates [0,1] (not screen pixels)
        if self.config.smoothing != SmoothingAlgorithm.NONE:
            x_norm = self._smoother_x.filter(x_norm, now)
            y_norm = self._smoother_y.filter(y_norm, now)

        # Convert to screen coordinates
        screen_x, screen_y = self._normalize_to_screen(x_norm, y_norm)

        # Apply velocity clamping (§24)
        # dt = now - previous_time (§18: compute dt BEFORE updating timestamp)
        if self._last_position is not None:
            dt = max(now - self._last_time, 0.001)
            dx = screen_x - self._last_position[0]
            dy = screen_y - self._last_position[1]
            dx, dy = self._clamp_velocity(dx, dy, dt)
            screen_x = self._last_position[0] + dx
            screen_y = self._last_position[1] + dy

        # Convert to integers
        result = (int(screen_x), int(screen_y))

        # Update state AFTER computing dt (§18)
        self._last_position = result
        self._last_time = now
        self._is_active = True

        return result

    def get_relative_movement(self, hand) -> Optional[Tuple[int, int]]:
        """
        Get relative mouse movement (dx, dy) for uinput.

        Args:
            hand: Hand object from hand_tracker

        Returns:
            (dx, dy) relative movement or None
        """
        if not hand or not hand.landmarks:
            return None

        # Save previous position before calling map_hand_to_cursor
        # (which updates _last_position internally)
        prev_pos = self._last_position

        current_pos = self.map_hand_to_cursor(hand)
        if current_pos is None:
            return None

        if prev_pos is None:
            return (0, 0)

        dx = current_pos[0] - prev_pos[0]
        dy = current_pos[1] - prev_pos[1]

        # Apply velocity clamping (§24)
        dt = max(time.time() - self._last_time, 0.001)
        dx, dy = self._clamp_velocity(dx, dy, dt)

        return (dx, dy)

    def get_relative_movement_from_plane(self, x_norm: float, y_norm: float) -> Optional[Tuple[int, int]]:
        """
        Get relative mouse movement from normalized plane coordinates (head-relative mode).

        Uses frame-to-frame deltas: delta = current - previous_frame_position.
        The _reference_point is the calibration zero point (initial hand position),
        used only to establish the first previous_frame_position.

        Args:
            x_norm: Normalized X position on virtual plane (0-1)
            y_norm: Normalized Y position on virtual plane (0-1)

        Returns:
            (dx, dy) relative movement in screen pixels for uinput, or None
        """
        current_time = time.monotonic()

        # Initialize reference point and previous frame position on first call
        if self._reference_point is None:
            self._reference_point = (x_norm, y_norm)
            self._last_plane_position = (x_norm, y_norm)
            self._last_time = current_time
            return (0, 0)  # No movement on first frame

        # Frame-to-frame delta (§4.1): delta = current - previous_frame
        # NOT delta = current - initial_reference
        if self._last_plane_position is None:
            self._last_plane_position = (x_norm, y_norm)
            return (0, 0)

        # Apply smoothing on plane coordinates [0,1] BEFORE computing frame-to-frame delta
        # This ensures smooth deltas, similar to how it works in map_hand_to_cursor
        if self.config.smoothing != SmoothingAlgorithm.NONE:
            x_norm = self._smoother_x.filter(x_norm, current_time)
            y_norm = self._smoother_y.filter(y_norm, current_time)

        dx = x_norm - self._last_plane_position[0]
        dy = y_norm - self._last_plane_position[1]

        # Update previous frame position for next call (after smoothing)
        self._last_plane_position = (x_norm, y_norm)

        # Apply dead zone
        dx, dy = self._apply_dead_zone(dx, dy)
        if dx == 0.0 and dy == 0.0:
            return (0, 0)

        # Apply sensitivity and acceleration
        dx, dy = self._apply_sensitivity(dx, dy)
        dx, dy = self._apply_acceleration(dx, dy)

        # Convert normalized movement to screen pixels
        screen_dx = dx * self._screen_width
        screen_dy = dy * self._screen_height

        # Apply velocity clamping (§24)
        dt = max(current_time - self._last_time, 0.001)
        screen_dx, screen_dy = self._clamp_velocity(screen_dx, screen_dy, dt)

        self._last_time = current_time

        # Fractional accumulator (§17): maintain subpixel remainder
        # to prevent cursor stickiness from int() truncation.
        # int() truncates toward zero, so subtracting the truncated
        # value gives the correct remainder for both signs.
        self._accumulator_x += screen_dx
        self._accumulator_y += screen_dy
        out_x = int(self._accumulator_x)
        out_y = int(self._accumulator_y)
        self._accumulator_x -= out_x
        self._accumulator_y -= out_y

        return (out_x, out_y)

    def reset(self):
        """Reset controller state."""
        self._last_position = None
        self._last_time = None
        self._is_active = False
        self._reference_point = None
        self._last_plane_position = None
        self._accumulator_x = 0.0
        self._accumulator_y = 0.0
        if hasattr(self._smoother_x, 'reset'):
            self._smoother_x.reset()
            self._smoother_y.reset()
        else:
            self._smoother_x.value = None
            self._smoother_y.value = None

    def set_active(self, active: bool):
        """Set whether controller is active (tracking hand)."""
        if not active:
            self.reset()
        self._is_active = active

    def is_active(self) -> bool:
        return self._is_active

    def set_sensitivity_mode(self, mode: SensitivityMode):
        """Set the sensitivity mode (Precision/Normal/Fast)."""
        self.config.sensitivity_mode = mode
        logger.info(f"Sensitivity mode changed to: {mode.value} (factor: {self.config.effective_sensitivity:.2f})")

    def get_sensitivity_mode(self) -> SensitivityMode:
        """Get current sensitivity mode."""
        return self.config.sensitivity_mode


def get_screen_size() -> Tuple[int, int]:
    """Get primary screen size using tkinter (cross-platform)."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return (width, height)
    except Exception as e:
        logger.warning(f"Could not detect screen size: {e}, using defaults")
        return (1920, 1080)


if __name__ == "__main__":
    # Simple test
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from vision.hand_tracker import Hand, Landmark, HandLandmark

    logging.basicConfig(level=logging.INFO)

    # Create mock hand for testing
    landmarks = []
    for i in range(21):
        # Create a hand pointing at center
        if i == HandLandmark.INDEX_TIP.value:
            landmarks.append(Landmark(x=0.5, y=0.5, z=0.0))
        elif i == HandLandmark.WRIST.value:
            landmarks.append(Landmark(x=0.5, y=0.7, z=0.0))
        else:
            landmarks.append(Landmark(x=0.5, y=0.6, z=0.0))

    hand = Hand(landmarks=landmarks, handedness="Right", confidence=1.0)

    config = CursorConfig(
        screen_width=1920,
        screen_height=1080,
        dead_zone_radius=0.02,
        sensitivity=1.5,
        acceleration=1.2,
        smoothing=SmoothingAlgorithm.EMA,
        ema_alpha=0.3
    )

    controller = CursorController(config)

    print("Testing CursorController...")
    for i in range(10):
        # Simulate hand moving right
        hand.index_tip.x = 0.5 + i * 0.02
        pos = controller.map_hand_to_cursor(hand)
        rel = controller.get_relative_movement(hand)
        print(f"Frame {i}: pos={pos}, rel={rel}")

    controller.reset()
    print("Test complete")