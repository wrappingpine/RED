"""
Cursor Control Module for Air Mouse

Supports two modes:
1. RELATIVE (new): Frame-to-frame hand deltas drive cursor like a physical mouse
2. VIRTUAL_PLANE (legacy): Absolute position mapping via virtual display plane

The new relative mode:
- Dead zone filtering for stationary hand stabilization
- One Euro Filter for low-latency adaptive smoothing
- Velocity-based acceleration curve
- Tracking loss detection and cursor jump prevention
- Reference position tracking with re-centering support
"""

import time
import math
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CursorMode(Enum):
    """Cursor control mode."""
    RELATIVE = "relative"        # New: frame-to-frame deltas like physical mouse
    VIRTUAL_PLANE = "virtual_plane"  # Legacy: absolute position via virtual plane


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

from airmouse.control.relative_movement import (
    RelativeMovementEngine, RelativeMouseConfig, MovementResult, MovementState
)


class CursorController:
    """
    Maps hand landmarks to screen cursor position with smoothing and acceleration.
    
    Supports two modes:
    - RELATIVE: New relative hand mouse (frame-to-frame deltas)
    - VIRTUAL_PLANE: Legacy virtual display plane (absolute mapping)
    """

    def __init__(self, config: Optional[CursorConfig] = None, mode: CursorMode = CursorMode.VIRTUAL_PLANE):
        self.config = config or CursorConfig()
        self.mode = mode
        self._screen_width = self.config.screen_width
        self._screen_height = self.config.screen_height
        self._camera_width = self.config.camera_width
        self._camera_height = self.config.camera_height

        # Initialize the appropriate movement engine based on mode
        if mode == CursorMode.RELATIVE:
            self._init_relative_engine()
        else:
            self._init_virtual_plane_engine()

    def _init_relative_engine(self):
        """Initialize the relative movement engine (new system)."""
        # Convert CursorConfig to RelativeMouseConfig
        relative_config = RelativeMouseConfig(
            dead_zone_radius=self.config.dead_zone_radius,
            sensitivity_precision=self.config.sensitivity_precision,
            sensitivity_normal=self.config.sensitivity_normal,
            sensitivity_fast=self.config.sensitivity_fast,
            base_sensitivity=self.config.base_sensitivity,
            acceleration_exponent=self.config.acceleration,
            max_velocity_pixels=max(self.config.max_velocity, self.config.max_velocity_precision),
            smoothing=self.config.smoothing.value if isinstance(self.config.smoothing, SmoothingAlgorithm) else self.config.smoothing,
            one_euro_min_cutoff=self.config.one_euro_min_cutoff,
            one_euro_beta=self.config.one_euro_beta,
            one_euro_d_cutoff=self.config.one_euro_d_cutoff,
            ema_alpha=self.config.ema_alpha,
            tracking_loss_frames=getattr(self.config, 'tracking_loss_frames', 5),
            recovery_frames=getattr(self.config, 'recovery_frames', 3),
            stationary_threshold=getattr(self.config, 'stationary_threshold', 0.002),
            stationary_frames=getattr(self.config, 'stationary_frames', 10),
            auto_recenter_threshold=getattr(self.config, 'auto_recenter_threshold', 0.0),
            invert_x=self.config.invert_x,
            invert_y=self.config.invert_y,
            screen_width=self._screen_width,
            screen_height=self._screen_height,
            preferred_handedness="Right",
            min_hand_confidence=getattr(self.config, 'min_hand_confidence', 0.7),
        )
        self._relative_engine = RelativeMovementEngine(relative_config)
        self._relative_engine.set_sensitivity_mode(self.config.sensitivity_mode)
        
        # Legacy state (for compatibility)
        self._last_position: Optional[Tuple[float, float]] = None
        self._last_time: Optional[float] = None
        self._is_active = False
        self._reference_point: Optional[Tuple[float, float]] = None
        self._last_plane_position: Optional[Tuple[float, float]] = None
        self._accumulator_x: float = 0.0
        self._accumulator_y: float = 0.0

    def _init_virtual_plane_engine(self):
        """Initialize the virtual plane engine (legacy system)."""
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
        if self.mode == CursorMode.RELATIVE and hasattr(self, '_relative_engine'):
            self._relative_engine.config.screen_width = width
            self._relative_engine.config.screen_height = height

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
        if self.mode == CursorMode.RELATIVE:
            return self._map_hand_to_cursor_relative(hand)
        else:
            return self._map_hand_to_cursor_virtual_plane(hand)

    def _map_hand_to_cursor_relative(self, hand) -> Optional[Tuple[int, int]]:
        """
        Map hand to cursor using relative movement engine.
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

        # Convert to normalized coordinates relative to camera frame
        x_norm = ref_point.x
        y_norm = ref_point.y

        # Process through relative movement engine
        result = self._relative_engine.process_hand(x_norm, y_norm, hand.confidence)

        if not result.has_movement:
            return self._last_position

        # Convert relative movement to absolute screen position
        if self._last_position is None:
            # First frame - center cursor
            self._last_position = (self._screen_width // 2, self._screen_height // 2)
        
        new_x = self._last_position[0] + result.dx
        new_y = self._last_position[1] + result.dy

        # Clamp to screen bounds
        new_x = max(0, min(self._screen_width - 1, new_x))
        new_y = max(0, min(self._screen_height - 1, new_y))

        result_pos = (int(new_x), int(new_y))
        self._last_position = result_pos
        self._is_active = True

        return result_pos

    def _map_hand_to_cursor_virtual_plane(self, hand) -> Optional[Tuple[int, int]]:
        """
        Map hand to cursor using virtual plane (legacy system).
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
        if self.mode == CursorMode.RELATIVE:
            return self._get_relative_movement_relative(hand)
        else:
            return self._get_relative_movement_virtual_plane(hand)

    def _get_relative_movement_relative(self, hand) -> Optional[Tuple[int, int]]:
        """
        Get relative movement using relative engine.
        """
        if not hand or not hand.landmarks:
            return None

        if self.config.use_index_tip:
            ref_point = hand.index_tip
        else:
            ref_point = hand.palm_center

        if not ref_point:
            return None

        x_norm = ref_point.x
        y_norm = ref_point.y

        result = self._relative_engine.process_hand(x_norm, y_norm, hand.confidence)

        if not result.has_movement:
            return (0, 0)

        return (result.dx, result.dy)

    def _get_relative_movement_virtual_plane(self, hand) -> Optional[Tuple[int, int]]:
        """
        Get relative mouse movement (dx, dy) for uinput using virtual plane.
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
        Legacy virtual plane mode only.
        """
        if self.mode == CursorMode.RELATIVE:
            # In relative mode, this is not used - use get_relative_movement instead
            logger.warning("get_relative_movement_from_plane called in RELATIVE mode - not supported")
            return None

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
        
        if self.mode == CursorMode.RELATIVE and hasattr(self, '_relative_engine'):
            self._relative_engine.reset()
        else:
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
        if self.mode == CursorMode.RELATIVE and hasattr(self, '_relative_engine'):
            self._relative_engine.set_sensitivity_mode(mode)
        logger.info(f"Sensitivity mode changed to: {mode.value} (factor: {self.config.effective_sensitivity:.2f})")

    def get_sensitivity_mode(self) -> SensitivityMode:
        """Get current sensitivity mode."""
        return self.config.sensitivity_mode

    # New methods for relative mode
    def recenter(self, position: Optional[Tuple[float, float]] = None):
        """Recenter the relative movement reference point."""
        if self.mode == CursorMode.RELATIVE and hasattr(self, '_relative_engine'):
            self._relative_engine.recenter(position)
        else:
            logger.warning("recenter() only available in RELATIVE mode")

    def get_movement_stats(self) -> dict:
        """Get movement engine statistics (relative mode)."""
        if self.mode == CursorMode.RELATIVE and hasattr(self, '_relative_engine'):
            return self._relative_engine.get_stats()
        return {}

    def set_mode(self, mode: CursorMode):
        """Switch cursor control mode."""
        if mode != self.mode:
            self.mode = mode
            self.reset()
            if mode == CursorMode.RELATIVE:
                self._init_relative_engine()
            else:
                self._init_virtual_plane_engine()
            logger.info(f"Cursor mode changed to: {mode.value}")


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

    from airmouse.vision.hand_tracker import Hand, Landmark, HandLandmark

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

    # Test RELATIVE mode
    controller = CursorController(config, mode=CursorMode.RELATIVE)

    print("Testing CursorController (RELATIVE mode)...")
    for i in range(10):
        # Simulate hand moving right
        hand.index_tip.x = 0.5 + i * 0.02
        pos = controller.map_hand_to_cursor(hand)
        rel = controller.get_relative_movement(hand)
        print(f"Frame {i}: pos={pos}, rel={rel}")

    controller.reset()
    print("Test complete")