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


class SmoothingAlgorithm(Enum):
    """Available smoothing algorithms."""
    NONE = "none"
    EMA = "ema"           # Exponential Moving Average
    ONE_EURO = "one_euro"  # One Euro Filter (better for varying speeds)


class SensitivityMode(Enum):
    """Cursor sensitivity modes."""
    PRECISION = "precision"  # 15% sensitivity - fine control
    NORMAL = "normal"        # 40% sensitivity - default
    FAST = "fast"            # 60% sensitivity - quick navigation


@dataclass
class CursorConfig:
    """Configuration for cursor mapping and smoothing."""
    # Screen dimensions (will be auto-detected if not set)
    screen_width: int = 1920
    screen_height: int = 1080

    # Camera frame dimensions (for normalization)
    camera_width: int = 640
    camera_height: int = 480

    # Dead zone: ignore small movements around center (0.0 to 1.0 normalized)
    dead_zone_radius: float = 0.02

    # Sensitivity mode (overrides base_sensitivity)
    sensitivity_mode: SensitivityMode = SensitivityMode.NORMAL

    # Base sensitivity (1.0 = 1:1 mapping) - multiplied by mode factor
    base_sensitivity: float = 1.0

    # Sensitivity multipliers for each mode
    sensitivity_precision: float = 0.15  # 15%
    sensitivity_normal: float = 0.40     # 40% (default)
    sensitivity_fast: float = 0.60       # 60%

    # Acceleration curve: 1.0 = linear, >1.0 = accelerated
    acceleration: float = 1.2

    # Smoothing algorithm
    smoothing: SmoothingAlgorithm = SmoothingAlgorithm.ONE_EURO

    # EMA alpha (0.0 to 1.0, lower = more smoothing)
    ema_alpha: float = 0.3

    # One Euro Filter parameters
    one_euro_min_cutoff: float = 1.0
    one_euro_beta: float = 0.0
    one_euro_d_cutoff: float = 1.0

    # Invert axes if needed
    invert_x: bool = False
    invert_y: bool = False

    # Use index finger tip (True) or palm center (False) as cursor point
    use_index_tip: bool = True

    @property
    def effective_sensitivity(self) -> float:
        """Get effective sensitivity based on current mode."""
        mode_factors = {
            SensitivityMode.PRECISION: self.sensitivity_precision,
            SensitivityMode.NORMAL: self.sensitivity_normal,
            SensitivityMode.FAST: self.sensitivity_fast,
        }
        return self.base_sensitivity * mode_factors.get(self.sensitivity_mode, self.sensitivity_normal)


class OneEuroFilter:
    """
    One Euro Filter for smoothing with adaptive cutoff frequency.
    Based on: https://cristal.univ-lille.fr/~casiez/1euro/
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: Optional[float] = None
        self.dx_prev: Optional[float] = None
        self.t_prev: Optional[float] = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        """Compute alpha for exponential smoothing."""
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, t: Optional[float] = None) -> float:
        """Filter a value with timestamp."""
        if t is None:
            t = time.time()

        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x

        dt = t - self.t_prev
        if dt <= 0:
            dt = 1e-3

        # Estimate derivative
        dx = (x - self.x_prev) / dt

        # Filter derivative
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        # Compute adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # Filter signal
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev

        # Update state
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat

    def reset(self):
        """Reset filter state."""
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None


class EMASmoother:
    """Exponential Moving Average smoother."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.value: Optional[float] = None

    def smooth(self, x: float) -> float:
        if self.value is None:
            self.value = x
            return x
        self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.value = None


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

        # Smoothers for X and Y
        if self.config.smoothing == SmoothingAlgorithm.ONE_EURO:
            self._smoother_x = OneEuroFilter(
                min_cutoff=self.config.one_euro_min_cutoff,
                beta=self.config.one_euro_beta,
                d_cutoff=self.config.one_euro_d_cutoff
            )
            self._smoother_y = OneEuroFilter(
                min_cutoff=self.config.one_euro_min_cutoff,
                beta=self.config.one_euro_beta,
                d_cutoff=self.config.one_euro_d_cutoff
            )
        else:
            self._smoother_x = EMASmoother(alpha=self.config.ema_alpha)
            self._smoother_y = EMASmoother(alpha=self.config.ema_alpha)

        # State
        self._last_position: Optional[Tuple[float, float]] = None
        self._last_time: Optional[float] = None
        self._is_active = False
        self._reference_point: Optional[Tuple[float, float]] = None  # Initial hand position

    def update_screen_size(self, width: int, height: int):
        """Update screen dimensions."""
        self._screen_width = width
        self._screen_height = height

    def update_camera_size(self, width: int, height: int):
        """Update camera frame dimensions."""
        self._camera_width = width
        self._camera_height = height

    def _normalize_to_screen(self, x_norm: float, y_norm: float) -> Tuple[float, float]:
        """Convert normalized coordinates (0-1) to screen pixels."""
        # Apply inversion if needed
        if self.config.invert_x:
            x_norm = 1.0 - x_norm
        if self.config.invert_y:
            y_norm = 1.0 - y_norm

        # Map to screen
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
        """Apply acceleration curve to movement."""
        if self.config.acceleration == 1.0:
            return (dx, dy)

        distance = math.sqrt(dx * dx + dy * dy)
        if distance == 0:
            return (0.0, 0.0)

        # Apply power curve for acceleration
        factor = distance ** (self.config.acceleration - 1.0)
        return (dx * factor, dy * factor)

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

        current_time = time.time()

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

        # Convert to screen coordinates
        screen_x, screen_y = self._normalize_to_screen(x_norm, y_norm)

        # Apply smoothing
        if self.config.smoothing != SmoothingAlgorithm.NONE:
            screen_x = self._smoother_x.smooth(screen_x)
            screen_y = self._smoother_y.smooth(screen_y)

        # Convert to integers
        result = (int(screen_x), int(screen_y))

        self._last_position = result
        self._last_time = current_time
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

        return (dx, dy)

    def get_relative_movement_from_plane(self, x_norm: float, y_norm: float) -> Optional[Tuple[int, int]]:
        """
        Get relative mouse movement from normalized plane coordinates (head-relative mode).

        Args:
            x_norm: Normalized X position on virtual plane (0-1)
            y_norm: Normalized Y position on virtual plane (0-1)

        Returns:
            (dx, dy) relative movement in screen pixels for uinput, or None
        """
        # Initialize reference point on first frame
        if self._reference_point is None:
            self._reference_point = (x_norm, y_norm)
            return (0, 0)  # No movement on first frame

        # Calculate delta from reference point
        dx = x_norm - self._reference_point[0]
        dy = y_norm - self._reference_point[1]

        # Apply dead zone
        dx, dy = self._apply_dead_zone(dx, dy)
        if dx == 0.0 and dy == 0.0:
            return None

        # Apply sensitivity and acceleration
        dx, dy = self._apply_sensitivity(dx, dy)
        dx, dy = self._apply_acceleration(dx, dy)

        # Convert normalized movement to screen pixels
        # Normalized plane coords [0,1] map to screen dimensions
        screen_dx = int(dx * self._screen_width)
        screen_dy = int(dy * self._screen_height)

        return (screen_dx, screen_dy)

    def reset(self):
        """Reset controller state."""
        self._last_position = None
        self._last_time = None
        self._is_active = False
        self._reference_point = None
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