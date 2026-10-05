"""
Relative Hand Mouse Movement Engine

Replaces virtual-display cursor control with a physical-mouse-like relative movement system.

Key features:
- Frame-to-frame hand landmark deltas drive cursor movement (like a physical mouse)
- Dead zone for stationary hand stabilization
- One Euro Filter for low-latency adaptive smoothing
- Velocity-based acceleration curve (like physical mouse)
- Reference position tracking with re-centering support
- Tracking loss detection and cursor jump prevention
- Sensitivity modes: Precision, Normal, Fast
- Configurable for low-end hardware performance

Design principles:
- No virtual plane, no head coordinate system, no ray projection
- Operates directly on normalized camera coordinates (0-1) from MediaPipe
- Frame-to-frame delta = relative movement (exactly like a physical mouse)
- Reference position resets on explicit re-center or tracking loss recovery
"""

import time
import math
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple
from enum import Enum

from .smoothing import OneEuroFilter, EmaFilter, SensitivityMode

logger = logging.getLogger(__name__)


class MovementState(Enum):
    """State of the relative movement engine."""
    IDLE = "idle"                    # No hand detected, waiting for first hand
    TRACKING = "tracking"            # Actively tracking hand movement
    RECOVERING = "recovering"        # Hand reacquired after loss, stabilizing
    RECENTERING = "recentering"      # Explicit re-center in progress


@dataclass
class RelativeMouseConfig:
    """Configuration for relative hand mouse movement."""
    
    # Dead zone radius (normalized camera coordinates)
    # Movements smaller than this produce zero cursor movement
    dead_zone_radius: float = 0.015
    
    # Sensitivity modes (multipliers for base sensitivity)
    sensitivity_precision: float = 0.12   # Reduced by 20% from 0.15
    sensitivity_normal: float = 0.32      # Reduced by 20% from 0.40
    sensitivity_fast: float = 0.48        # Reduced by 20% from 0.60
    
    # Base sensitivity (applied after mode multiplier)
    base_sensitivity: float = 1.0
    
    # Acceleration exponent (>1.0 = accelerated, 1.0 = linear, <1.0 = decelerated)
    acceleration_exponent: float = 1.2
    
    # Maximum velocity in pixels per frame (clamped)
    max_velocity_pixels: float = 500.0
    
    # Smoothing algorithm: "one_euro" (adaptive), "ema" (exponential), "none"
    smoothing: str = "one_euro"
    
    # One Euro Filter parameters (for adaptive smoothing) — tuned for low latency
    one_euro_min_cutoff: float = 0.5
    one_euro_beta: float = 0.08
    one_euro_d_cutoff: float = 0.5
    
    # EMA smoothing alpha (if smoothing="ema")
    ema_alpha: float = 0.3
    
    # Tracking loss detection
    # Frames without hand before considering tracking lost
    tracking_loss_frames: int = 5
    
    # Recovery frames - how many frames to stabilize after reacquisition
    recovery_frames: int = 3
    
    # Stationary hand detection
    # If hand movement < this threshold for this many frames, apply extra smoothing
    stationary_threshold: float = 0.002
    stationary_frames: int = 10
    
    # Re-centering
    # Distance from center (0.5, 0.5) to trigger auto-recenter (0 = disabled)
    auto_recenter_threshold: float = 0.0
    
    # Invert axes
    invert_x: bool = False
    invert_y: bool = False
    
    # Screen dimensions for velocity clamping
    screen_width: int = 1920
    screen_height: int = 1080
    
    # Preferred handedness
    preferred_handedness: str = "Right"
    
    # Confidence threshold for accepting hand data
    min_hand_confidence: float = 0.7


@dataclass
class MovementResult:
    """Result of a movement computation."""
    # Pixel movement this frame (dx, dy)
    dx: int = 0
    dy: int = 0
    
    # Whether movement was produced (False = dead zone or no valid input)
    has_movement: bool = False
    
    # Current movement state
    state: MovementState = MovementState.IDLE
    
    # Debug info
    raw_delta_x: float = 0.0
    raw_delta_y: float = 0.0
    smoothed_delta_x: float = 0.0
    smoothed_delta_y: float = 0.0
    velocity: float = 0.0
    confidence: float = 0.0
    frames_since_hand: int = 0


class RelativeMovementEngine:
    """
    Core relative movement engine for hand-controlled mouse.
    
    Operates like a physical mouse:
    - Frame-to-frame hand position delta = cursor movement
    - Dead zone prevents jitter when hand is stationary
    - Adaptive smoothing (One Euro Filter) for low latency + stability
    - Velocity-based acceleration curve
    - Reference position resets on re-center or tracking recovery
    """
    
    def __init__(self, config: Optional[RelativeMouseConfig] = None):
        self.config = config or RelativeMouseConfig()
        
        # State
        self._state = MovementState.IDLE
        self._reference_position: Optional[Tuple[float, float]] = None
        self._last_position: Optional[Tuple[float, float]] = None
        self._last_valid_position: Optional[Tuple[float, float]] = None
        
        # Smoothing filters (one per axis)
        self._x_filter: Optional[OneEuroFilter] = None
        self._y_filter: Optional[OneEuroFilter] = None
        self._x_ema: Optional[EmaFilter] = None
        self._y_ema: Optional[EmaFilter] = None
        
        # Tracking loss / recovery
        self._frames_without_hand: int = 0
        self._recovery_frame_count: int = 0
        
        # Stationary detection
        self._stationary_frame_count: int = 0
        self._is_stationary: bool = False
        
        # Sensitivity mode
        self._sensitivity_mode = SensitivityMode.NORMAL
        self._current_sensitivity: float = self.config.sensitivity_normal * self.config.base_sensitivity
        
        # Stats
        self._total_frames: int = 0
        self._tracking_frames: int = 0
        self._lost_tracking_count: int = 0
        
        # Initialize filters
        self._init_filters()
        
        logger.info(f"RelativeMovementEngine initialized: dead_zone={self.config.dead_zone_radius}, "
                   f"smoothing={self.config.smoothing}, sensitivity={self._current_sensitivity}")
    
    def _init_filters(self):
        """Initialize smoothing filters based on config."""
        if self.config.smoothing == "one_euro":
            self._x_filter = OneEuroFilter(
                min_cutoff=self.config.one_euro_min_cutoff,
                beta=self.config.one_euro_beta,
                d_cutoff=self.config.one_euro_d_cutoff
            )
            self._y_filter = OneEuroFilter(
                min_cutoff=self.config.one_euro_min_cutoff,
                beta=self.config.one_euro_beta,
                d_cutoff=self.config.one_euro_d_cutoff
            )
        elif self.config.smoothing == "ema":
            self._x_ema = EmaFilter(alpha=self.config.ema_alpha)
            self._y_ema = EmaFilter(alpha=self.config.ema_alpha)
    
    @property
    def state(self) -> MovementState:
        return self._state
    
    @property
    def sensitivity_mode(self) -> SensitivityMode:
        return self._sensitivity_mode
    
    @property
    def reference_position(self) -> Optional[Tuple[float, float]]:
        return self._reference_position
    
    @property
    def is_tracking(self) -> bool:
        return self._state == MovementState.TRACKING
    
    def set_sensitivity_mode(self, mode: SensitivityMode):
        """Change sensitivity mode at runtime."""
        self._sensitivity_mode = mode
        if mode == SensitivityMode.PRECISION:
            self._current_sensitivity = self.config.sensitivity_precision * self.config.base_sensitivity
        elif mode == SensitivityMode.FAST:
            self._current_sensitivity = self.config.sensitivity_fast * self.config.base_sensitivity
        else:
            self._current_sensitivity = self.config.sensitivity_normal * self.config.base_sensitivity
        logger.info(f"Sensitivity mode changed to {mode.name}: {self._current_sensitivity}")
    
    def set_base_sensitivity(self, sensitivity: float):
        """Set base sensitivity multiplier."""
        self.config.base_sensitivity = max(0.1, min(5.0, sensitivity))
        self.set_sensitivity_mode(self._sensitivity_mode)  # Recalculate
    
    def recenter(self, position: Optional[Tuple[float, float]] = None):
        """
        Reset reference position for relative movement.
        
        Args:
            position: New reference position (normalized 0-1). If None, uses current hand position.
        """
        if position is not None:
            self._reference_position = position
            self._last_position = position
            self._last_valid_position = position
        elif self._last_valid_position is not None:
            self._reference_position = self._last_valid_position
            self._last_position = self._last_valid_position
        else:
            # Default to center if no position available
            self._reference_position = (0.5, 0.5)
            self._last_position = (0.5, 0.5)
        
        # Reset filters on recenter
        self._reset_filters()
        self._state = MovementState.RECENTERING
        logger.info(f"Recentered to {self._reference_position}")
    
    def _reset_filters(self):
        """Reset smoothing filters."""
        if self._x_filter:
            self._x_filter.reset()
        if self._y_filter:
            self._y_filter.reset()
        if self._x_ema:
            self._x_ema.reset()
        if self._y_ema:
            self._y_ema.reset()
    
    def process_hand(self, hand_x: float, hand_y: float, confidence: float) -> MovementResult:
        """
        Process a hand position and compute cursor movement.
        
        Args:
            hand_x: Normalized hand X position (0-1) in camera coordinates
            hand_y: Normalized hand Y position (0-1) in camera coordinates
            confidence: Hand detection confidence (0-1)
            
        Returns:
            MovementResult with pixel movement and state info
        """
        self._total_frames += 1
        result = MovementResult()
        result.confidence = confidence
        
        # Validate input
        if not (0.0 <= hand_x <= 1.0 and 0.0 <= hand_y <= 1.0):
            result.state = self._state
            result.frames_since_hand = self._frames_without_hand
            return result
        
        if confidence < self.config.min_hand_confidence:
            # Low confidence - treat as no hand
            return self._process_no_hand(result)
        
        # Hand detected with sufficient confidence
        self._frames_without_hand = 0
        self._tracking_frames += 1
        
        # Initialize reference on first valid hand
        if self._reference_position is None:
            self._reference_position = (hand_x, hand_y)
            self._last_position = (hand_x, hand_y)
            self._last_valid_position = (hand_x, hand_y)
            self._state = MovementState.TRACKING
            self._reset_filters()
            result.state = self._state
            return result  # First frame = no movement
        
        # Check if recovering from tracking loss
        if self._state == MovementState.RECOVERING:
            self._recovery_frame_count += 1
            if self._recovery_frame_count >= self.config.recovery_frames:
                self._state = MovementState.TRACKING
                self._reference_position = (hand_x, hand_y)  # Reset reference to current position
                self._last_position = (hand_x, hand_y)
                self._reset_filters()
                logger.debug("Tracking recovered, reference reset")
            else:
                # During recovery, don't produce movement but update last position
                self._last_position = (hand_x, hand_y)
                self._last_valid_position = (hand_x, hand_y)
                result.state = self._state
                return result
        
        # Check if recentered - transition to TRACKING on next valid frame
        if self._state == MovementState.RECENTERING:
            self._state = MovementState.TRACKING
            # Reference already set in recenter(), last_position already set
            # Just continue to compute movement from new reference
        
        # Compute frame-to-frame delta (relative movement like physical mouse)
        raw_dx = hand_x - self._last_position[0]
        raw_dy = hand_y - self._last_position[1]
        
        result.raw_delta_x = raw_dx
        result.raw_delta_y = raw_dy
        
        # Apply dead zone
        delta_magnitude = math.hypot(raw_dx, raw_dy)
        
        if delta_magnitude < self.config.dead_zone_radius:
            # Inside dead zone - no movement, but update last position for next frame
            self._last_position = (hand_x, hand_y)
            self._last_valid_position = (hand_x, hand_y)
            
            # Track stationary state
            self._stationary_frame_count += 1
            if self._stationary_frame_count >= self.config.stationary_frames:
                self._is_stationary = True
            
            result.state = self._state
            result.has_movement = False
            return result
        
        # Outside dead zone - movement detected
        self._stationary_frame_count = 0
        self._is_stationary = False
        
        # Apply smoothing
        current_time = time.monotonic()
        
        if self.config.smoothing == "one_euro" and self._x_filter and self._y_filter:
            smoothed_dx = self._x_filter.filter(raw_dx, current_time)
            smoothed_dy = self._y_filter.filter(raw_dy, current_time)
        elif self.config.smoothing == "ema" and self._x_ema and self._y_ema:
            smoothed_dx = self._x_ema.filter(raw_dx)
            smoothed_dy = self._y_ema.filter(raw_dy)
        else:
            # No smoothing
            smoothed_dx = raw_dx
            smoothed_dy = raw_dy
        
        result.smoothed_delta_x = smoothed_dx
        result.smoothed_delta_y = smoothed_dy
        
        # Apply sensitivity and acceleration
        # Velocity = magnitude of smoothed delta
        velocity = math.hypot(smoothed_dx, smoothed_dy)
        result.velocity = velocity
        
        # Acceleration curve: velocity^exponent
        if velocity > 0:
            acceleration_factor = velocity ** (self.config.acceleration_exponent - 1.0)
        else:
            acceleration_factor = 1.0
        
        # Apply sensitivity
        scaled_dx = smoothed_dx * self._current_sensitivity * acceleration_factor
        scaled_dy = smoothed_dy * self._current_sensitivity * acceleration_factor
        
        # Apply axis inversion
        if self.config.invert_x:
            scaled_dx = -scaled_dx
        if self.config.invert_y:
            scaled_dy = -scaled_dy
        
        # Convert to pixels
        pixel_dx = scaled_dx * self.config.screen_width
        pixel_dy = scaled_dy * self.config.screen_height
        
        # Clamp to max velocity
        pixel_velocity = math.hypot(pixel_dx, pixel_dy)
        if pixel_velocity > self.config.max_velocity_pixels:
            scale = self.config.max_velocity_pixels / pixel_velocity
            pixel_dx *= scale
            pixel_dy *= scale
        
        # Convert to integers
        result.dx = int(round(pixel_dx))
        result.dy = int(round(pixel_dy))
        result.has_movement = (result.dx != 0 or result.dy != 0)
        result.state = self._state
        
        # Update positions
        self._last_position = (hand_x, hand_y)
        self._last_valid_position = (hand_x, hand_y)
        
        return result
    
    def _process_no_hand(self, result: MovementResult) -> MovementResult:
        """Handle frames without valid hand detection."""
        self._frames_without_hand += 1
        result.frames_since_hand = self._frames_without_hand
        
        if self._state == MovementState.TRACKING:
            if self._frames_without_hand >= self.config.tracking_loss_frames:
                self._state = MovementState.RECOVERING
                self._recovery_frame_count = 0
                self._lost_tracking_count += 1
                logger.debug(f"Tracking lost after {self._frames_without_hand} frames without hand")
        
        result.state = self._state
        result.has_movement = False
        return result
    
    def force_recovery(self):
        """Force recovery state (e.g., after explicit user action)."""
        if self._state != MovementState.TRACKING:
            self._state = MovementState.RECOVERING
            self._recovery_frame_count = 0
            logger.info("Forced recovery state")
    
    def get_stats(self) -> dict:
        """Get engine statistics."""
        return {
            "state": self._state.value,
            "total_frames": self._total_frames,
            "tracking_frames": self._tracking_frames,
            "lost_tracking_count": self._lost_tracking_count,
            "frames_without_hand": self._frames_without_hand,
            "is_stationary": self._is_stationary,
            "stationary_frame_count": self._stationary_frame_count,
            "reference_position": self._reference_position,
            "last_position": self._last_position,
            "sensitivity_mode": self._sensitivity_mode.name,
            "current_sensitivity": self._current_sensitivity,
        }
    
    def reset(self):
        """Full reset of the engine."""
        self._state = MovementState.IDLE
        self._reference_position = None
        self._last_position = None
        self._last_valid_position = None
        self._frames_without_hand = 0
        self._recovery_frame_count = 0
        self._stationary_frame_count = 0
        self._is_stationary = False
        self._total_frames = 0
        self._tracking_frames = 0
        self._lost_tracking_count = 0
        self._reset_filters()
        logger.info("RelativeMovementEngine reset")


def create_default_config() -> RelativeMouseConfig:
    """Create default configuration for relative mouse."""
    return RelativeMouseConfig()


# Backwards compatibility alias
RelativeMouseEngine = RelativeMovementEngine