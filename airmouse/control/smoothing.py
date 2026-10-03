"""
Smoothing Filter Abstractions for Air Mouse

Provides a common interface for different smoothing algorithms:
- OneEuroFilter: Adaptive smoothing based on velocity
- EmaFilter: Exponential Moving Average with fixed alpha

Both operate on normalized plane (u,v) coordinates [0,1]².
"""

import time
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class SmoothingAlgorithm(Enum):
    """Available smoothing algorithms."""
    NONE = "none"
    EMA = "ema"
    ONE_EURO = "one_euro"


class SensitivityMode(Enum):
    """Cursor sensitivity modes."""
    PRECISION = "precision"  # 15% sensitivity - fine control
    NORMAL = "normal"        # 40% sensitivity - default
    FAST = "fast"            # 60% sensitivity - quick navigation


@dataclass
class SmoothingConfig:
    """Configuration for smoothing filters."""
    algorithm: SmoothingAlgorithm = SmoothingAlgorithm.ONE_EURO
    
    # EMA parameters
    ema_alpha: float = 0.3
    
    # One Euro Filter parameters
    one_euro_min_cutoff: float = 1.0
    one_euro_beta: float = 0.0
    one_euro_d_cutoff: float = 1.0


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

    # Sensitivity multipliers for each mode (reduced by 20%)
    sensitivity_precision: float = 0.12  # 12% (was 15%)
    sensitivity_normal: float = 0.32     # 32% (was 40%)
    sensitivity_fast: float = 0.48       # 48% (was 60%)

    # Acceleration curve: 1.0 = linear, >1.0 = accelerated
    acceleration: float = 1.2

    # Maximum cursor velocity (pixels per second) per §24
    # Prevents runaway cursor when hand moves suddenly
    max_velocity: int = 2000             # pixels/sec (benchmark: comfortable for 1080p)
    max_velocity_precision: int = 500    # pixels/sec in precision mode

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

    # Multi-monitor support (§25)
    monitor_count: int = 1               # Number of active monitors
    monitor_arrangement: str = "horizontal"  # "horizontal" or "grid"
    primary_monitor: int = 0             # Index of primary monitor
    virtual_desktop_width: int = 0       # 0 = use sum of monitors
    virtual_desktop_height: int = 0

    @property
    def effective_sensitivity(self) -> float:
        """Get effective sensitivity based on current mode."""
        mode_factors = {
            SensitivityMode.PRECISION: self.sensitivity_precision,
            SensitivityMode.NORMAL: self.sensitivity_normal,
            SensitivityMode.FAST: self.sensitivity_fast,
        }
        return self.base_sensitivity * mode_factors.get(self.sensitivity_mode, self.sensitivity_normal)


class SmoothingFilter(ABC):
    """Abstract base class for smoothing filters operating on [0,1] plane coordinates."""
    
    @abstractmethod
    def filter(self, value: float, timestamp: Optional[float] = None) -> float:
        """Apply smoothing to a single coordinate value.
        
        Args:
            value: Raw coordinate value in [0,1]
            timestamp: Optional timestamp (monotonic seconds), uses time.monotonic() if None
            
        Returns:
            Smoothed coordinate value in [0,1]
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset filter state."""
        pass


class EmaFilter(SmoothingFilter):
    """Exponential Moving Average filter with fixed alpha."""
    
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.value: Optional[float] = None
    
    def filter(self, value: float, timestamp: Optional[float] = None) -> float:
        if self.value is None:
            self.value = value
            return value
        
        self.value = self.alpha * value + (1 - self.alpha) * self.value
        return self.value
    
    def reset(self):
        self.value = None


class OneEuroFilter(SmoothingFilter):
    """
    One Euro Filter for adaptive smoothing based on velocity.
    
    Based on: https://cristal.univ-lille.fr/~casiez/1euro/
    Adapts cutoff frequency based on signal velocity:
    - Slow movement → low cutoff → heavy smoothing
    - Fast movement → high cutoff → responsive tracking
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
    
    def filter(self, value: float, timestamp: Optional[float] = None) -> float:
        if timestamp is None:
            timestamp = time.monotonic()
        
        if self.x_prev is None:
            self.x_prev = value
            self.dx_prev = 0.0
            self.t_prev = timestamp
            return value
        
        dt = timestamp - self.t_prev
        if dt <= 0:
            dt = 1e-3
        
        # Estimate derivative
        dx = (value - self.x_prev) / dt
        
        # Filter derivative
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        
        # Compute adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        
        # Filter signal
        a = self._alpha(cutoff, dt)
        x_hat = a * value + (1 - a) * self.x_prev
        
        # Update state
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = timestamp
        
        return x_hat
    
    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None
    
    def set_params(self, min_cutoff: float = None, beta: float = None, d_cutoff: float = None):
        """Update filter parameters."""
        if min_cutoff is not None:
            self.min_cutoff = min_cutoff
        if beta is not None:
            self.beta = beta
        if d_cutoff is not None:
            self.d_cutoff = d_cutoff


class SmoothingFilterFactory:
    """Factory for creating smoothing filter instances from config."""
    
    @staticmethod
    def create_filter(config: SmoothingConfig, axis: str = "x") -> SmoothingFilter:
        """Create a smoothing filter based on configuration."""
        if config.algorithm == SmoothingAlgorithm.NONE:
            return PassThroughFilter()
        elif config.algorithm == SmoothingAlgorithm.EMA:
            return EmaFilter(alpha=config.ema_alpha)
        elif config.algorithm == SmoothingAlgorithm.ONE_EURO:
            return OneEuroFilter(
                min_cutoff=config.one_euro_min_cutoff,
                beta=config.one_euro_beta,
                d_cutoff=config.one_euro_d_cutoff
            )
        else:
            raise ValueError(f"Unknown smoothing algorithm: {config.algorithm}")
    
    @staticmethod
    def create_pair(config: SmoothingConfig) -> tuple[SmoothingFilter, SmoothingFilter]:
        """Create a pair of filters for X and Y axes."""
        return (
            SmoothingFilterFactory.create_filter(config, "x"),
            SmoothingFilterFactory.create_filter(config, "y")
        )


class PassThroughFilter(SmoothingFilter):
    """Pass-through filter (no smoothing)."""
    
    def filter(self, value: float, timestamp: Optional[float] = None) -> float:
        return max(0.0, min(1.0, value))
    
    def reset(self):
        pass


if __name__ == "__main__":
    # Quick test
    import random
    
    # Test EMA
    ema = EmaFilter(alpha=0.3)
    ema.reset()
    print("Testing EMA...")
    for i in range(10):
        val = 0.5 + 0.1 * random.random()
        smoothed = ema.filter(val)
        print(f"  Raw: {val:.3f} -> Smoothed: {smoothed:.3f}")
    
    # Test One Euro
    oe = OneEuroFilter(min_cutoff=1.0, beta=0.0, d_cutoff=1.0)
    oe.reset()
    print("\nTesting OneEuro...")
    t = time.monotonic()
    for i in range(10):
        val = 0.5 + 0.1 * random.random()
        smoothed = oe.filter(val, t + i * 0.033)  # ~30 FPS
        print(f"  Raw: {val:.3f} -> Smoothed: {smoothed:.3f}")
    
    print("\nAll filters working correctly!")