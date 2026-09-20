"""
Tracking Confidence System for Air Mouse

Implements explicit confidence states as defined in IDEA2.md §16:
- HIGH: Stable, reliable tracking
- MEDIUM: Acceptable but watch for issues
- LOW: Unreliable, degraded operation
- LOST: Hand not detected or tracking failed

Confidence influences:
- Cursor movement (stronger filtering in LOW state)
- Clicking (higher threshold for LOW state)
- Dragging (higher stability required)
- Gesture activation (confidence-gated)
"""

import time
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

logger = logging.getLogger(__name__)


class ConfidenceState(Enum):
    """Explicit confidence states for hand tracking (spec §16)."""
    HIGH = auto()    # Stable, reliable tracking
    MEDIUM = auto()  # Acceptable but watch for issues
    LOW = auto()     # Unreliable, degraded operation
    LOST = auto()    # Hand not detected or tracking failed


@dataclass
class ConfidenceInfo:
    """Tracking confidence information with evidence."""
    state: ConfidenceState = ConfidenceState.LOST
    confidence_value: float = 0.0  # Raw confidence (0.0-1.0)
    frames_in_state: int = 0
    last_change_time: float = field(default_factory=time.time)
    evidence: str = ""
    
    # Running statistics for adaptive thresholds
    _recent_confidences: list = field(default_factory=list)
    _consecutive_low: int = 0
    _consecutive_high: int = 0

    def update(self, raw_confidence: float, frame_count: int = 0, 
               evidence: str = "", update_time: Optional[float] = None) -> None:
        """Update confidence state based on new evidence."""
        if update_time is None:
            update_time = time.time()
        
        old_state = self.state
        self.confidence_value = raw_confidence
        self._recent_confidences.append(raw_confidence)
        
        # Keep window of recent values
        if len(self._recent_confidences) > 50:
            self._recent_confidences.pop(0)
        
        # Determine new state
        self.state = self._classify(raw_confidence, evidence)
        
        # Track consecutive counts
        if self.state == ConfidenceState.HIGH:
            self._consecutive_high += 1
            self._consecutive_low = 0
        elif self.state == ConfidenceState.LOW:
            self._consecutive_low += 1
            self._consecutive_high = 0
        
        # Update timestamp if state changed
        if old_state != self.state:
            self.frames_in_state = 1
            self.last_change_time = update_time
            self.evidence = evidence
        else:
            self.frames_in_state += 1
    
    def _classify(self, raw_confidence: float, evidence: str) -> ConfidenceState:
        """Classify raw confidence into explicit states."""
        if raw_confidence >= 0.85:
            return ConfidenceState.HIGH
        elif raw_confidence >= 0.50:
            return ConfidenceState.MEDIUM
        elif raw_confidence >= 0.15:
            return ConfidenceState.LOW
        else:
            return ConfidenceState.LOST
    
    def get_movement_safety_multiplier(self) -> float:
        """
        Get multiplier for cursor movement safety.
        
        Higher confidence = more responsive movement (lower multiplier)
        Lower confidence = more conservative movement (higher multiplier)
        """
        multipliers = {
            ConfidenceState.HIGH: 1.0,    # Normal responsiveness
            ConfidenceState.MEDIUM: 1.5,  # 50% more conservative
            ConfidenceState.LOW: 3.0,     # 200% more conservative
            ConfidenceState.LOST: float('inf'),  # Frozen - no movement allowed
        }
        return multipliers.get(self.state, 1.0)
    
    def get_click_threshold_multiplier(self) -> float:
        """
        Get multiplier for click activation thresholds.
        
        Higher confidence = more lenient click detection
        Lower confidence = stricter click detection
        """
        multipliers = {
            ConfidenceState.HIGH: 0.9,    # 10% stricter
            ConfidenceState.MEDIUM: 1.0,   # Normal
            ConfidenceState.LOW: 1.5,      # 50% stricter
            ConfidenceState.LOST: float('inf'),  # Frozen - no clicks
        }
        return multipliers.get(self.state, 1.0)
    
    def get_drag_stability_multiplier(self) -> float:
        """
        Get multiplier for drag stability.
        
        Lower confidence = more stable (less sensitive) drag
        """
        multipliers = {
            ConfidenceState.HIGH: 1.0,
            ConfidenceState.MEDIUM: 1.2,
            ConfidenceState.LOW: 2.0,
            ConfidenceState.LOST: float('inf'),
        }
        return multipliers.get(self.state, 1.0)
    
    def is_movement_allowed(self) -> bool:
        """Check if cursor movement is allowed."""
        return self.state != ConfidenceState.LOST
    
    def is_click_allowed(self) -> bool:
        """Check if clicks are allowed."""
        return self.state != ConfidenceState.LOST
    
    def is_gesture_allowed(self) -> bool:
        """Check if gestures are allowed."""
        return self.state in (ConfidenceState.HIGH, ConfidenceState.MEDIUM)
    
    def reset(self) -> None:
        """Reset confidence state."""
        self.__init__()


# Global confidence thresholds per spec §53
CONFIDENCE_THRESHOLDS = {
    "movement_primary": 0.30,    # Movement always allowed above this
    "click_primary": 0.50,        # Clicks require this minimum (adaptive)
    "click_strict": 0.75,         # Explicit confirmation click
    "drag_hold": 0.40,           # Drag requires this minimum
    "gesture_activation": 0.45,  # Gestures require this minimum
    "system_command": 0.85,      # System commands require high confidence
}


def get_confidence_state(raw_confidence: float,
                          recent_history: list = None,
                          evidence: str = "") -> ConfidenceState:
    """
    Get explicit confidence state from raw confidence value.
    
    Args:
        raw_confidence: Raw confidence value (0.0-1.0)
        recent_history: List of recent confidence values for trend analysis
        evidence: Human-readable evidence for the classification
    
    Returns:
        ConfidenceState classification
    """
    if recent_history is None:
        recent_history = []
    
    # Use trend if we have history
    if len(recent_history) >= 3:
        avg_recent = sum(recent_history[-5:]) / min(len(recent_history[-5:]), 1)
        trend = avg_recent - recent_history[-1] if len(recent_history) >= 2 else 0
        
        if raw_confidence >= 0.7 and trend > 0.1:
            return ConfidenceState.HIGH
        elif raw_confidence >= 0.5:
            return ConfidenceState.MEDIUM
    
    # Static classification
    if raw_confidence >= 0.85:
        return ConfidenceState.HIGH
    elif raw_confidence >= 0.50:
        return ConfidenceState.MEDIUM
    elif raw_confidence >= 0.15:
        return ConfidenceState.LOW
    else:
        return ConfidenceState.LOST
