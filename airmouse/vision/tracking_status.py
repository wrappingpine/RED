"""
Tracking Loss Safety System for Air Mouse

Implements spec §17 and §18:
- When hand is lost: freeze pointer, stop input, no gestures
- When hand returns: detect, verify confidence, stabilize, resume

Behavior per §17:
    Hand lost
       ↓
    Stop pointer updates
       ↓
    Stop gesture recognition
       ↓
    Do NOT generate clicks
       ↓
    Do NOT generate keyboard events
       ↓
    Wait for stable reacquisition

Per §18 Reacquisition:
    1. Detect the hand
    2. Verify confidence
    3. Stabilize tracking
    4. Establish a new motion baseline
    5. Resume control

The cursor must not jump because the hand reappeared at a different camera position.
"""

import time
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable

from .confidence import ConfidenceInfo, ConfidenceState, get_confidence_state

logger = logging.getLogger(__name__)


class TrackingPhase(Enum):
    """Phases of the tracking lifecycle."""
    STARTING = auto()      # Initial startup or after cold start
    TRACKING = auto()      # Normal tracking operation
    DEGRADED = auto()      # Low confidence but still functional
    FROZEN = auto()        # Tracking paused (e.g., fist gesture)
    LOST = auto()          # Hand lost - freeze everything
    REACQUIRING = auto()  # Hand detected but not yet stable
    STABILIZING = auto()   # Post-reacquisition stabilization


class LostReason(Enum):
    """Reasons tracking was lost."""
    NONE = auto()
    NO_HAND_DETECTED = auto()
    LOW_CONFIDENCE = auto()
    TRACKING_JUMP = auto()
    CONFIDENCE_DROP = auto()
    MANUAL_PAUSE = auto()
    ERROR = auto()


@dataclass
class TrackingStatus:
    """Complete tracking status including loss detection and reacquisition state."""
    phase: TrackingPhase = TrackingPhase.STARTING
    lost_reason: LostReason = LostReason.NONE
    confidence: ConfidenceInfo = field(default_factory=ConfidenceInfo)
    last_lost_time: Optional[float] = None
    lost_duration: float = 0.0
    frames_since_loss: int = 0
    reacquisition_attempts: int = 0
    last_reacquisition_time: Optional[float] = None
    stabilization_frames: int = 0
    is_frozen: bool = False
    
    # Callbacks for state changes
    _on_tracking_lost: Optional[Callable[[LostReason], None]] = None
    _on_tracking_recovered: Optional[Callable[[], None]] = None
    _on_phase_change: Optional[Callable[[TrackingPhase, TrackingPhase], None]] = None
    
    def set_callbacks(self,
                      on_lost: Optional[Callable[[LostReason], None]] = None,
                      on_recovered: Optional[Callable[[], None]] = None,
                      on_phase_change: Optional[Callable[[TrackingPhase, TrackingPhase], None]] = None):
        """Set state change callbacks."""
        self._on_tracking_lost = on_lost
        self._on_tracking_recovered = on_recovered
        self._on_phase_change = on_phase_change
    
    def record_hand_detected(self, raw_confidence: float, evidence: str = "") -> ConfidenceState:
        """Record a hand detection event."""
        state = get_confidence_state(raw_confidence, evidence=evidence)
        self.confidence.update(raw_confidence, evidence=evidence)
        return state
    
    def record_hand_lost(self, reason: LostReason, evidence: str = "") -> None:
        """
        Record that hand was lost.
        
        Per spec §17: freeze pointer, stop gestures, no input.
        """
        old_phase = self.phase
        
        self.phase = TrackingPhase.LOST
        self.lost_reason = reason
        self.last_lost_time = time.time()
        self.lost_duration = 0.0
        self.frames_since_loss = 0
        self.reacquisition_attempts = 0
        
        if self._on_tracking_lost:
            try:
                self._on_tracking_lost(reason)
            except Exception as e:
                logger.error(f"Error in tracking_lost callback: {e}")
        
        if self._on_phase_change and old_phase != self.phase:
            try:
                self._on_phase_change(old_phase, self.phase)
            except Exception as e:
                logger.error(f"Error in phase_change callback: {e}")
        
        logger.info(f"Tracking LOST: reason={reason.name}, confidence={self.confidence.confidence_value:.2f}")
    
    def record_reacquisition_start(self) -> None:
        """
        Record start of reacquisition sequence (per §18 step 1).
        
        Per §18:
        1. Detect the hand
        2. Verify confidence
        3. Stabilize tracking
        4. Establish a new motion baseline
        5. Resume control
        """
        old_phase = self.phase
        self.phase = TrackingPhase.REACQUIRING
        self.reacquisition_attempts += 1
        self.stabilization_frames = 0
        
        logger.info(f"Reacquisition started (attempt {self.reacquisition_attempts})")
        
        if self._on_phase_change and old_phase != self.phase:
            try:
                self._on_phase_change(old_phase, self.phase)
            except Exception as e:
                logger.error(f"Error in phase_change callback: {e}")
    
    def record_confidence_verified(self, raw_confidence: float) -> bool:
        """
        Record confidence verification (per §18 step 2).
        
        Returns True if confidence is acceptable for resume.
        """
        state = self.record_hand_detected(raw_confidence, evidence="reacquisition_verification")
        
        # For reacquisition, require at least MEDIUM confidence
        acceptable = state in (ConfidenceState.HIGH, ConfidenceState.MEDIUM)
        
        logger.info(f"Reacquisition confidence check: {state.name} (value={raw_confidence:.2f})")
        
        return acceptable
    
    def record_stabilization_frame(self) -> bool:
        """
        Record a stabilization frame (per §18 step 3).
        
        During stabilization, we:
        - Accept hand but don't generate input
        - Continue monitoring confidence
        - Track stabilization progress
        
        Returns True when stabilization is complete.
        """
        old_phase = self.phase
        self.stabilization_frames += 1
        
        # Require minimum stabilization frames
        # More frames for worse confidence
        required_frames = {
            ConfidenceState.HIGH: 3,
            ConfidenceState.MEDIUM: 5,
            ConfidenceState.LOW: 10,
            ConfidenceState.LOST: 15,  # Shouldn't happen, but safety
        }
        
        target = required_frames.get(self.confidence.state, 5)
        
        logger.debug(f"Stabilization frame {self.stabilization_frames}/{target}")
        
        if self.stabilization_frames >= target:
            self.phase = TrackingPhase.STABILIZING
            
            if self._on_phase_change and old_phase != self.phase:
                try:
                    self._on_phase_change(old_phase, self.phase)
                except Exception as e:
                    logger.error(f"Error in phase_change callback: {e}")
            
            logger.info(f"Stabilization complete after {target} frames")
            return True
        
        return False
    
    def record_motion_baseline_established(self) -> None:
        """
        Record establishment of motion baseline (per §18 step 4).
        
        After stabilization, we establish a new reference point
        so the cursor doesn't jump when resuming.
        """
        old_phase = self.phase
        self.phase = TrackingPhase.TRACKING
        self.frames_since_loss = 0
        self.last_reacquisition_time = time.time()
        
        if self._on_tracking_recovered:
            try:
                self._on_tracking_recovered()
            except Exception as e:
                logger.error(f"Error in tracking_recovered callback: {e}")
        
        if self._on_phase_change and old_phase != self.phase:
            try:
                self._on_phase_change(old_phase, self.phase)
            except Exception as e:
                logger.error(f"Error in phase_change callback: {e}")
        
        logger.info(f"Tracking RESUMED: hand reacquired successfully")
    
    def record_normal_tracking(self, raw_confidence: float, evidence: str = "") -> None:
        """Record normal tracking operation."""
        self.frames_since_loss += 1
        
        # Check for confidence drops that might indicate impending loss
        if self.confidence.state == ConfidenceState.HIGH and raw_confidence < 0.7:
            logger.warning(f"Confidence drop detected: {raw_confidence:.2f}")
        elif self.confidence.state == ConfidenceState.MEDIUM and raw_confidence < 0.4:
            logger.warning(f"Confidence drop detected: {raw_confidence:.2f}")
        
        self.confidence.update(raw_confidence, evidence=evidence)
    
    def record_frozen(self, is_frozen: bool) -> None:
        """Record tracking frozen state (e.g., fist gesture)."""
        if is_frozen and self.phase != TrackingPhase.FROZEN:
            old_phase = self.phase
            self.phase = TrackingPhase.FROZEN
            self.is_frozen = True
            
            if self._on_phase_change and old_phase != self.phase:
                try:
                    self._on_phase_change(old_phase, self.phase)
                except Exception as e:
                    logger.error(f"Error in phase_change callback: {e}")
            
            logger.info("Tracking frozen (paused)")
        elif not is_frozen and self.phase == TrackingPhase.FROZEN:
            old_phase = self.phase
            self.phase = TrackingPhase.TRACKING
            self.is_frozen = False
            
            if self._on_phase_change and old_phase != self.phase:
                try:
                    self._on_phase_change(old_phase, self.phase)
                except Exception as e:
                    logger.error(f"Error in phase_change callback: {e}")
            
            logger.info("Tracking resumed (unfrozen)")
    
    def get_movement_permission(self) -> bool:
        """
        Get whether cursor movement is allowed.
        
        Per spec §17: NO movement when LOST.
        """
        if self.phase == TrackingPhase.LOST:
            return False
        if self.phase == TrackingPhase.FROZEN:
            return False
        if self.confidence.state == ConfidenceState.LOST:
            return False
        return True
    
    def get_gesture_permission(self) -> bool:
        """
        Get whether gesture recognition is allowed.
        
        Per spec §17: NO gestures when LOST.
        """
        if self.phase == TrackingPhase.LOST:
            return False
        if self.phase == TrackingPhase.FROZEN:
            return False
        if self.confidence.state == ConfidenceState.LOST:
            return False
        if self.confidence.state == ConfidenceState.LOW:
            return False
        return True
    
    def get_click_permission(self) -> bool:
        """
        Get whether clicks are allowed.
        
        Per spec §53: clicks require higher confidence.
        """
        if not self.get_gesture_permission():
            return False
        if self.confidence.state == ConfidenceState.LOW:
            return False
        return True
    
    def get_input_permission(self) -> bool:
        """Get whether ANY input (movement, gestures, clicks) is allowed."""
        return self.get_movement_permission() and self.get_gesture_permission()
    
    def update(self) -> None:
        """Update status timers and state."""
        current_time = time.time()
        
        if self.phase == TrackingPhase.LOST:
            self.lost_duration = current_time - self.last_lost_time if self.last_lost_time else 0.0
        elif self.phase == TrackingPhase.STABILIZING:
            # In stabilization, check if we should proceed to tracking
            if self.stabilization_frames >= 10:  # Safety timeout
                self.record_motion_baseline_established()
        
        # Cleanup old state
        if self.confidence.state == ConfidenceState.LOST:
            if self.phase != TrackingPhase.LOST and self.phase != TrackingPhase.REACQUIRING:
                self.record_hand_lost(LostReason.CONFIDENCE_DROP, "confidence_below_threshold")
    
    def reset(self) -> None:
        """Reset all tracking state."""
        self.__init__()
        self.confidence.reset()
    
    def get_state_info(self) -> dict:
        """Get complete state information for debugging."""
        return {
            "phase": self.phase.name,
            "lost_reason": self.lost_reason.name if self.lost_reason else None,
            "confidence_state": self.confidence.state.name,
            "confidence_value": self.confidence.confidence_value,
            "lost_duration": self.lost_duration,
            "frames_since_loss": self.frames_since_loss,
            "reacquisition_attempts": self.reacquisition_attempts,
            "stabilization_frames": self.stabilization_frames,
            "is_frozen": self.is_frozen,
        }
