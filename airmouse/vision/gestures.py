"""
Gesture Recognition Module for Air Mouse

Recognizes gestures from hand landmarks:
- Left click (pinch: thumb + index)
- Right click (pinch: thumb + middle)
- Drag (hold pinch + move)
- Scroll (two fingers extended: index + middle, move up/down)
- Pause/Resume tracking (fist)
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Callable, List
from enum import Enum, auto

from .hand_tracker import Hand, HandLandmark

logger = logging.getLogger(__name__)


class GestureType(Enum):
    """Types of recognized gestures."""
    NONE = auto()
    LEFT_CLICK = auto()
    RIGHT_CLICK = auto()
    DRAG_START = auto()
    DRAG_END = auto()
    SCROLL_UP = auto()
    SCROLL_DOWN = auto()
    SCROLL_HORIZONTAL = auto()
    PAUSE_TRACKING = auto()
    RESUME_TRACKING = auto()
    MIDDLE_CLICK = auto()
    PINCH_CONFIRM = auto()
    PINCH_END = auto()


class TrackingState(Enum):
    """Explicit tracking states for hand tracking."""
    NO_HAND = auto()           # No hand detected
    TRACKING_ONE_HAND = auto() # Single hand being tracked
    TRACKING_TWO_HANDS = auto() # Two hands being tracked (primary + secondary)
    LOST_TRACK = auto()        # Previously tracking, now lost
    FROZEN = auto()            # Tracking frozen (e.g., fist gesture)
    PRECISION_MODE = auto()    # Precision mode active (secondary hand or gesture)


@dataclass
class GestureEvent:
    """A detected gesture event."""
    gesture_type: GestureType
    hand: Hand
    timestamp: float = field(default_factory=time.time)
    data: dict = field(default_factory=dict)


@dataclass
class GestureConfig:
    """Configuration for gesture recognition."""
    # Pinch thresholds with hysteresis (enter < confirm < release)
    # Enter: start detecting pinch
    # Confirm: confirm pinch is active (stricter)
    # Release: release pinch (more lenient to prevent flicker)
    pinch_enter_threshold: float = 0.045    # Enter pinch state (spec: 0.045)
    pinch_confirm_threshold: float = 0.040  # Confirm pinch (stricter) (spec: 0.040)
    pinch_release_threshold: float = 0.070  # Release pinch (more lenient) (spec: 0.070)

    # Legacy single threshold (for backward compatibility)
    pinch_threshold: float = 0.05

    # Scroll detection
    scroll_sensitivity: float = 1.0  # minimum finger movement for scroll (normalized coords)
    scroll_cooldown: float = 0.1  # seconds between scroll events

    # Fist detection for pause
    fist_hold_time: float = 0.5  # seconds to hold fist for pause (spec: 0.5s)

    # Drag
    drag_hold_time: float = 0.2  # seconds to hold pinch before drag starts (spec: 0.2s)
    drag_movement_threshold: float = 0.03  # min movement for drag (spec: 0.03)

    # Click
    click_max_duration: float = 0.3  # max time for click (not drag)
    click_max_movement: float = 0.03  # max movement during click

    # Gesture cooldowns (prevent rapid re-triggering)
    gesture_cooldown: float = 0.3

    # Two-hand tracking
    enable_two_hand: bool = True
    secondary_hand_precision_mode: bool = True
    preferred_handedness: str = "Right"  # "Right" or "Left"


class GestureState:
    """Tracks state for gesture recognition."""

    def __init__(self):
        # Tracking state
        self.tracking_state = TrackingState.NO_HAND
        self.primary_hand_id = None
        self.secondary_hand_id = None

        # Pinch states with hysteresis
        self.left_pinch_active = False
        self.left_pinch_confirmed = False
        self.left_pinch_start_time = 0.0
        self.left_pinch_start_pos = None
        self.left_pinch_was_drag = False

        self.right_pinch_active = False
        self.right_pinch_confirmed = False
        self.right_pinch_start_time = 0.0
        self.right_pinch_start_pos = None

        # Scroll state
        self.last_scroll_time = 0.0
        self.last_scroll_y = 0.0

        # Fist state
        self.fist_start_time = 0.0
        self.fist_active = False
        self.fist_confirmed = False
        self.tracking_paused = False

        # General
        self.last_gesture_time = 0.0
        self.last_gesture_type = GestureType.NONE
        self.last_hand_count = 0

    def reset(self):
        """Reset all state."""
        self.__init__()


class GestureRecognizer:
    """
    Recognizes gestures from hand landmarks and emits events.
    """

    def __init__(self, config: Optional[GestureConfig] = None,
                 callback: Optional[Callable[[GestureEvent], None]] = None):
        self.config = config or GestureConfig()
        self.callback = callback
        self._state = GestureState()
        # For stable hand identity tracking across frames
        self._hand_history: Dict[str, Hand] = {}  # handedness -> Hand

    @property
    def state(self) -> TrackingState:
        """Get current tracking state (for test compatibility)."""
        return self._state.tracking_state

    @property
    def tracking_state(self) -> TrackingState:
        """Get current tracking state."""
        return self._state.tracking_state

    def process(self, hands: List[Hand]) -> List[GestureEvent]:
        """
        Process hands and detect gestures with hysteresis state machine.

        Args:
            hands: List of Hand objects from hand_tracker

        Returns:
            List of detected GestureEvents
        """
        events = []
        current_time = time.time()

        # Build hand map by handedness for stable identity
        hand_map = {}
        for hand in hands:
            if hand and hand.handedness:
                hand_map[hand.handedness] = hand
                self._hand_history[hand.handedness] = hand

        # Determine primary and secondary hand based on config preference
        primary_hand = None
        secondary_hand = None

        if self.config.enable_two_hand and len(hand_map) >= 2:
            # Two hands detected - use preferred handedness for primary
            pref = self.config.preferred_handedness
            if pref in hand_map:
                primary_hand = hand_map[pref]
                # Secondary is the other hand
                for h in hand_map.values():
                    if h.handedness != pref:
                        secondary_hand = h
                        break
            else:
                # Fallback: first two hands
                hands_list = list(hand_map.values())
                primary_hand = hands_list[0] if hands_list else None
                secondary_hand = hands_list[1] if len(hands_list) > 1 else None
        elif len(hand_map) == 1:
            # Single hand - use it as primary
            primary_hand = list(hand_map.values())[0]
        else:
            # No hands
            primary_hand = None
            secondary_hand = None

        hand_count = len(hand_map)

        # Update tracking state based on hand count and config
        if hand_count == 0:
            self._state.tracking_state = TrackingState.NO_HAND
        elif hand_count == 1:
            if self._state.tracking_state in (TrackingState.NO_HAND, TrackingState.LOST_TRACK, TrackingState.FROZEN):
                self._state.tracking_state = TrackingState.TRACKING_ONE_HAND
        elif hand_count >= 2:
            if self.config.enable_two_hand:
                self._state.tracking_state = TrackingState.TRACKING_TWO_HANDS
                if self.config.secondary_hand_precision_mode:
                    self._state.tracking_state = TrackingState.PRECISION_MODE
            else:
                # Two-hand disabled but two hands present - use primary only
                self._state.tracking_state = TrackingState.TRACKING_ONE_HAND
        else:
            self._state.tracking_state = TrackingState.LOST_TRACK

        # Check for fist (pause tracking) - only on primary hand
        if primary_hand and primary_hand.is_fist():
            if not self._state.fist_active:
                self._state.fist_active = True
                self._state.fist_start_time = current_time
            elif (current_time - self._state.fist_start_time >= self.config.fist_hold_time
                  and not self._state.tracking_paused
                  and not self._state.fist_confirmed):
                # Confirm fist
                self._state.fist_confirmed = True
                self._state.tracking_paused = True
                self._state.tracking_state = TrackingState.FROZEN
                events.append(GestureEvent(
                    gesture_type=GestureType.PAUSE_TRACKING,
                    hand=primary_hand,
                    timestamp=current_time
                ))
        else:
            # Hand not in fist
            if self._state.fist_active and self._state.tracking_paused:
                # Resume tracking
                self._state.tracking_paused = False
                self._state.fist_confirmed = False
                self._state.tracking_state = TrackingState.TRACKING_ONE_HAND
                events.append(GestureEvent(
                    gesture_type=GestureType.RESUME_TRACKING,
                    hand=primary_hand,
                    timestamp=current_time
                ))
            self._state.fist_active = False
            self._state.fist_start_time = 0.0
            self._state.fist_confirmed = False

        # If tracking paused, don't process other gestures
        if self._state.tracking_paused:
            self._state.last_hand_count = hand_count
            return events

        # No hand - release any active pinches
        if not primary_hand or not primary_hand.landmarks:
            if self._state.left_pinch_active:
                events.extend(self._end_left_pinch(current_time, drag=False))
            if self._state.right_pinch_active:
                events.append(self._end_right_pinch(current_time))
            self._state.last_hand_count = hand_count
            return events

        # Check left pinch (thumb + index) - Left Click / Drag with hysteresis
        left_pinch_dist = primary_hand.pinch_distance("thumb", "index")

        if not self._state.left_pinch_active:
            # Check enter threshold
            if left_pinch_dist < self.config.pinch_enter_threshold:
                self._state.left_pinch_active = True
                self._state.left_pinch_confirmed = False
                self._state.left_pinch_start_time = current_time
                self._state.left_pinch_start_pos = primary_hand.index_tip
                self._state.left_pinch_was_drag = False
                # Emit LEFT_CLICK (pinch start) event
                events.append(GestureEvent(
                    gesture_type=GestureType.LEFT_CLICK,
                    hand=primary_hand,
                    timestamp=current_time
                ))
        else:
            # Pinch active - check confirm/release
            if not self._state.left_pinch_confirmed:
                # Waiting for confirmation
                if left_pinch_dist < self.config.pinch_confirm_threshold:
                    self._state.left_pinch_confirmed = True
                    # Emit PINCH_CONFIRM event
                    events.append(GestureEvent(
                        gesture_type=GestureType.PINCH_CONFIRM,
                        hand=primary_hand,
                        timestamp=current_time
                    ))
            else:
                # Confirmed - check release
                if left_pinch_dist > self.config.pinch_release_threshold:
                    events.extend(self._end_left_pinch(current_time))

        # If left pinch confirmed and held - check for drag
        if self._state.left_pinch_confirmed and self._state.left_pinch_active:
            hold_duration = current_time - self._state.left_pinch_start_time
            if (hold_duration >= self.config.drag_hold_time
                    and not self._state.left_pinch_was_drag):
                if self._state.left_pinch_start_pos and primary_hand.index_tip:
                    dx = primary_hand.index_tip.x - self._state.left_pinch_start_pos.x
                    dy = primary_hand.index_tip.y - self._state.left_pinch_start_pos.y
                    movement = (dx * dx + dy * dy) ** 0.5
                    if movement > self.config.click_max_movement:
                        self._state.left_pinch_was_drag = True
                        events.append(GestureEvent(
                            gesture_type=GestureType.DRAG_START,
                            hand=primary_hand,
                            timestamp=current_time,
                            data={"start_pos": (self._state.left_pinch_start_pos.x,
                                                  self._state.left_pinch_start_pos.y)}
                        ))

        # Check right pinch (thumb + middle) - Right Click with hysteresis
        right_pinch_dist = primary_hand.pinch_distance("thumb", "middle")

        if not self._state.right_pinch_active:
            if right_pinch_dist < self.config.pinch_enter_threshold:
                self._state.right_pinch_active = True
                self._state.right_pinch_confirmed = False
                self._state.right_pinch_start_time = current_time
                self._state.right_pinch_start_pos = primary_hand.middle_tip
        else:
            if not self._state.right_pinch_confirmed:
                if right_pinch_dist < self.config.pinch_confirm_threshold:
                    self._state.right_pinch_confirmed = True
            else:
                if right_pinch_dist > self.config.pinch_release_threshold:
                    events.append(self._end_right_pinch(current_time))

        # Check scroll gesture (index + middle extended, others folded) - primary hand
        if primary_hand.is_scroll_gesture():
            self._process_scroll(primary_hand, current_time, events)

        # Check middle click (thumb + ring pinch)
        middle_pinch_dist = primary_hand.pinch_distance("thumb", "ring")
        if middle_pinch_dist < self.config.pinch_enter_threshold:
            if not hasattr(self.state, '_middle_pinch_active') or not self._state._middle_pinch_active:
                self._state._middle_pinch_active = True
                events.append(GestureEvent(
                    gesture_type=GestureType.MIDDLE_CLICK,
                    hand=primary_hand,
                    timestamp=current_time
                ))
        else:
            self._state._middle_pinch_active = False

        # Emit events via callback
        for event in events:
            if self.callback:
                self.callback(event)

        self._state.last_hand_count = hand_count
        return events

    def _end_left_pinch(self, current_time: float, drag: bool = True) -> List[GestureEvent]:
        """Handle left pinch release."""
        was_drag = self._state.left_pinch_was_drag
        hold_duration = current_time - self._state.left_pinch_start_time

        self._state.left_pinch_active = False
        self._state.left_pinch_was_drag = False
        self._state.left_pinch_start_pos = None
        self._state.left_pinch_confirmed = False

        events = []

        # Always emit PINCH_END event
        events.append(GestureEvent(
            gesture_type=GestureType.PINCH_END,
            hand=None,
            timestamp=current_time,
            data={"duration": hold_duration, "was_drag": was_drag}
        ))

        if drag and was_drag:
            events.append(GestureEvent(
                gesture_type=GestureType.DRAG_END,
                hand=None,
                timestamp=current_time,
                data={"duration": hold_duration}
            ))
        elif not drag or (hold_duration <= self.config.click_max_duration
                          and not was_drag):
            # Quick click
            events.append(GestureEvent(
                gesture_type=GestureType.LEFT_CLICK,
                hand=None,
                timestamp=current_time,
                data={"duration": hold_duration}
            ))
        return events

    def _end_right_pinch(self, current_time: float) -> GestureEvent:
        """Handle right pinch release."""
        hold_duration = current_time - self._state.right_pinch_start_time
        self._state.right_pinch_active = False
        self._state.right_pinch_start_pos = None

        if hold_duration <= self.config.click_max_duration:
            return GestureEvent(
                gesture_type=GestureType.RIGHT_CLICK,
                hand=None,
                timestamp=current_time,
                data={"duration": hold_duration}
            )
        return GestureEvent(
            gesture_type=GestureType.NONE,
            hand=None,
            timestamp=current_time
        )

    def _process_scroll(self, hand: Hand, current_time: float, events: List[GestureEvent]):
        """Process scroll gesture (two fingers extended)."""
        # Check cooldown
        if current_time - self._state.last_scroll_time < self.config.scroll_cooldown:
            return

        if hand.index_tip and hand.middle_tip:
            # Average Y position of index and middle tips
            avg_y = (hand.index_tip.y + hand.middle_tip.y) / 2

            if self._state.last_scroll_y > 0:
                dy = avg_y - self._state.last_scroll_y
                if abs(dy) > self.config.scroll_sensitivity:
                    if dy < 0:  # Moving up (smaller y)
                        events.append(GestureEvent(
                            gesture_type=GestureType.SCROLL_UP,
                            hand=hand,
                            timestamp=current_time,
                            data={"amount": int(-dy * 100)}
                        ))
                    else:  # Moving down
                        events.append(GestureEvent(
                            gesture_type=GestureType.SCROLL_DOWN,
                            hand=hand,
                            timestamp=current_time,
                            data={"amount": int(dy * 100)}
                        ))
                    self._state.last_scroll_time = current_time

            self._state.last_scroll_y = avg_y
        else:
            self._state.last_scroll_y = 0.0

    @property
    def tracking_state(self) -> TrackingState:
        """Get current tracking state."""
        return self._state.tracking_state

    def is_tracking_paused(self) -> bool:
        return self._state.tracking_paused

    def set_tracking_paused(self, paused: bool):
        self._state.tracking_paused = paused

    def reset(self):
        self._state.reset()


# Convenience function for simple gesture detection
def detect_gesture(hand: Hand, config: Optional[GestureConfig] = None) -> GestureType:
    """
    Simple one-shot gesture detection (stateless).

    Args:
        hand: Hand object
        config: Optional gesture config

    Returns:
        Single GestureType (most prominent)
    """
    if not hand or not hand.landmarks:
        return GestureType.NONE

    cfg = config or GestureConfig()

    # Priority: fist (pause) > pinch > scroll
    if hand.is_fist():
        return GestureType.PAUSE_TRACKING

    # Left pinch (uses enter threshold for detection)
    if hand.is_pinch("thumb", "index", cfg.pinch_enter_threshold):
        return GestureType.LEFT_CLICK

    # Right pinch
    if hand.is_pinch("thumb", "middle", cfg.pinch_enter_threshold):
        return GestureType.RIGHT_CLICK

    # Scroll
    if hand.is_scroll_gesture():
        return GestureType.SCROLL_UP  # Direction determined by movement

    # Middle click
    if hand.is_pinch("thumb", "ring", cfg.pinch_enter_threshold):
        return GestureType.MIDDLE_CLICK

    return GestureType.NONE


if __name__ == "__main__":
    # Test with mock hand
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from hand_tracker import Hand, Landmark, HandLandmark

    logging.basicConfig(level=logging.INFO)

    print("Testing GestureRecognizer...")

    # Create mock hand
    def create_hand(index_extended=True, middle_extended=False,
                    thumb_pinched=False, fist=False):
        landmarks = []
        for i in range(21):
            if i == HandLandmark.INDEX_TIP.value:
                landmarks.append(Landmark(x=0.5, y=0.3 if index_extended else 0.6, z=0.0))
            elif i == HandLandmark.MIDDLE_TIP.value:
                landmarks.append(Landmark(x=0.55, y=0.3 if middle_extended else 0.6, z=0.0))
            elif i == HandLandmark.THUMB_TIP.value:
                landmarks.append(Landmark(x=0.48 if thumb_pinched else 0.4, y=0.5, z=0.0))
            elif i == HandLandmark.RING_TIP.value:
                landmarks.append(Landmark(x=0.6, y=0.6, z=0.0))
            elif i == HandLandmark.PINKY_TIP.value:
                landmarks.append(Landmark(x=0.65, y=0.6, z=0.0))
            elif i == HandLandmark.WRIST.value:
                landmarks.append(Landmark(x=0.5, y=0.7, z=0.0))
            else:
                landmarks.append(Landmark(x=0.5, y=0.6, z=0.0))
        return Hand(landmarks=landmarks, handedness="Right", confidence=1.0)

    recognizer = GestureRecognizer()

    # Test 1: Left click (pinch thumb+index)
    print("\n1. Left pinch (click)...")
    hand = create_hand(index_extended=False, thumb_pinched=True)
    events = recognizer.process(hand)
    print(f"   Events: {[e.gesture_type.name for e in events]}")

    # Release pinch
    hand = create_hand(index_extended=True, thumb_pinched=False)
    events = recognizer.process(hand)
    print(f"   After release: {[e.gesture_type.name for e in events]}")

    # Test 2: Right click
    print("\n2. Right pinch (click)...")
    hand = create_hand(middle_extended=False)
    # Manually set thumb+middle close
    hand.landmarks[HandLandmark.THUMB_TIP.value] = Landmark(x=0.5, y=0.5, z=0.0)
    hand.landmarks[HandLandmark.MIDDLE_TIP.value] = Landmark(x=0.52, y=0.5, z=0.0)
    hand._compute_derived()
    events = recognizer.process(hand)
    print(f"   Events: {[e.gesture_type.name for e in events]}")

    hand = create_hand()
    events = recognizer.process(hand)
    print(f"   After release: {[e.gesture_type.name for e in events]}")

    # Test 3: Fist (pause)
    print("\n3. Fist (pause)...")
    hand = create_hand(fist=True)
    # Force fist detection by making all fingers folded
    for fname in ["index", "middle", "ring", "pinky"]:
        hand._finger_states[fname] = False
    hand._finger_states["thumb"] = False
    events = recognizer.process(hand)
    print(f"   Events: {[e.gesture_type.name for e in events]}")

    # Hold fist
    for _ in range(10):
        events = recognizer.process(hand)
    print(f"   After hold: {[e.gesture_type.name for e in events]}")

    # Test 4: Scroll gesture
    print("\n4. Scroll gesture (index+middle extended)...")
    recognizer.reset()
    hand = create_hand(index_extended=True, middle_extended=True)
    hand.landmarks[HandLandmark.INDEX_TIP.value] = Landmark(x=0.5, y=0.4, z=0.0)
    hand.landmarks[HandLandmark.MIDDLE_TIP.value] = Landmark(x=0.55, y=0.4, z=0.0)
    hand._compute_derived()
    events = recognizer.process(hand)
    print(f"   Initial: {[e.gesture_type.name for e in events]}")

    # Move up
    hand.landmarks[HandLandmark.INDEX_TIP.value] = Landmark(x=0.5, y=0.35, z=0.0)
    hand.landmarks[HandLandmark.MIDDLE_TIP.value] = Landmark(x=0.55, y=0.35, z=0.0)
    hand._compute_derived()
    events = recognizer.process(hand)
    print(f"   Move up: {[e.gesture_type.name for e in events]}")

    print("\nTest complete")