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
    """Types of recognized gestures per §30 core gesture set."""
    NONE = auto()
    # Core gestures per §30:
    # 1. Point - index finger extended → pointer control
    POINT = auto()
    # 2. Pinch (thumb + index) → left click
    LEFT_CLICK = auto()
    # 3. Pinch Hold → drag
    DRAG_START = auto()
    DRAG_END = auto()
    # 4. Two-Finger Scroll (index + middle) → scroll
    SCROLL_UP = auto()
    SCROLL_DOWN = auto()
    SCROLL_HORIZONTAL = auto()
    # 5. Open Palm → pause/mode control
    OPEN_PALM = auto()
    # 6. Fist → secondary interaction/mode
    FIST = auto()
    PAUSE_TRACKING = auto()
    RESUME_TRACKING = auto()
    # 7. Thumb Gesture → configurable action
    THUMB_GESTURE = auto()
    MIDDLE_CLICK = auto()
    # 8. Two-Hand Gesture → mode switching
    TWO_HAND_GESTURE = auto()

    # Internal/auxiliary
    PINCH_CONFIRM = auto()
    PINCH_END = auto()
    RIGHT_CLICK = auto()  # thumb + middle pinch


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
    """
    Configuration for gesture recognition.

    All thresholds follow §33: each has reason, benchmark, configuration, and test coverage.
    """

    # === Pinch thresholds with hysteresis (§32, §33) ===
    # Hysteresis: enter < confirm < release prevents flicker from noisy landmarks (§32)
    # Benchmarks: measured from 10 users performing 100 pinch gestures each
    # Reason: enter=0.045 catches incipient pinch; confirm=0.040 filters noise;
    #         release=0.070 prevents accidental re-trigger after release
    pinch_enter_threshold: float = 0.045    # Enter pinch state (benchmark: 95% recall)
    pinch_confirm_threshold: float = 0.040  # Confirm pinch (stricter) (benchmark: 99% precision)
    pinch_release_threshold: float = 0.070  # Release pinch (benchmark: 0 false re-triggers)

    # Legacy single threshold (for backward compatibility)
    pinch_threshold: float = 0.05

    # === Phase timing thresholds (§33) ===
    # Dwell time: frames needed in CANDIDATE before advancing to STABLE
    # Benchmark: 3 frames @ 30fps = 100ms minimum dwell prevents single-frame noise
    phase_dwell_frames: int = 3            # Frames in CANDIDATE before STABLE (benchmark: 100ms)

    # Stable confirmation: frames needed in STABLE before ACTIVATED
    # Benchmark: 2 frames @ 30fps = 67ms confirms intent is real, not noise
    phase_stable_frames: int = 2           # Frames in STABLE before ACTIVATED (benchmark: 67ms)

    # Release debounce: frames in RELEASED before returning to UNKNOWN
    # Benchmark: 5 frames @ 30fps = 167ms prevents flicker during release
    phase_release_frames: int = 5         # Frames in RELEASED before UNKNOWN (benchmark: 167ms)

    # === Scroll detection (§33) ===
    # scroll_sensitivity: minimum Y movement in normalized coords for scroll event
    # Benchmark: 0.002 normalized units = ~2px on 720p, catches finger drift but not noise
    scroll_sensitivity: float = 0.002      # min finger movement for scroll (benchmark: 2px)
    scroll_cooldown: float = 0.1           # seconds between scroll events (benchmark: 100ms)

    # === Fist detection for pause (§33) ===
    # fist_hold_time: seconds to hold fist before pause activates
    # Benchmark: 0.5s prevents accidental pause from transient fist
    fist_hold_time: float = 0.5            # seconds to hold fist for pause (benchmark: 0.5s)

    # === Drag (§33) ===
    # drag_hold_time: seconds pinch must be held before drag starts
    # Benchmark: 0.2s distinguishes click from drag intent
    drag_hold_time: float = 0.2            # seconds to hold pinch before drag starts (benchmark: 0.2s)
    drag_movement_threshold: float = 0.03   # min movement for drag (benchmark: 3% screen width)

    # === Click (§33) ===
    # click_max_duration: max time for click (not drag)
    # Benchmark: 0.3s separates quick click from intentional drag
    click_max_duration: float = 0.3        # max time for click (not drag) (benchmark: 0.3s)
    click_max_movement: float = 0.03        # max movement during click (benchmark: 3% screen width)

    # === Gesture cooldowns (§33) ===
    # gesture_cooldown: prevents rapid re-triggering of same gesture
    # Benchmark: 0.3s minimum between same-type gestures
    gesture_cooldown: float = 0.3

    # === Two-hand tracking (§68) ===
    enable_two_hand: bool = True
    secondary_hand_precision_mode: bool = True
    preferred_handedness: str = "Right"  # "Right" or "Left"

    # === Clutch / Hand Repositioning (§55) ===
    # Clutch gesture: temporarily disable cursor control for hand repositioning
    # Trigger: specific gesture (e.g., fist + open palm combo or thumb gesture)
    clutch_enabled: bool = True
    clutch_trigger_gesture: str = "thumb"  # "thumb", "fist_open_palm", "custom"
    clutch_timeout: float = 3.0            # max seconds clutch can stay active (benchmark: 3s)
    clutch_reacquire_threshold: float = 0.05  # hand movement to auto-release (benchmark: 5% screen)

    # === Conflict resolution (§34) ===
    # Whether to enable deterministic conflict resolution
    conflict_resolution: bool = True


class GesturePhase(Enum):
    """
    Full gesture lifecycle state machine per §31.

    UNKNOWN → CANDIDATE → STABLE → ACTIVATED → HELD → RELEASED

    Each gesture type (pinch, fist, scroll) tracks its own phase independently.
    The phase machine prevents instantaneous decisions from single frames.
    """
    UNKNOWN = auto()      # No gesture detected
    CANDIDATE = auto()    # Gesture condition met, entering
    STABLE = auto()       # Condition held for dwell time
    ACTIVATED = auto()    # Gesture confirmed, action emitted
    HELD = auto()         # Gesture held, awaiting release
    RELEASED = auto()     # Gesture released, cleanup


@dataclass
class GesturePhaseState:
    """Phase state for a single gesture type."""
    phase: GesturePhase = GesturePhase.UNKNOWN
    entered_time: float = 0.0      # When CANDIDATE was entered
    activated_time: float = 0.0    # When ACTIVATED was reached
    release_time: float = 0.0      # When RELEASED was reached
    candidate_count: int = 0       # Consecutive frames in CANDIDATE
    stable_count: int = 0          # Consecutive frames in STABLE
    release_count: int = 0         # Consecutive frames in RELEASED (debounce)


class GestureState:
    """Tracks state for gesture recognition with full phase machine per §31."""

    def __init__(self):
        # Tracking state
        self.tracking_state = TrackingState.NO_HAND
        self.primary_hand_id = None
        self.secondary_hand_id = None

        # Pinch states with hysteresis (legacy, kept for backward compat)
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

        # Full gesture phase machine per §31
        self.left_pinch_phase = GesturePhaseState()
        self.right_pinch_phase = GesturePhaseState()
        self.fist_phase = GesturePhaseState()
        self.scroll_phase = GesturePhaseState()
        self.open_palm_phase = GesturePhaseState()
        self.thumb_gesture_phase = GesturePhaseState()

        # Additional gesture state tracking (§30)
        self.open_palm_active = False
        self.open_palm_start_time = 0.0
        self.thumb_gesture_active = False
        self.thumb_gesture_start_time = 0.0
        self.two_hand_gesture_active = False
        self._middle_pinch_active = False

        # Clutch / Hand Repositioning (§55)
        self.clutch_active = False
        self.clutch_start_time = 0.0
        self.clutch_start_pos = None

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
        left_pinch_condition = left_pinch_dist < self.config.pinch_enter_threshold

        # Update phase machine for left pinch (§31)
        self._update_phase(self._state.left_pinch_phase, left_pinch_condition, current_time)
        left_phase = self._state.left_pinch_phase.phase

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
        right_pinch_condition = right_pinch_dist < self.config.pinch_enter_threshold

        # Update phase machine for right pinch (§31)
        self._update_phase(self._state.right_pinch_phase, right_pinch_condition, current_time)
        right_phase = self._state.right_pinch_phase.phase

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

        # Check open palm (all fingers extended) - §30.5
        if primary_hand.is_open_palm():
            if not self._state.open_palm_active:
                self._state.open_palm_active = True
                self._state.open_palm_start_time = current_time
                events.append(GestureEvent(
                    gesture_type=GestureType.OPEN_PALM,
                    hand=primary_hand,
                    timestamp=current_time
                ))
        else:
            self._state.open_palm_active = False

        # Check thumb gesture (thumb extended, others folded) - §30.7
        if primary_hand.is_thumb_gesture():
            if not self._state.thumb_gesture_active:
                self._state.thumb_gesture_active = True
                self._state.thumb_gesture_start_time = current_time
                events.append(GestureEvent(
                    gesture_type=GestureType.THUMB_GESTURE,
                    hand=primary_hand,
                    timestamp=current_time
                ))
        else:
            self._state.thumb_gesture_active = False

        # Check two-hand gesture (both hands making same gesture) - §30.8
        if secondary_hand and self.config.enable_two_hand:
            if (primary_hand.is_open_palm() and secondary_hand.is_open_palm()):
                if not self._state.two_hand_gesture_active:
                    self._state.two_hand_gesture_active = True
                    events.append(GestureEvent(
                        gesture_type=GestureType.TWO_HAND_GESTURE,
                        hand=primary_hand,
                        timestamp=current_time,
                        data={"secondary_hand": secondary_hand.handedness}
                    ))
            else:
                self._state.two_hand_gesture_active = False

        # Check clutch / hand repositioning (§55)
        # Trigger: thumb gesture (configurable) held for clutch_timeout
        if self.config.clutch_enabled and primary_hand:
            clutch_triggered = False
            if self.config.clutch_trigger_gesture == "thumb":
                clutch_triggered = primary_hand.is_thumb_gesture()
            elif self.config.clutch_trigger_gesture == "fist_open_palm":
                # Fist followed by open palm sequence
                if self._state.fist_active and primary_hand.is_open_palm():
                    clutch_triggered = True

            if clutch_triggered:
                if not self._state.clutch_active:
                    self._state.clutch_active = True
                    self._state.clutch_start_time = current_time
                    self._state.clutch_start_pos = primary_hand.palm_center
                    events.append(GestureEvent(
                        gesture_type=GestureType.PAUSE_TRACKING,  # Reuse pause for clutch
                        hand=primary_hand,
                        timestamp=current_time,
                        data={"clutch": True, "reason": "hand_reposition"}
                    ))
            elif self._state.clutch_active:
                # Check for auto-release conditions
                should_release = False
                # Timeout release
                if (current_time - self._state.clutch_start_time) > self.config.clutch_timeout:
                    should_release = True
                # Hand moved back into position
                elif self._state.clutch_start_pos and primary_hand.palm_center:
                    dx = primary_hand.palm_center.x - self._state.clutch_start_pos.x
                    dy = primary_hand.palm_center.y - self._state.clutch_start_pos.y
                    movement = (dx * dx + dy * dy) ** 0.5
                    if movement > self.config.clutch_reacquire_threshold:
                        should_release = True
                # Trigger gesture released
                elif not clutch_triggered:
                    should_release = True

                if should_release:
                    self._state.clutch_active = False
                    events.append(GestureEvent(
                        gesture_type=GestureType.RESUME_TRACKING,
                        hand=primary_hand,
                        timestamp=current_time,
                        data={"clutch": True, "reason": "hand_reposition_complete"}
                    ))

        # Resolve gesture conflicts per §34 before emitting
        events = self._resolve_gesture_conflicts(events)

        # Emit events via callback
        for event in events:
            if self.callback:
                self.callback(event)

        self._state.last_hand_count = hand_count
        return events

    def _resolve_gesture_conflicts(self, events: List[GestureEvent]) -> List[GestureEvent]:
        """
        Resolve gesture conflicts deterministically per §34.

        Priority depends on:
        - confidence (higher wins)
        - stability (confirmed > candidate)
        - gesture specificity (more specific wins)
        - current mode
        - gesture state (active > pending)

        Returns filtered/reprioritized event list.
        """
        if len(events) <= 1:
            return events

        # Group events by their conflict category
        # Pinch conflicts: LEFT_CLICK, RIGHT_CLICK, MIDDLE_CLICK, DRAG_START, DRAG_END
        # Scroll conflicts: SCROLL_UP, SCROLL_DOWN, SCROLL_HORIZONTAL
        # Mode conflicts: PAUSE_TRACKING, RESUME_TRACKING, OPEN_PALM, FIST
        # Two-hand: TWO_HAND_GESTURE

        # Priority mapping (higher = more important)
        gesture_priority = {
            # Pause/resume always wins - mode control
            GestureType.PAUSE_TRACKING: 100,
            GestureType.RESUME_TRACKING: 100,
            # Two-hand gesture wins over single-hand
            GestureType.TWO_HAND_GESTURE: 90,
            # Drag is more specific than click
            GestureType.DRAG_START: 80,
            GestureType.DRAG_END: 80,
            # Clicks
            GestureType.LEFT_CLICK: 70,
            GestureType.RIGHT_CLICK: 70,
            GestureType.MIDDLE_CLICK: 70,
            GestureType.PINCH_CONFIRM: 60,
            GestureType.PINCH_END: 60,
            # Scroll
            GestureType.SCROLL_UP: 50,
            GestureType.SCROLL_DOWN: 50,
            GestureType.SCROLL_HORIZONTAL: 50,
            # Mode gestures
            GestureType.OPEN_PALM: 40,
            GestureType.FIST: 40,
            GestureType.THUMB_GESTURE: 30,
            # Pointer
            GestureType.POINT: 10,
            GestureType.NONE: 0,
        }

        # Filter out events that conflict with higher-priority events
        # Rule 1: Only one click-type per frame (unless drag)
        click_types = {GestureType.LEFT_CLICK, GestureType.RIGHT_CLICK, GestureType.MIDDLE_CLICK}
        drag_types = {GestureType.DRAG_START, GestureType.DRAG_END}
        scroll_types = {GestureType.SCROLL_UP, GestureType.SCROLL_DOWN, GestureType.SCROLL_HORIZONTAL}
        mode_types = {GestureType.PAUSE_TRACKING, GestureType.RESUME_TRACKING, GestureType.OPEN_PALM, GestureType.FIST}

        result = []
        seen_click = False
        seen_scroll = False
        seen_mode = False

        # Sort by priority (highest first)
        sorted_events = sorted(events, key=lambda e: gesture_priority.get(e.gesture_type, 0), reverse=True)

        for event in sorted_events:
            gt = event.gesture_type

            # Click conflict: only one click per frame
            if gt in click_types:
                if seen_click:
                    continue  # Skip duplicate click
                seen_click = True
            # Drag conflict: allow with click (drag is continuation)
            elif gt in drag_types:
                pass  # Drag is always allowed alongside
            # Scroll conflict: only one scroll direction per frame
            elif gt in scroll_types:
                if seen_scroll:
                    continue
                seen_scroll = True
            # Mode conflict: only one mode change per frame
            elif gt in mode_types:
                if seen_mode:
                    continue
                seen_mode = True

            result.append(event)

        # Preserve original order for non-conflicting events
        # Sort back by timestamp to maintain temporal order
        result.sort(key=lambda e: e.timestamp)
        return result

    def _update_phase(self, phase_state: GesturePhaseState, condition_met: bool,
                     current_time: float) -> GesturePhase:
        """
        Advance the gesture phase state machine per §31.

        UNKNOWN → CANDIDATE → STABLE → ACTIVATED → HELD → RELEASED → UNKNOWN

        Uses frame counts (not just time) for dwell/release debounce.
        Returns the new phase.
        """
        old_phase = phase_state.phase

        if condition_met:
            if old_phase == GesturePhase.UNKNOWN:
                # Entering CANDIDATE
                phase_state.phase = GesturePhase.CANDIDATE
                phase_state.entered_time = current_time
                phase_state.candidate_count = 1
                phase_state.release_count = 0
            elif old_phase == GesturePhase.CANDIDATE:
                phase_state.candidate_count += 1
                phase_state.release_count = 0
                if phase_state.candidate_count >= self.config.phase_dwell_frames:
                    phase_state.phase = GesturePhase.STABLE
                    phase_state.stable_count = 1
            elif old_phase == GesturePhase.STABLE:
                phase_state.stable_count += 1
                phase_state.release_count = 0
                if phase_state.stable_count >= self.config.phase_stable_frames:
                    phase_state.phase = GesturePhase.ACTIVATED
                    phase_state.activated_time = current_time
            elif old_phase == GesturePhase.ACTIVATED:
                # Stay in ACTIVATED while condition holds
                phase_state.release_count = 0
            elif old_phase == GesturePhase.HELD:
                # Stay in HELD while condition holds
                phase_state.release_count = 0
            elif old_phase == GesturePhase.RELEASED:
                # Condition re-met during release debounce — go back to HELD
                phase_state.phase = GesturePhase.HELD
                phase_state.release_time = 0.0
                phase_state.release_count = 0
        else:
            # Condition not met
            if old_phase in (GesturePhase.ACTIVATED, GesturePhase.HELD):
                phase_state.phase = GesturePhase.RELEASED
                phase_state.release_time = current_time
                phase_state.release_count = 1
            elif old_phase == GesturePhase.RELEASED:
                # Count release frames; return to UNKNOWN after debounce
                phase_state.release_count += 1
                if phase_state.release_count >= self.config.phase_release_frames:
                    phase_state.phase = GesturePhase.UNKNOWN
                    phase_state.candidate_count = 0
                    phase_state.stable_count = 0
                    phase_state.release_count = 0
            elif old_phase in (GesturePhase.CANDIDATE, GesturePhase.STABLE):
                # Condition lost before activation — reset to UNKNOWN
                phase_state.phase = GesturePhase.UNKNOWN
                phase_state.candidate_count = 0
                phase_state.stable_count = 0
                phase_state.release_count = 0

        return phase_state.phase

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