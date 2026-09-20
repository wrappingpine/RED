import sys
sys.path.insert(0, '/home/shubham/airmouse')
from airmouse.tests.test_gesture_hysteresis import TestGestureRecognizer, create_mock_hand
from airmouse.vision.gesture_types import GestureType, TrackingState
from airmouse.vision.gesture_state_machine import GestureState

t = TestGestureRecognizer()
t.setup_method()
print('Setup OK')

for i in range(15):
    hand = create_mock_hand(pinch_dist=0.045)
    events = t.recognizer.process([hand])
    pinch_events = [e for e in events if e.gesture_type == GestureType.LEFT_CLICK]
    sm = t.recognizer._state_machine
    state = sm._gesture_states[GestureType.LEFT_CLICK].state
    consec = sm._gesture_states[GestureType.LEFT_CLICK].consecutive_detected

    # Check filter pipeline candidate
    filter_candidate = t.recognizer._filter_pipeline.confirmation_filter.get_candidate(GestureType.LEFT_CLICK)
    filter_consec = filter_candidate.consecutive_frames if filter_candidate else 0

    print(f'Frame {i}: state={state}, consec_detected={consec}, filter_consec={filter_consec}')
    for e in pinch_events:
        print(f'  Event: {e.gesture_type}, state: {e.state}')