import sys
sys.path.insert(0, '/home/shubham/airmouse')
from airmouse.tests.test_gesture_hysteresis import TestGestureRecognizer, create_mock_hand
from airmouse.vision.gesture_types import GestureType

t = TestGestureRecognizer()
t.setup_method()
print('Setup OK')

for i in range(12):
    hand = create_mock_hand(pinch_dist=0.045)
    events = t.recognizer.process([hand])
    pinch_events = [e for e in events if e.gesture_type == GestureType.LEFT_CLICK]
    print(f'Frame {i}: pinch_events={len(pinch_events)}, events={len(events)}, tracking_state={t.recognizer.gesture_state.tracking_state}')
    for e in pinch_events:
        print(f'  Event: {e.gesture_type}, state: {e.state}')