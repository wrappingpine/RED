import sys
sys.path.insert(0, '/home/shubham/airmouse')
from airmouse.tests.test_gesture_hysteresis import TestGestureRecognizer

t = TestGestureRecognizer()
t.setup_method()
print('Setup OK')
hand = t.create_mock_hand(pinch_dist=0.045)
print('Hand created OK')
events = t.recognizer.process([hand])
print('Process OK')
print(f'Events: {len(events)}')
for e in events:
    print(f'  Event: {e.gesture_type}, state: {e.state}')