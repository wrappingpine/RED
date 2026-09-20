import sys
sys.path.insert(0, '/home/shubham/airmouse')
from airmouse.tests.test_gesture_hysteresis import TestGestureRecognizer, create_mock_hand
from airmouse.vision.gesture_types import GestureType, TrackingState

t = TestGestureRecognizer()
t.setup_method()
print('Setup OK')

for i in range(15):
    hand = create_mock_hand(pinch_dist=0.045)
    # Call filter pipeline directly
    filter_result = t.recognizer._filter_pipeline.process(GestureType.LEFT_CLICK, hand, 0.8)
    print(f'Frame {i}: filter_result={filter_result}')
    # Also check confirmation filter
    candidate = t.recognizer._filter_pipeline.confirmation_filter.get_candidate(GestureType.LEFT_CLICK)
    if candidate:
        print(f'  Candidate: consecutive_frames={candidate.consecutive_frames}, confirmed={candidate.is_confirmed}')
    else:
        print(f'  Candidate: None')