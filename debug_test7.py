import sys
sys.path.insert(0, '/home/shubham/airmouse')
from airmouse.tests.test_gesture_hysteresis import TestGestureRecognizer, create_mock_hand
from airmouse.vision.gesture_types import GestureType, TrackingState

t = TestGestureRecognizer()
t.setup_method()
print('Setup OK')

for i in range(15):
    hand = create_mock_hand(pinch_dist=0.045)
    # Call filter pipeline directly with confidence 1.0
    stable_states = t.recognizer._filter_pipeline.finger_stability.update(hand.handedness, hand.finger_states)
    filter_result = t.recognizer._filter_pipeline.process_with_stable_states(
        GestureType.LEFT_CLICK, hand, 1.0, stable_states
    )
    print(f'Frame {i}: stable_states={bool(stable_states)}, filter_result={filter_result}')
    # Also check confirmation filter
    candidate = t.recognizer._filter_pipeline.confirmation_filter.get_candidate(GestureType.LEFT_CLICK)
    if candidate:
        print(f'  Candidate: consecutive_frames={candidate.consecutive_frames}, total_frames={candidate.total_frames}')
    else:
        print(f'  Candidate: None')