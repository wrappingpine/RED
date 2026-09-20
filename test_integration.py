import logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
from airmouse.control.main_loop import AirMouseController, AirMouseConfig
config = AirMouseConfig()
config.camera.device_index = 0
config.tracking.use_head_relative = True
controller = AirMouseController(config)

# Enable debug logging for all vision modules
for name in ['airmouse.vision.gestures', 'airmouse.vision.gesture_state_machine', 'airmouse.vision.gesture_filter',
             'airmouse.vision.hand_tracker', 'airmouse.vision.face_tracker', 'airmouse.control.main_loop', 'airmouse.vision.tracking_processor']:
    logging.getLogger(name).setLevel(logging.DEBUG)

def on_gesture(event):
    print(f'>>> GESTURE CALLBACK: {event.gesture_type.value} - {event.data}')
controller.on_gesture = on_gesture

# Also add callback for hand detection
original_on_hand = controller.on_hand_detected
def debug_on_hand(hand):
    print(f'>>> HAND DETECTED: handedness={hand.handedness}, confidence={hand.confidence:.2f}')
    if original_on_hand:
        original_on_hand(hand)
controller.on_hand_detected = debug_on_hand

# Also patch face tracker (will be created in initialize)
def wait_for_face_tracker():
    import time
    for _ in range(50):  # Wait up to 5 seconds
        if controller.face_tracker:
            return True
        time.sleep(0.1)
    return False

# We need to patch after initialize - let's patch the initialize method
original_initialize = controller.initialize
def debug_initialize():
    result = original_initialize()
    if controller.face_tracker:
        original_face_process = controller.face_tracker.process
        def debug_face_process(frame):
            print(f'>>> FACE TRACKER CALLED')
            faces = original_face_process(frame)
            if faces:
                print(f'>>> FACE DETECTED: {len(faces)} face(s), conf={faces[0].confidence:.2f}')
            else:
                print(f'>>> FACE TRACKER RETURNED: []')
            return faces
        controller.face_tracker.process = debug_face_process
    return result
controller.initialize = debug_initialize

result = controller.start()
import time
time.sleep(2)  # Wait for initialization
# Also monkey-patch to see what gesture_recognizer returns
if controller.gesture_recognizer:
    original_process = controller.gesture_recognizer.process
    def debug_process(hands):
        events = original_process(hands)
        if events:
            print(f'>>> GESTURE_RECOGNIZER EVENTS: {[e.gesture_type.value for e in events]}')
        return events
    controller.gesture_recognizer.process = debug_process

# Also patch tracking processor process
if controller.tracking_processor:
    original_tp_process = controller.tracking_processor.process
    def debug_tp_process(hands, faces):
        result = original_tp_process(hands, faces)
        if result and result.projection:
            print(f'>>> TRACKING PROCESSOR: pos=({result.projection.u:.3f},{result.projection.v:.3f}), state={result.tracking_state}')
        return result
    controller.tracking_processor.process = debug_tp_process

time.sleep(5)
controller.stop()