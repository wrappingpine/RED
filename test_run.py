from airmouse.control.main_loop import AirMouseController, AirMouseConfig
from airmouse.camera.manager import CameraManager, CameraSettings
from airmouse.vision.hand_tracker import HandTracker, HandTrackerSettings
from airmouse.vision.face_tracker import FaceTracker, FaceTrackerSettings
from airmouse.input.uinput_mouse import VirtualMouse
import logging
import time

logging.basicConfig(level=logging.DEBUG)

print('Initializing Air Mouse...')

camera = CameraManager()
camera.open_camera(CameraSettings(device_index=0, width=640, height=480, fps=30))
hand_tracker = HandTracker(HandTrackerSettings())
face_tracker = FaceTracker(FaceTrackerSettings())
virtual_mouse = VirtualMouse()

config = AirMouseConfig()
config.tracking.use_head_relative = True
config.tracking.use_virtual_plane = True

controller = AirMouseController(config)
controller.camera = camera
controller.hand_tracker = hand_tracker
controller.face_tracker = face_tracker
controller.virtual_mouse = virtual_mouse

if controller.initialize():
    print('Initialized successfully!')
    controller.start()
    print('Started!')
    time.sleep(5)
    controller.stop()
    print('Stopped!')
else:
    print('Initialization failed!')