import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

model_path = 'hand_landmarker.task'
base_options = mp_python.BaseOptions(model_asset_path=model_path)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=1,
)
landmarker = mp_vision.HandLandmarker.create_from_options(options)

for w, h in [(640, 480), (480, 360), (320, 240), (160, 120)]:
    frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    mp_image.image_dimensions = (w, h)

    for i in range(5):
        landmarker.detect_for_video(mp_image, i * 33333)

    print(f'warmup done for {w}x{h}')
    t0 = time.perf_counter()
    ts = 5 * 33333
    for i in range(10):
        landmarker.detect_for_video(mp_image, ts)
        ts += 33333
    t1 = time.perf_counter()
    print(f'{w}x{h}: {(t1-t0)*1000:.1f}ms total, {(t1-t0)*100:.1f}ms/frame, FPS={10/(t1-t0):.1f}')

for i in range(5):
    landmarker.detect_for_video(mp_image, i * 33333)

print('warmup done')
t0 = time.perf_counter()
ts = 5 * 33333  # continue from warmup
for i in range(10):
    landmarker.detect_for_video(mp_image, ts)
    ts += 33333
t1 = time.perf_counter()
print(f'10 frames: {(t1-t0)*1000:.1f}ms total, {(t1-t0)*100:.1f}ms/frame')
landmarker.close()