import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

model_path = 'face_landmarker.task'

for w, h in [(640, 480), (480, 360), (320, 240), (160, 120)]:
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    mp_image.image_dimensions = (w, h)

    for i in range(5):
        landmarker.detect(mp_image)

    print(f'warmup done for {w}x{h}')
    t0 = time.perf_counter()
    for i in range(20):
        landmarker.detect(mp_image)
    t1 = time.perf_counter()
    print(f'{w}x{h}: {(t1-t0)*1000:.1f}ms total, {(t1-t0)*50:.1f}ms/frame, FPS={20/(t1-t0):.1f}')

    landmarker.close()