import cv2
import numpy as np
import time
import sys
import argparse
sys.path.insert(0, '.')
from airmouse.camera.manager import CameraManager, CameraSettings
from airmouse.vision.hand_tracker import HandTracker, HandTrackerSettings
from airmouse.vision.face_tracker import FaceTracker, FaceTrackerSettings
import logging
logging.basicConfig(level=logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description='AirMouse latency benchmark')
    parser.add_argument('--iterations', type=int, default=30,
                        help='Frames per resolution (default: 30)')
    parser.add_argument('--ci', action='store_true',
                        help='Emit P95=<ms> line for CI latency gate')
    args = parser.parse_args()

    all_total_times = []

    for w, h in [(640, 480), (320, 240)]:
        camera = CameraManager()
        if not camera.open_camera(CameraSettings(device_index=0, width=w, height=h)):
            print(f'Failed to open camera at {w}x{h}')
            continue

        hand_tracker = HandTracker(HandTrackerSettings(max_hands=1, model_complexity=0))
        face_tracker = FaceTracker(FaceTrackerSettings(max_faces=1))

        # Warmup
        for _ in range(5):
            ret, frame = camera.read_frame()
            if ret:
                hand_tracker.process(frame)
                face_tracker.process(frame)

        print(f'warmup done for {w}x{h}')
        hand_times = []
        face_times = []
        total_times = []
        cam_times = []

        for _ in range(args.iterations):
            t0 = time.perf_counter()
            ret, frame = camera.read_frame()
            if not ret:
                continue
            t_cam = time.perf_counter()

            hands = hand_tracker.process(frame)
            t_hand = time.perf_counter()

            faces = face_tracker.process(frame)
            t_face = time.perf_counter()

            cam_times.append((t_cam - t0) * 1000)
            hand_times.append((t_hand - t_cam) * 1000)
            face_times.append((t_face - t_hand) * 1000)
            total_times.append((t_face - t0) * 1000)

        if total_times:
            all_total_times.extend(total_times)
            print(f'{w}x{h}: cam={np.mean(cam_times):.1f}ms '
                  f'hand={np.mean(hand_times):.1f}ms '
                  f'face={np.mean(face_times):.1f}ms '
                  f'total={np.mean(total_times):.1f}ms '
                  f'(FPS={1000/np.mean(total_times):.1f})')

        camera.close_camera()
        hand_tracker.close()
        face_tracker.close()

    if all_total_times and args.ci:
        p95 = float(np.percentile(all_total_times, 95))
        print(f'P95={p95:.1f}')


if __name__ == '__main__':
    main()