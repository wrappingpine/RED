#!/usr/bin/env python3
"""Air Mouse - Main entry points for console scripts."""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from airmouse.control.main_loop import AirMouseController, AirMouseConfig
from airmouse.camera.manager import CameraManager, CameraSettings
from airmouse.vision.hand_tracker import HandTracker, HandTrackerSettings
from airmouse.vision.face_tracker import FaceTracker, FaceTrackerSettings
from airmouse.input.uinput_mouse import VirtualMouse


def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def diagnose():
    """Run system diagnostics."""
    setup_logging(logging.INFO)
    print("=" * 50)
    print("Air Mouse System Diagnostics")
    print("=" * 50)

    # Check Python version
    print(f"\nPython: {sys.version}")

    # Check dependencies
    try:
        import cv2
        print(f"OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"OpenCV: NOT FOUND - {e}")

    try:
        import numpy
        print(f"NumPy: {numpy.__version__}")
    except ImportError as e:
        print(f"NumPy: NOT FOUND - {e}")

    try:
        import mediapipe
        print(f"MediaPipe: {mediapipe.__version__}")
    except ImportError as e:
        print(f"MediaPipe: NOT FOUND - {e}")

    try:
        import PySide6
        print(f"PySide6: {PySide6.__version__}")
    except ImportError as e:
        print(f"PySide6: NOT FOUND - {e}")

    try:
        import psutil
        print(f"psutil: {psutil.__version__}")
    except ImportError as e:
        print(f"psutil: NOT FOUND - {e}")

    # Check uinput
    import os
    if os.path.exists('/dev/uinput'):
        print("uinput: AVAILABLE (/dev/uinput exists)")
        if os.access('/dev/uinput', os.W_OK):
            print("uinput: WRITABLE")
        else:
            print("uinput: NOT WRITABLE (need sudo or udev rule)")
    else:
        print("uinput: NOT AVAILABLE (/dev/uinput missing)")

    # Check cameras
    print("\nCamera Detection:")
    for i in range(4):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                print(f"  /dev/video{i}: {w}x{h} - OK")
            else:
                print(f"  /dev/video{i}: OPEN but no frame")
            cap.release()
        else:
            print(f"  /dev/video{i}: NOT FOUND")

    # Check screen
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        print(f"\nScreen: {w}x{h}")
        root.destroy()
    except Exception as e:
        print(f"\nScreen: Could not detect - {e}")

    print("\n" + "=" * 50)
    print("Diagnostics complete")
    print("=" * 50)


def test_camera():
    """Test camera capture."""
    setup_logging(logging.INFO)
    print("Testing camera... Press 'q' to quit")

    camera = CameraManager()
    if not camera.open_camera(CameraSettings(device_index=0, width=640, height=480, fps=30)):
        print("Failed to open camera")
        return

    import cv2
    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret:
                continue

            cv2.imshow("Camera Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.close_camera()
        cv2.destroyAllWindows()


def test_hand():
    """Test hand tracking."""
    setup_logging(logging.INFO)
    print("Testing hand tracking... Press 'q' to quit")

    camera = CameraManager()
    if not camera.open_camera(CameraSettings(device_index=0, width=640, height=480, fps=30)):
        print("Failed to open camera")
        return

    tracker = HandTracker(HandTrackerSettings(max_hands=1))

    import cv2
    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret:
                continue

            hands = tracker.process(frame)

            if hands:
                hand = hands[0]
                print(f"\rHand: {hand.handedness}, "
                      f"Index extended: {hand.is_finger_extended('index')}, "
                      f"Pinch (thumb+index): {hand.is_pinch('thumb', 'index'):.3f}, "
                      f"Fist: {hand.is_fist()}", end="")

                annotated = tracker.draw_landmarks(frame, hands)
            else:
                annotated = frame

            cv2.imshow("Hand Tracking Test", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.close_camera()
        tracker.close()
        cv2.destroyAllWindows()


def test_mouse():
    """Test virtual mouse."""
    setup_logging(logging.INFO)
    print("Testing virtual mouse... Press Ctrl+C to quit")

    mouse = VirtualMouse()
    if not mouse.create():
        print("Failed to create virtual mouse. Need /dev/uinput writable.")
        return

    print("Virtual mouse created. Testing movements...")

    try:
        import time
        # Test movement
        print("Moving right...")
        mouse.move(100, 0)

        time.sleep(0.5)

        print("Moving left...")
        mouse.move(-100, 0)

        time.sleep(0.5)

        # Test click
        print("Left click...")
        mouse.left_click()

        time.sleep(0.5)

        # Test scroll
        print("Scroll up...")
        mouse.scroll(3)

        time.sleep(0.5)

        print("Scroll down...")
        mouse.scroll(-3)

        time.sleep(0.5)

        print("All tests passed!")
    except KeyboardInterrupt:
        pass
    finally:
        mouse.destroy()


def run_gui():
    """Run the GUI application."""
    from airmouse.ui.main_window import run_gui as gui_main
    gui_main()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m airmouse.main <command>")
        print("Commands: diagnose, test-camera, test-hand, test-mouse, gui")
        sys.exit(1)

    command = sys.argv[1]
    if command == "diagnose":
        diagnose()
    elif command == "test-camera":
        test_camera()
    elif command == "test-hand":
        test_hand()
    elif command == "test-mouse":
        test_mouse()
    elif command == "gui":
        run_gui()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)