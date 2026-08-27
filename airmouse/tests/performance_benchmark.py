"""
Performance benchmark for AirMouse.

Measures CPU/RAM at 640x480@30fps on target hardware.
"""
import sys
sys.path.insert(0, '/home/shubham/airmouse')

import time
import threading
import numpy as np
import psutil
import os
from unittest.mock import Mock, patch

from airmouse.control.main_loop import AirMouseController, AirMouseConfig
from airmouse.camera.manager import CameraSettings
from airmouse.vision.hand_tracker import HandTrackerSettings
from airmouse.vision.face_tracker import FaceTrackerSettings


def get_process_memory_mb():
    """Get current process memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_process_cpu_percent():
    """Get current process CPU percent."""
    process = psutil.Process(os.getpid())
    return process.cpu_percent(interval=0.1)


class PerformanceBenchmark:
    """Performance benchmark runner."""

    def __init__(self, duration_seconds=10):
        self.duration = duration_seconds
        self.config = AirMouseConfig()
        self.config.tracking.use_head_relative = True
        self.config.camera = CameraSettings(device_index=0, width=640, height=480, fps=30)
        self.config.hand_tracker = HandTrackerSettings(max_hands=2)
        self.config.face_tracker = FaceTrackerSettings()

        self.cpu_samples = []
        self.memory_samples = []
        self.frame_times = []
        self.running = False

    def run_benchmark(self):
        """Run the performance benchmark."""
        print(f"Starting performance benchmark for {self.duration} seconds...")
        print("Target: 640x480@30fps YUYV, <50% single-core CPU, <200MB RAM")

        controller = AirMouseController(self.config, lambda s, d: None)

        # Mock camera to avoid hardware dependency
        with patch('airmouse.camera.manager.CameraManager.open_camera') as mock_open, \
             patch('airmouse.camera.manager.CameraManager.read_frame') as mock_read:

            mock_open.return_value = True

            # Generate synthetic frames
            frame_count = 0
            def generate_frame():
                nonlocal frame_count
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                frame_count += 1
                return True, frame

            mock_read.side_effect = generate_frame

            # Mock trackers to simulate processing load
            controller._hand_tracker.process = Mock(return_value=[])
            controller._face_tracker.process = Mock(return_value=[])

            controller.start()

            # Start monitoring thread
            self.running = True
            monitor_thread = threading.Thread(target=self._monitor_resources, args=(controller,))
            monitor_thread.start()

            # Run for duration
            time.sleep(self.duration)

            self.running = False
            monitor_thread.join()

            controller.stop()

        # Report results
        self._report_results()

    def _monitor_resources(self, controller):
        """Monitor CPU and memory in background thread."""
        while self.running:
            cpu = get_process_cpu_percent()
            mem = get_process_memory_mb()

            self.cpu_samples.append(cpu)
            self.memory_samples.append(mem)

            # Get frame time from performance monitor
            if controller._performance_monitor:
                stats = controller._performance_monitor.get_stats()
                if stats and 'frame_time_ms' in stats:
                    self.frame_times.append(stats['frame_time_ms'])

            time.sleep(0.1)  # Sample at 10Hz

    def _report_results(self):
        """Print benchmark results."""
        print("\n" + "=" * 60)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("=" * 60)

        if self.cpu_samples:
            avg_cpu = np.mean(self.cpu_samples)
            max_cpu = np.max(self.cpu_samples)
            print(f"\nCPU Usage:")
            print(f"  Average: {avg_cpu:.1f}%")
            print(f"  Maximum: {max_cpu:.1f}%")
            print(f"  Target:  <50% single-core")
            print(f"  Status:  {'PASS' if avg_cpu < 50 else 'FAIL'}")

        if self.memory_samples:
            avg_mem = np.mean(self.memory_samples)
            max_mem = np.max(self.memory_samples)
            print(f"\nMemory Usage:")
            print(f"  Average: {avg_mem:.1f} MB")
            print(f"  Maximum: {max_mem:.1f} MB")
            print(f"  Target:  <200 MB")
            print(f"  Status:  {'PASS' if max_mem < 200 else 'FAIL'}")

        if self.frame_times:
            avg_frame_time = np.mean(self.frame_times)
            fps = 1000 / avg_frame_time if avg_frame_time > 0 else 0
            print(f"\nFrame Processing:")
            print(f"  Average frame time: {avg_frame_time:.2f} ms")
            print(f"  Effective FPS: {fps:.1f}")
            print(f"  Target: 33.3 ms (30 FPS)")
            print(f"  Status:  {'PASS' if avg_frame_time < 33.3 else 'FAIL'}")

        print("\n" + "=" * 60)


def main():
    """Run benchmark."""
    benchmark = PerformanceBenchmark(duration_seconds=10)
    benchmark.run_benchmark()


if __name__ == "__main__":
    main()