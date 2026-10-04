#!/usr/bin/env python3
"""Force valid cursor coordinates to test backend."""
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
from airmouse.input.linux_input import LinuxInputManager
from airmouse.control.cursor import CursorController, CursorConfig, CursorMode
import time

print("=== Phase 4: Force valid cursor coordinates ===")

# Test 1: Direct backend movement
mgr = LinuxInputManager()
mgr.initialize()
print(f"Backend: {mgr.get_backend_type()}, healthy: {mgr.is_healthy()}, reason: {mgr.get_health_reason()}")

tests = [(100, 0), (0, 100), (-100, 0), (0, -100)]
for dx, dy in tests:
    ok = mgr.move(dx, dy)
    print(f"move({dx},{dy}) -> {ok}")
    time.sleep(0.2)

# Test 2: CursorController pipeline with forced normalized coords
config = CursorConfig()
controller = CursorController(config, mode=CursorMode.VIRTUAL_PLANE)
controller._screen_width = 1920
controller._screen_height = 1080

# Simulate movement from center
# Force reference point
controller._reference_point = (0.5, 0.5)
controller._last_plane_position = (0.5, 0.5)
controller._last_time = time.monotonic()

coords = [(0.6, 0.5), (0.7, 0.5), (0.8, 0.5), (0.6, 0.6)]
for u, v in coords:
    move = controller.get_relative_movement_from_plane(u, v)
    print(f"Plane ({u},{v}) -> relative movement: {move}")
    if move:
        ok = mgr.move(*move)
        print(f"  Applied -> {ok}")
    time.sleep(0.2)

mgr.cleanup()
print("Phase 4 complete")
