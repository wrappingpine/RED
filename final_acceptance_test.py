#!/usr/bin/env python3
"""Final acceptance test for cursor output pipeline."""
import sys
sys.path.insert(0, '/home/shubham/airmouse')
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
from airmouse.input.linux_input import LinuxInputManager
from airmouse.control.cursor import CursorController, CursorConfig, CursorMode
import time

print("="*70)
print("FINAL ACCEPTANCE TEST - Cursor Output Pipeline")
print("="*70)

# Test 1: Backend health
print("\n[1] Backend health check")
mgr = LinuxInputManager()
mgr.initialize()
healthy = mgr.is_healthy()
backend = mgr.get_backend_type()
reason = mgr.get_health_reason()
print(f"  Backend: {backend}")
print(f"  Healthy: {healthy}")
print(f"  Reason: {reason}")
test1 = healthy and backend.name == 'UINPUT'
print(f"  Result: {'PASS' if test1 else 'FAIL'}")

# Test 2: Basic movement
print("\n[2] Basic cursor movement")
moves = [(50, 0), (0, 50), (-50, 0), (0, -50)]
test2 = True
for dx, dy in moves:
    ok = mgr.move(dx, dy)
    print(f"  move({dx},{dy}) -> {'OK' if ok else 'FAIL'}")
    test2 = test2 and ok
    time.sleep(0.1)
print(f"  Result: {'PASS' if test2 else 'FAIL'}")

# Test 3: Projection pipeline
print("\n[3] Projection pipeline integration")
config = CursorConfig()
controller = CursorController(config, mode=CursorMode.VIRTUAL_PLANE)
controller._screen_width = 1920
controller._screen_height = 1080
controller._reference_point = (0.5, 0.5)
controller._last_plane_position = (0.5, 0.5)
controller._last_time = time.monotonic()

coords = [(0.55, 0.5), (0.6, 0.5), (0.65, 0.5)]
test3 = True
for u, v in coords:
    move = controller.get_relative_movement_from_plane(u, v)
    print(f"  Plane ({u},{v}) -> movement {move}")
    if move and (move[0] != 0 or move[1] != 0):
        ok = mgr.move(*move)
        test3 = test3 and ok
        time.sleep(0.1)
print(f"  Result: {'PASS' if test3 else 'FAIL'}")

mgr.cleanup()

print("\n" + "="*70)
if test1 and test2 and test3:
    print("ACCEPTANCE TEST: PASS - Cursor output pipeline working")
    print("Physical cursor should be moving during tests")
else:
    print("ACCEPTANCE TEST: FAIL - Issues detected")
print("="*70)