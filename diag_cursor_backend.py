#!/usr/bin/env python3
"""
Direct cursor backend diagnostic — bypasses all vision/camera/MediaPipe.

Tests the production InputBackend directly:
  - Initializes the LinuxInputManager
  - Reports which backend was selected
  - Sends relative movement commands
  - Reports success/failure at each stage
"""

import os
import sys
import time
import logging

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from airmouse.input.linux_input import (
    LinuxInputManager,
    InputBackend,
    UInputBackend,
    YdotoolBackend,
    X11Backend,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("diag")


def test_uinput_direct():
    """Test uinput backend directly — no LinuxInputManager."""
    logger.info("=== Direct uinput backend test ===")
    backend = UInputBackend()
    logger.info(f"uinput available: {backend.is_available()} (reason={backend.get_availability_reason()})")

    if not backend.initialize():
        logger.error(f"uinput init FAILED: {backend.get_health_reason()}")
        return False

    logger.info(f"uinput initialized OK (health={backend.get_health_reason()})")

    # Test movement
    logger.info("Testing move(50, 0) → right...")
    ok = backend.move(50, 0)
    logger.info(f"  move(50,0) returned: {ok}")
    time.sleep(0.2)

    logger.info("Testing move(0, 50) → down...")
    ok = backend.move(0, 50)
    logger.info(f"  move(0,50) returned: {ok}")
    time.sleep(0.2)

    logger.info("Testing move(-50, 0) → left...")
    ok = backend.move(-50, 0)
    logger.info(f"  move(-50,0) returned: {ok}")
    time.sleep(0.2)

    logger.info("Testing move(0, -50) → up...")
    ok = backend.move(0, -50)
    logger.info(f"  move(0,-50) returned: {ok}")
    time.sleep(0.2)

    # Test absolute move
    logger.info("Testing move_absolute(500, 300)...")
    ok = backend.move_absolute(500, 300)
    logger.info(f"  move_absolute(500,300) returned: {ok}")
    time.sleep(0.2)

    # Test click
    logger.info("Testing click(1) → left click...")
    ok = backend.click(1)
    logger.info(f"  click(1) returned: {ok}")
    time.sleep(0.2)

    backend.cleanup()
    logger.info("uinput backend cleaned up")
    return True


def test_linux_input_manager():
    """Test the full LinuxInputManager — which backend gets selected."""
    logger.info("=== LinuxInputManager test ===")
    manager = LinuxInputManager()

    logger.info(f"DE: {manager.detect_desktop_environment().value}")
    logger.info(f"Session: {os.environ.get('XDG_SESSION_TYPE', 'unknown')}")
    logger.info(f"WAYLAND_DISPLAY: {os.environ.get('WAYLAND_DISPLAY', 'not set')}")
    logger.info(f"DISPLAY: {os.environ.get('DISPLAY', 'not set')}")

    if not manager.initialize():
        logger.error(f"Manager init FAILED: {manager.get_health_reason()}")
        return False

    backend_type = manager.get_backend_type()
    caps = manager.get_capabilities()
    logger.info(f"Selected backend: {backend_type.value}")
    logger.info(f"Health: {manager.get_health_reason()}")
    logger.info(f"Capabilities: {caps}")

    # Test movement
    logger.info("Testing move(100, 0) → right...")
    ok = manager.move(100, 0)
    logger.info(f"  returned: {ok}")
    time.sleep(0.3)

    logger.info("Testing move(0, 100) → down...")
    ok = manager.move(0, 100)
    logger.info(f"  returned: {ok}")
    time.sleep(0.3)

    logger.info("Testing move(-100, 0) → left...")
    ok = manager.move(-100, 0)
    logger.info(f"  returned: {ok}")
    time.sleep(0.3)

    logger.info("Testing move(0, -100) → up...")
    ok = manager.move(0, -100)
    logger.info(f"  returned: {ok}")
    time.sleep(0.3)

    # Test absolute
    logger.info("Testing move_absolute(800, 500)...")
    ok = manager.move_absolute(800, 500)
    logger.info(f"  returned: {ok}")
    time.sleep(0.3)

    manager.cleanup()
    logger.info("Manager cleaned up")
    return True


def test_ydotool_direct():
    """Test ydotool backend directly."""
    logger.info("=== Direct ydotool backend test ===")
    backend = YdotoolBackend()
    logger.info(f"ydotool available: {backend.is_available()} (reason={backend.get_availability_reason()})")

    if not backend.initialize():
        logger.error(f"ydotool init FAILED: {backend.get_health_reason()}")
        return False

    logger.info(f"ydotool initialized OK (health={backend.get_health_reason()})")

    logger.info("Testing move(50, 0)...")
    ok = backend.move(50, 0)
    logger.info(f"  returned: {ok}")
    time.sleep(0.5)

    backend.cleanup()
    return True


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("CURSOR BACKEND DIAGNOSTIC")
    logger.info("=" * 60)

    # Test 1: Direct uinput
    logger.info("\n--- Test 1: Direct uinput backend ---")
    try:
        test_uinput_direct()
    except Exception as e:
        logger.exception(f"uinput test failed: {e}")

    time.sleep(1)

    # Test 2: LinuxInputManager (auto-select)
    logger.info("\n--- Test 2: LinuxInputManager (auto-select) ---")
    try:
        test_linux_input_manager()
    except Exception as e:
        logger.exception(f"manager test failed: {e}")

    time.sleep(1)

    # Test 3: Direct ydotool
    logger.info("\n--- Test 3: Direct ydotool backend ---")
    try:
        test_ydotool_direct()
    except Exception as e:
        logger.exception(f"ydotool test failed: {e}")

    logger.info("\n=== Diagnostic complete ===")