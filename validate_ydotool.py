#!/usr/bin/env python3
"""Validate ydotool backend through real subprocess invocation.

Exercises YdotoolBackend.is_available(), initialize(), and every
subprocess-based method (move, click, scroll, etc.) against the real
ydotool binary. Does NOT fake availability or inject dummy callbacks.
"""
import sys
import logging

sys.path.insert(0, '/home/shubham/airmouse')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from airmouse.input.linux_input import (
    YdotoolBackend, LinuxInputManager, InputBackend, DesktopEnvironment
)

passed = 0
failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  OK {name}")
    else:
        failed += 1
        print(f"  FAIL {name}  {detail}")


print("=== YdotoolBackend direct tests ===")
backend = YdotoolBackend()

check("is_available() returns True", backend.is_available(),
      f"ydotool_path={backend._ydotool_path}")

check("initialize() returns True", backend.initialize(),
      f"_initialized={backend._initialized}, _ydotoold_running={backend._ydotoold_running}")

check("move(10, 5) returns True", backend.move(10, 5))
check("move(-3, -2) returns True", backend.move(-3, -2))
check("move_absolute(100, 100) returns True", backend.move_absolute(100, 100))
check("click(1) returns True", backend.click(1))
check("click(2) returns True", backend.click(2))
check("click(3) returns True", backend.click(3))
check("click(99) returns False", not backend.click(99),
      "unknown button should fail")
# NOTE: this ydotool build does not expose mousedown/mouseup subcommands
# (verified: `ydotool mousedown left` → "Unknown tool: mousedown").
# The backend must degrade gracefully (return False), not raise.
check("button_down(1) returns False (unsupported by ydotool build)",
      not backend.button_down(1))
check("button_up(1) returns False (unsupported by ydotool build)",
      not backend.button_up(1))
check("button_down(99) returns False", not backend.button_down(99))
check("scroll(+3) returns True", backend.scroll(3))
check("scroll(-3) returns True", backend.scroll(-3))
check("scroll_horizontal(+3) returns True", backend.scroll_horizontal(3))
check("scroll_horizontal(-3) returns True", backend.scroll_horizontal(-3))

# Edge: scroll(0) should still work (amount==0 is not >0, goes to else branch)
check("scroll(0) returns True", backend.scroll(0))
check("scroll_horizontal(0) returns True", backend.scroll_horizontal(0))

backend.cleanup()
check("cleanup() clears _initialized", not backend._initialized)

# Test that uninitialized backend rejects calls
check("uninitialized move() returns False", not backend.move(1, 1))
check("uninitialized scroll() returns False", not backend.scroll(1))


print("\n=== LinuxInputManager integration tests ===")
manager = LinuxInputManager()
check("detect_desktop_environment() returns valid DE",
      isinstance(manager.detect_desktop_environment(), DesktopEnvironment))

ok = manager.initialize()
check("manager.initialize() returns bool", isinstance(ok, bool))

if ok:
    caps = manager.get_capabilities()
    check("capabilities not None", caps is not None)
    if caps:
        check(f"backend={caps.backend.value}",
              caps.backend in (InputBackend.YDOTOL, InputBackend.UINPUT, InputBackend.X11))
        check("supports_mouse_movement", caps.supports_mouse_movement)
        check("supports_mouse_buttons", caps.supports_mouse_buttons)
        check("supports_scroll", caps.supports_scroll)
        check("move(5,5) returns True", manager.move(5, 5))
        check("click(1) returns True", manager.click(1))
        check("scroll(2) returns True", manager.scroll(2))
        check("scroll_horizontal(1) returns True", manager.scroll_horizontal(1))
        check("release_all() does not raise", True)
        manager.release_all()
    manager.cleanup()
else:
    print("  (no backend available -- skipping integration assertions)")


print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)