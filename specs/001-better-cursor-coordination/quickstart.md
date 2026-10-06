# Quickstart Validation Guide

**Feature**: 001-better-cursor-coordination
**Date**: 2026-10-03

## Prerequisites

```bash
# Clone and setup
cd /home/shubham/airmouse
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Ensure uinput permissions
sudo usermod -aG input $USER
# Log out and back in, or:
newgrp input

# Load uinput module
sudo modprobe uinput
```

## Validation Scenarios

### Scenario 1: Basic Direction Correctness (SC-001)

**Purpose**: Verify cursor moves in same direction as hand.

**Setup**: 
```bash
# Run with synthetic landmarks (no camera needed)
python -m airmouse.tests.test_projection --synthetic --directions
```

**Test**: Feed synthetic hand landmarks moving in 4 cardinal directions + diagonals.

**Expected**: 
- Hand right → cursor right
- Hand left → cursor left
- Hand up → cursor up
- Hand down → cursor down
- Diagonals match

**Pass Criteria**: 100% direction match in synthetic test.

---

### Scenario 2: Head-Movement Invariance (SC-002)

**Purpose**: Verify cursor stays stable when head moves but hand fixed relative to head.

**Setup**:
```bash
python -m pytest airmouse/tests/test_head_relative.py -v
```

**Test**: 
1. Fix synthetic hand at (0.5, 0.5) in plane coordinates
2. Rotate head pose ±30° yaw, ±20° pitch
3. Translate head ±0.1m in X,Y,Z
4. Run 100 frames per pose

**Expected**: Cursor position drift <2 pixels RMS

**Pass Criteria**: Max drift <2 pixels across all head poses.

---

### Scenario 3: End-to-End Latency (SC-003)

**Purpose**: Measure camera-to-cursor latency.

**Setup**:
```bash
python -m airmouse.benchmark_full --latency --frames 300
```

**Test**: Timestamp at camera frame capture → timestamp at uinput write.

**Expected**: 
- P50 latency <35ms at 60 FPS
- P95 latency <50ms at 30 FPS
- P99 latency <60ms

**Pass Criteria**: P95 <50ms at configured FPS.

---

### Scenario 4: Jitter Reduction (SC-004)

**Purpose**: Verify stationary hand produces stable cursor.

**Setup**:
```bash
python -m airmouse.tests.test_cursor_smoothing --jitter --noise 0.01 --frames 1000
```

**Test**: Feed synthetic landmarks with Gaussian noise (σ=0.01 normalized), hand stationary at plane center.

**Expected**: Cursor position standard deviation <1 pixel

**Pass Criteria**: σ_x <1px, σ_y <1px over 1000 frames.

---

### Scenario 5: Velocity Limiting (SC-005)

**Purpose**: Verify max velocity is respected.

**Setup**:
```bash
python -m airmouse.tests.test_velocity_limiter --max-vel 2000 --test-vel 5000
```

**Test**: Feed synthetic movement equivalent to 5000 px/s with max_velocity=2000.

**Expected**: Output velocity capped at 2000 ±5% (1900-2100 px/s)

**Pass Criteria**: Measured velocity in [1900, 2100] px/s.

---

### Scenario 6: Legacy Mode Regression (SC-006)

**Purpose**: Ensure legacy camera-coordinate mode still works.

**Setup**:
```bash
python -m airmouse.tests.test_projection --legacy-mode
```

**Test**: Run existing projection tests with `use_head_relative=False`.

**Expected**: All existing tests pass, behavior identical to pre-fix version.

**Pass Criteria**: 100% test pass rate, pixel-perfect output match on recorded sequences.

---

### Scenario 7: Full Pipeline Integration (SC-007)

**Purpose**: End-to-end test with real camera.

**Setup**:
```bash
# Run air mouse with debug overlay
./run.sh --debug --duration 30
```

**Test**: 
1. Start air mouse
2. Move hand naturally for 30 seconds
3. Verify: no crashes, cursor follows hand, smooth motion
4. Check logs for pipeline stage timings

**Expected**: 
- No exceptions in logs
- Stage timings logged every second
- FPS ≥30
- CPU <5%

**Pass Criteria**: Runs 30s without error, meets performance targets.

---

## Running All Validations

```bash
# Run all test suites
cd /home/shubham/airmouse
python -m pytest airmouse/tests/ -v -k "projection or cursor_smoothing or velocity_limiter or head_relative" 2>&1 | tee test_results.log

# Run benchmarks
python benchmark_full.py --iterations 5 2>&1 | tee benchmark_results.log

# Check for regressions
python -m pytest airmouse/tests/ --tb=short 2>&1 | tail -20
```

## Expected Outputs

### test_results.log
```
============================= test session starts =============================
test_projection.py::test_camera_to_plane_mapping PASSED
test_projection.py::test_head_movement_invariance PASSED
test_projection.py::test_legacy_mode_unchanged PASSED
test_cursor_smoothing.py::test_one_euro_filter_adaptive PASSED
test_cursor_smoothing.py::test_jitter_reduction PASSED
test_velocity_limiter.py::test_velocity_capping PASSED
test_velocity_limiter.py::test_no_overshoot PASSED
test_head_relative.py::test_reference_point_every_frame PASSED
test_head_relative.py::test_fallback_to_legacy PASSED
========================= 9 passed in 12.34s =================================
```

### benchmark_results.log
```
Benchmark: 5 iterations
FPS:        mean=42.3, std=1.2, min=40.1, max=44.0
Latency:    P50=28ms, P95=42ms, P99=48ms
CPU:        mean=3.8%, max=4.5%
Memory:     RSS=156MB, peak=162MB
Jitter:     σ_x=0.7px, σ_y=0.6px (stationary)
Velocity:   max=1998px/s (configured 2000)
```

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| "uinput permission denied" | User not in input group | `sudo usermod -aG input $USER` + relogin |
| "No camera found" | /dev/video* missing | Check `v4l2-ctl --list-devices` |
| "Face tracking failed" | Lighting/angle | Improve lighting, face camera |
| "Import error MediaPipe" | Wrong version | `pip install mediapipe==0.10.14` |
| Tests hang | No display/headless | Use `xvfb-run -a pytest ...` |

## CI Integration

Add to `.github/workflows/ci.yml`:
```yaml
- name: Run cursor coordination tests
  run: |
    xvfb-run -a python -m pytest airmouse/tests/ -k "projection or cursor_smoothing or velocity_limiter or head_relative"
- name: Run benchmarks
  run: |
    python benchmark_full.py --iterations 3 --ci
```

## Manual Verification Checklist

- [ ] Cursor moves right when index finger moves right
- [ ] Cursor moves up when index finger moves up  
- [ ] No mirroring/inversion (unless configured)
- [ ] Head movement doesn't move cursor (hand fixed to head)
- [ ] Smooth motion, no visible jitter
- [ ] Fast hand movement → accelerated cursor
- [ ] Slow hand movement → precise cursor
- [ ] Pinch click works
- [ ] Corner escape (50px, 0.5s) triggers emergency stop
- [ ] Super+Alt+A triggers emergency stop
- [ ] Settings open via Super+Alt+S (not main window)
- [ ] System tray shows status, pause/resume, quit
- [ ] Closing window minimizes to tray, doesn't stop tracking