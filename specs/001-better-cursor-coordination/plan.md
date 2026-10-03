# Implementation Plan: Better Cursor Coordination

**Branch**: `001-better-cursor-coordination` | **Date**: 2026-10-03 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/001-better-cursor-coordination/spec.md`

## Summary

Fix the cursor coordination pipeline by reordering stages: MediaPipe landmarks → virtual plane projection → smoothing (One Euro Filter on plane coordinates) → velocity limiting → cursor output. This addresses three critical bugs from the architecture audit: (1) smoothing before projection destroys head-relative geometry, (2) reference point not updated in head-relative mode, (3) VelocityLimiter contract mismatch treating deltas as positions.

**Status**: All 8 phases complete. 38/38 tasks done (T001-T036). 200 tests passing.

## Technical Context

**Language/Version**: Python 3.10+

**Primary Dependencies**: 
- MediaPipe (hand/face landmarkers)
- OpenCV (camera I/O)
- NumPy (math)
- PySide6 (GUI, optional)
- Linux uinput (cursor injection)

**Storage**: JSON configuration profiles (`~/.airmouse/profiles/`), model files (`.task`)

**Testing**: pytest, hypothesis (property-based), synthetic landmark generators, custom benchmarks (`benchmark_*.py`)

**Target Platform**: Linux (X11 and Wayland), primary: Pop!_OS / Ubuntu-based

**Project Type**: Desktop application (background service + optional GUI)

**Performance Goals**: 
- End-to-end latency <50ms at 30 FPS, <35ms at 60 FPS
- CPU <5% on AMD Ryzen 3 3250U equivalent
- Memory <200 MB RSS
- Jitter: cursor position std dev <1 pixel with stationary hand

**Constraints**: 
- Must maintain backward compatibility with legacy camera-coordinate mode
- Must work without face tracking (graceful degradation)
- No global mutable state outside config singletons
- All thresholds documented with § references to IDEA2.md

**Scale/Scope**: Single-user desktop application, 10+ modules affected in `airmouse/` package

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Linux-First Development | ✅ Pass | All changes Linux-native |
| II. Wayland Compatibility | ✅ Pass | No X11-only changes |
| III. Low CPU/RAM Usage | ✅ Pass | Pipeline reordering reduces redundant computation |
| IV. Low-Latency Hand Tracking | ✅ Pass | Projection before smoothing reduces frame processing |
| V. Smooth Cursor Movement | ✅ Pass | One Euro Filter on plane coordinates is correct approach |
| VI. Cursor Stabilization | ✅ Pass | Dead zone and velocity limiting preserved |
| VII. Reduced Cursor Vibration | ✅ Pass | Adaptive One Euro Filter on plane coords |
| VIII. Reliable Gesture Recognition | ✅ Pass | Unaffected - gestures use plane coordinates |
| IX. Minimizing False Clicks | ✅ Pass | Unaffected |
| X. Slow Movement → Precision | ✅ Pass | Precision mode uses plane coordinates |
| XI. Fast Movement → Acceleration | ✅ Pass | Acceleration on plane coordinates |
| XII. Stationary Hand → Stabilization | ✅ Pass | Stabilization frames on plane coordinates |
| XIII. Universal Application Profiles | ✅ Pass | New config params added to profile schema |
| XIV. Background Operation | ✅ Pass | Core pipeline unchanged |
| XV. Settings UI via Shortcut | ✅ Pass | No UI changes needed |
| XVI. Modular Architecture | ✅ Pass | Coordinate Contract enforced between stages |
| XVII. Testability | ✅ Pass | Synthetic tests for each pipeline stage |
| XVIII. Accessibility & Safety | ✅ Pass | Emergency stop, corner escape unchanged |
| XIX. Graceful Device Failure | ✅ Pass | Face tracking loss → legacy mode fallback |

**No violations** - all principles satisfied.

## Project Structure

### Documentation (this feature)

```text
specs/001-better-cursor-coordination/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
└── tasks.md             # Phase 2 output (later)
```

### Source Code (repository root)

```text
airmouse/
├── control/
│   ├── coordinate_contract.py      # Coordinate space definitions (EXISTING - extended)
│   ├── cursor.py                   # Cursor mapping & smoothing (MODIFY - pipeline reorder)
│   ├── main_loop.py                # Main tracking loop (MODIFY - integration)
│   └── smoothing.py                # SmoothingFilter ABC + OneEuroFilter/EmaFilter (NEW)
├── vision/
│   ├── tracking_processor.py       # Hand/face tracking pipeline (MODIFY - Bug 1, 2, 3 fixes)
│   ├── head_coords.py              # Head coordinate system (MODIFY - reference point)
│   ├── virtual_plane.py            # Virtual plane math (MODIFY - projection logic)
│   ├── projection.py               # Virtual plane projection (MODIFY - core fix)
│   └── gestures.py                 # Gesture recognition (MAYBE - uses plane coords)
├── ui/
│   ├── hotkeys.py                  # Hotkey registration (MODIFY - Super+Alt+M runtime toggle)
│   ├── main_window.py              # MAYBE - settings for new config params
│   └── safety.py                   # VelocityLimiter + SafetyEvent (EXISTING - extended)
├── config/
│   ├── schema.py                   # Config schema (MODIFY - new params)
│   └── profiles.py                 # Profile definitions (MODIFY - new defaults)
├── tests/
│   ├── test_projection.py          # MODIFY - head-movement invariance tests
│   ├── test_cursor_smoothing.py    # MODIFY - plane coordinate smoothing tests
│   ├── test_virtual_plane.py       # MODIFY - projection tests
│   ├── test_velocity_limiter.py    # NEW - velocity limiter tests
│   └── test_head_relative.py       # NEW - head-relative mode tests
└── debug/
    └── diagnostics.py              # MODIFY - pipeline stage timing logs
```

**Structure Decision**: Modular monorepo - all changes within existing `airmouse/` package. No new top-level directories.

## Complexity Tracking

> No Constitution Check violations - no justification needed.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| (none) | | |