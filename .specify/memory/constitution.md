<!--
Sync Impact Report:
- Version: 1.0.0 (initial)
- Principles added: 17 (all new)
- Sections added: Core Principles, Development Constraints, Development Workflow, Governance
- TODOs: RATIFICATION_DATE (mark as today), LAST_AMENDED_DATE (today)
-->

# AirMouse Constitution

## Core Principles

### I. Linux-First Development
AirMouse is developed for Linux as the primary platform. All core features, input backends, and system integrations target Linux first. Wayland and X11 compatibility are mandatory; no feature may rely on Windows- or macOS-specific APIs without a Linux-native equivalent. Cross-platform support is a long-term goal, never a substitute for Linux correctness.

### II. Wayland Compatibility
All input injection, hotkey registration, and window management must work on Wayland compositors (GNOME, KDE, COSMIC, Hyprland, sway, etc.). The xdg-desktop-portal backend is preferred for hotkeys; uinput is used for pointer injection. X11 fallbacks are acceptable but must not be the only path. No feature may assume X11-only APIs.

### III. Low CPU and RAM Usage
The vision pipeline, gesture engine, and input layer must run within a tight resource budget. Target: <5% CPU on a modern mid-range laptop (e.g., AMD Ryzen 3 3250U) at 30 FPS; <200 MB RSS. MediaPipe model complexity, camera resolution, and processing frequency are tunable via profiles to meet this budget on lower-end hardware. Unused subsystems (e.g., face tracking, brightness control) must be fully disableable.

### IV. Low-Latency Hand Tracking
End-to-end latency from camera frame to cursor movement must be minimized. Target: <50 ms at 30 FPS, <35 ms at 60 FPS. The pipeline uses a pull-based frame loop with minimal buffering. MediaPipe inference runs asynchronously; landmark results are consumed immediately. No stage may block the main loop for >5 ms. Frame drops are logged and measurable.

### V. Smooth Cursor Movement
Cursor motion must feel natural, not robotic. The pipeline applies smoothing (One Euro Filter or EMA) on normalized plane coordinates, not raw camera landmarks. Acceleration curves map slow hand movements to fine control and fast movements to rapid traversal. Sensitivity modes (Precision/Normal/Fast) are user-selectable. The One Euro Filter is the default for adaptive smoothing across speeds.

### VI. Cursor Stabilization
When the hand is stationary, the cursor must not drift or jitter. A dead zone (configurable radius, default 0.02 normalized) suppresses micro-movements. Velocity limiting caps maximum cursor speed (default 2000 px/s normal, 500 px/s precision) and acceleration. Stabilization frames (default 10) are required after tracking loss before cursor updates resume.

### VII. Reduced Cursor Vibration
High-frequency noise from camera quantization and landmark jitter is attenuated by the One Euro Filter's minimum cutoff and derivative cutoff parameters. The filter adapts to movement speed: more smoothing at low speed, less at high speed. Velocity limiting prevents sudden spikes. All filter parameters are profile-configurable and tested against synthetic vibration benchmarks.

### VIII. Reliable Gesture Recognition
Gestures use multi-threshold hysteresis to prevent false triggers. Pinch click requires: ENTER threshold (0.045) → CONFIRM threshold (0.040) → RELEASE threshold (0.070). Minimum hold durations (e.g., 0.15 s for click, 0.3 s for drag) and stable frame counts (default 3) are enforced. All thresholds are documented with rationale, benchmarks, and test coverage.

### IX. Minimizing False Clicks
False clicks are prevented by: (a) three-threshold pinch hysteresis, (b) minimum gesture duration, (c) velocity gating (no click while moving fast), (d) gesture confirmation requiring stable landmarks, (e) emergency corner escape (hold cursor in any screen corner 0.5 s), (f) global hotkey (Super+Alt+A) for immediate disable. Click behavior is tested with adversarial hand motion.

### X. Slow Movement → Precision
When hand velocity is below a configurable threshold, the system enters precision mode: sensitivity drops (default 12%), dead zone tightens, smoothing increases, max velocity caps at 500 px/s. This mode can also be triggered explicitly via a secondary hand gesture (two-hand precision) or hotkey (Super+Alt+M). Precision mode is indicated in the UI and tray.

### XI. Fast Movement → Acceleration
When hand velocity exceeds a threshold, an acceleration curve (default exponent 1.2) expands the cursor travel distance non-linearly. This enables crossing large screens without excessive hand motion. Acceleration is capped by max velocity limits. The curve is profile-configurable and benchmarked for overshoot.

### XII. Stationary Hand → Stabilization
After N frames (default 10) of hand position within the dead zone, the cursor position is snapped to the filtered coordinate and further micro-movements are suppressed until the dead zone is exited. This eliminates "breathing" cursor drift. The stabilization frame count is profile-configurable.

### XIII. Universal Application Profiles
Configuration profiles (JSON) define complete cursor, tracking, gesture, and brightness settings. Profiles are selected manually, via hotkey, or automatically by application window class (future). Built-in profiles: "precision", "normal", "fast", "presentation", "accessibility". Profiles are portable, versioned, and validated against a JSON schema. The active profile persists across restarts.

### XIV. Background Operation
AirMouse runs as a background service with optional GUI. The main loop starts automatically on launch; the GUI window can be closed while tracking continues. System tray provides status, pause/resume, settings, and quit. No user-facing window is required for operation. Startup via systemd/user unit or autostart .desktop is supported.

### XV. Settings UI Opened Only Through Configured Shortcut
The settings dialog is not accessible from the main window toolbar by default. It opens only via: (a) global hotkey (Super+Alt+S), (b) system tray menu, (c) CLI flag `--settings`, (d) explicit user action in the UI. This prevents accidental settings changes during use. The main window shows only camera preview, controls, and status.

### XVI. Modular Architecture
The codebase separates concerns into independent modules: camera, vision (hand/face tracking), projection (virtual plane), gestures, cursor mapping, input (uinput), config, UI, safety. Each module has a defined interface (Coordinate Contract for vision→cursor, GestureEvent for gestures→input). Modules are independently testable with synthetic inputs. No circular dependencies. New tracking backends or input backends can be swapped without modifying core logic.

### XVII. Testability and Production-Quality Implementation
All critical paths have unit tests (pipeline stages, coordinate transforms, gesture logic, smoothing filters, velocity limiters). Integration tests cover: camera→landmarks→projection→cursor→uinput, head-movement invariance, two-hand precision mode, gesture hysteresis. Benchmarks track FPS, latency, CPU, memory. CI runs tests on every commit. Code uses type hints, structured logging, and explicit error handling. No `print` in production code; all output goes through `logging`.

### XVIII. Accessibility and Safety
Safety is layered: (1) corner escape (hold cursor in corner 0.5 s → emergency stop), (2) global hotkey (Super+Alt+A → immediate disable, releases all buttons), (3) focus loss auto-pause (X11/Wayland), (4) velocity/acceleration limiting, (5) gesture confirmation (stable frames + duration), (6) inactivity timeout (optional). Emergency stop releases all mouse buttons and notifies the user. The system fails gracefully: if camera unavailable, logs error and retries; if uinput unavailable, falls back to XTest/portal or reports actionable error. Accessibility profile provides larger dead zones, lower sensitivity, voice feedback hooks (future).

### XIX. Graceful Failure on Device Unavailability
Camera: if `/dev/video*` missing or busy, log clear error, show UI notification, retry every 5 s. uinput: if `/dev/uinput` missing or unwritable, instruct user to `modprobe uinput` and add to `input` group; fall back to XTest if on X11. MediaPipe models: if `.task` files missing, download or prompt user. All errors are actionable, not generic. The application never crashes silently on device loss; it enters a safe paused state.

## Development Constraints

### Technology Stack
- **Language**: Python 3.10+
- **Computer Vision**: MediaPipe (hand/face landmarkers), OpenCV (camera I/O)
- **Input**: Linux uinput (primary), XTest / xdg-desktop-portal (fallback)
- **GUI**: PySide6 (Qt6) — optional, background-first
- **Config**: JSON Schema-validated profiles, Pydantic-style dataclasses
- **Testing**: pytest, hypothesis for property-based tests, synthetic landmark generators
- **Benchmarking**: Custom harnesses (`benchmark_*.py`) measuring FPS, latency, CPU, memory

### Performance Standards
- Target FPS: ≥30 on 720p, ≥60 on 480p (configurable)
- End-to-end latency: <50 ms (30 FPS), <35 ms (60 FPS)
- CPU: <5% on Ryzen 3 3250U / equivalent
- Memory: <200 MB RSS
- Startup time: <2 s to first frame processed

### Code Quality
- Type hints on all public functions and dataclasses
- Structured logging (JSON-compatible) with `logging` module
- No global mutable state outside config singletons
- Explicit contracts (Coordinate Contract) between pipeline stages
- All thresholds and magic numbers documented with § references to IDEA2.md

## Development Workflow

### Branching and Commits
- `main` branch is always deployable
- Feature branches named `feature/<short-description>`
- Fix branches named `fix/<issue-id>-<short-description>`
- Commit messages: `<scope>: <imperative summary>` (e.g., `cursor: add One Euro Filter smoothing`)
- Conventional commits encouraged

### Testing Gates
- Unit tests pass on every commit (CI)
- Integration tests pass on PR merge
- Benchmarks run on PR; regression >5% fails
- Property-based tests for coordinate transforms and gesture logic

### Release Process
- Version: MAJOR.MINOR.PATCH (semantic)
- CHANGELOG.md updated per release
- Profiles versioned with schema version
- `.task` model files versioned separately

## Governance

This constitution supersedes all other development practices for the AirMouse project. Amendments require:
1. A proposal documenting the change and rationale
2. Review against all 19 principles — no principle may be weakened without explicit justification
3. Update to this constitution with version bump (MINOR for new principles, PATCH for clarifications)
4. Migration plan for affected code/config/tests
5. Ratification by project maintainers

**Version**: 1.0.0 | **Ratified**: 2026-10-03 | **Last Amended**: 2026-10-03