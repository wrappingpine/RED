# RED — Air Mouse

<p align="center">
  <strong>Turn your camera into a mouse.</strong><br>
  Control your Linux desktop with your hands — no physical mouse required.
</p>

<p align="center">
  <a href="https://github.com/wrappingpine/RED">
    <img src="https://img.shields.io/badge/GitHub-RED-black?style=for-the-badge&logo=github" alt="GitHub">
  </a>
  <img src="https://img.shields.io/badge/Linux-Supported-success?style=for-the-badge&logo=linux" alt="Linux">
  <img src="https://img.shields.io/badge/OpenCV-Vision-blue?style=for-the-badge&logo=opencv" alt="OpenCV">
  <img src="https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange?style=for-the-badge" alt="MediaPipe">
  <img src="https://img.shields.io/badge/Status-Active%20Development-yellow?style=for-the-badge" alt="Status">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

---

## What is RED?

RED is a **camera-based, hands-free mouse for Linux**. It uses computer vision to track your hand and translate your movements and gestures into desktop input.

```
🖐️  Move / Gesture
    │
    ▼
📷 Webcam
    │
    ▼
🧠 Computer Vision (MediaPipe + OpenCV)
    │
    ▼
🎯 Spatial Mapping & Gesture Engine
    │
    ▼
🖱️ Linux Input (Wayland / X11 / uinput)
    │
    ▼
🖥️ Linux Desktop
```

> **No special hardware. No wearable controller. Just a camera and your hand.**

---

## Features

| Feature | Description |
|---------|-------------|
| **Hand Tracking** | Real-time hand landmark detection using MediaPipe Tasks API (VIDEO mode) |
| **Face/Head Tracking** | Head-relative coordinate system for 3D-aware pointing |
| **Cursor Control** | Smooth, low-latency cursor movement with adaptive smoothing (OneEuroFilter + VelocityLimiter) |
| **Gesture Input** | Configurable gestures for click, right-click, drag, scroll, pause/resume |
| **Multi-Hand Support** | Architecture designed for reliable two-hand interaction with stable identity |
| **Linux Desktop Integration** | Wayland-first design with multi-backend fallback (Wayland native, ydotool, X11, uinput) |
| **System Tray** | Cross-desktop support (AppIndicator3, StatusNotifierItem, QSystemTrayIcon) |
| **Global Hotkeys** | Emergency disable (Super+Alt+A), pause/resume, calibrate, settings, precision toggle |
| **Safety System** | Corner escape, velocity limiting, focus loss detection, inactivity timeout |
| **Performance Optimized** | Kalman predictive tracking, frame coordination with timestamps, backpressure handling |

---

## How It Works

RED's processing pipeline is divided into several stages:

```
┌───────────────────────┐
│       📷 CAMERA       │  V4L2 + auto-format (MJPG/YUYV) + health scoring
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│      OpenCV           │  Frame capture, conversion, preprocessing
│    Frame Capture      │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│     MediaPipe         │  Hand Landmarker + Face Landmarker (Tasks API)
│   Hand + Face Vision  │  21 hand landmarks + 468 face landmarks
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│   Spatial Mapping     │  Virtual plane (30cm), ray-plane intersection
│  Camera → Screen      │  Head-relative: eye midpoint = origin
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│   Smoothing &         │  OneEuroFilter + VelocityLimiter + Kalman predictor
│   Prediction          │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ Gesture / State Logic │  Hysteresis state machines (IDLE→DETECTING→CONFIRMED→ACTIVE→RELEASING→COOLDOWN)
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│    Linux Input        │  Multi-backend: Wayland > ydotool > X11 > uinput
└───────────┬───────────┘
            │
            ▼
       🖥️ DESKTOP
```

---

## Interaction Model

| Action | Input |
|--------|-------|
| Move cursor | ☝️ Index finger (head-relative) |
| Left click | 🤏 Pinch (thumb + index) |
| Right click | 🤏 Pinch (thumb + middle) |
| Middle click | 🤏 Pinch (thumb + ring) |
| Drag | Pinch hold + move |
| Scroll | 🤏 Pinch + vertical movement |
| Pause tracking | ✊ Fist |
| Resume tracking | ✋ Open hand |

> The gesture system uses hysteresis state machines to minimize false positives.

---

## Linux Desktop Integration

RED is built **Linux-first** with modern desktop environments in mind:

### Input Backends (Auto-detected, Priority Order)

| Backend | Desktop Environments | Method |
|---------|---------------------|--------|
| **Wayland Native** | GNOME, KDE, COSMIC, Sway, Hyprland | `virtual-pointer-unstable-v1` / `relative-pointer-unstable-v1` |
| **ydotool** | Any Wayland (requires daemon) | `ydotool` socket |
| **X11 (XTest)** | X11 sessions, XWayland | `XTestFakeMotionEvent` |
| **uinput** | All (requires `/dev/uinput` access) | Kernel `/dev/uinput` via ctypes |

### System Tray
- **AppIndicator3** (Primary) — Ubuntu, Pop!_OS, GNOME extensions
- **StatusNotifierItem** (KDE) — Plasma, KDE Neon
- **QSystemTrayIcon** (Fallback) — Generic Qt

### Global Hotkeys
| Hotkey | Action |
|--------|--------|
| `Super+Alt+A` | **Emergency disable** (immediate stop) |
| `Super+Alt+P` | Pause/Resume tracking |
| `Super+Alt+C` | Calibrate |
| `Super+Alt+S` | Open settings |
| `Super+Alt+M` | Toggle precision mode |

### Safety System
- **Corner Escape** — Move cursor to any screen corner → emergency stop
- **Velocity Limit** — Excessive cursor speed → pause
- **Focus Loss** — Application loses focus → pause (configurable)
- **Inactivity Timeout** — No hand detected for N seconds → pause
- **Gesture Timeout** — Gesture held too long → release

---

## Quick Start

### Prerequisites
- Linux (tested on Pop!_OS, Ubuntu, Fedora, Arch)
- Python 3.10+
- Webcam
- **For Wayland**: `libwayland-client`, `wayland-protocols`, `ydotool` (optional)
- **For X11**: `libx11`, `libxtst`
- **For uinput**: User in `input` group (`sudo usermod -a -G input $USER`)

### Installation

```bash
# 1. Clone
git clone https://github.com/wrappingpine/RED.git
cd RED

# 2. Create environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
# Or with optional desktop integration extras:
pip install -e .[wayland,x11,appindicator]

# 4. Run
./run.sh
```

### Model Files
Download MediaPipe task models and place in project root:
- `hand_landmarker.task` — [MediaPipe Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
- `face_landmarker.task` — [MediaPipe Face Landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)

---

## Configuration

RED uses a configuration system with sensible defaults. Key settings:

```python
# Cursor smoothing
smoothing_algorithm: "one_euro"  # or "kalman", "ema", "none"
min_cutoff: 1.0                  # OneEuroFilter minimum cutoff
beta: 0.01                       # OneEuroFilter beta

# Gesture thresholds
pinch_distance_threshold: 0.04   # Normalized distance for pinch
pinch_hold_frames: 3             # Frames to confirm pinch
gesture_cooldown: 0.5            # Seconds between gestures

# Safety
emergency_stop_corner: true      # Enable corner escape
velocity_limit: 5000             # Max pixels/second
inactivity_timeout: 300          # Seconds before auto-pause

# Tracking
use_head_relative: true          # Enable 3D head-relative tracking
virtual_plane_distance: 0.3      # Virtual plane distance (meters)
```

---

## Project Structure

```
RED/
├── airmouse/                    # Core application package
│   ├── camera/                  # Camera management (V4L2, health scoring)
│   ├── control/                 # Main loop, cursor controller
│   ├── input/                   # Linux input backends
│   │   ├── linux_input.py       # Multi-backend LinuxInputManager
│   │   └── uinput_mouse.py      # Legacy uinput direct access
│   ├── ui/                      # Desktop integration
│   │   ├── system_tray.py       # SystemTrayManager (AppIndicator3, SNI, Qt)
│   │   ├── hotkeys.py           # GlobalHotkeyManager (X11, Portal, evdev)
│   │   ├── safety.py            # SafetyManager (corner, velocity, focus, inactivity)
│   │   └── main_window.py       # Qt GUI with tray integration
│   ├── vision/                  # Computer vision pipeline
│   │   ├── hand_tracker.py      # MediaPipe Hand Landmarker
│   │   ├── face_tracker.py      # MediaPipe Face Landmarker
│   │   ├── gestures.py          # GestureRecognizer + hysteresis FSMs
│   │   ├── tracking_processor.py# Frame coordination, smoothing, prediction
│   │   └── virtual_plane.py     # 3D head-relative coordinate system
│   ├── brightness/              # Auto-brightness (backlight + IIO sensors)
│   └── debug/                   # Performance monitoring
├── hand_landmarker.task         # Hand tracking model (download separately)
├── face_landmarker.task         # Face tracking model (download separately)
├── benchmark_*.py               # Performance benchmarks
├── test_*.py                    # Test suite (155 tests)
├── requirements.txt             # Python dependencies
├── setup.py                     # Package configuration
├── install.sh                   # Installation helper
└── run.sh                       # Launcher script
```

---

## Development

### Running Tests
```bash
# All tests
python3 -m pytest airmouse/tests/ -v

# Specific test modules
python3 -m pytest airmouse/tests/test_gesture_hysteresis.py -v
python3 -m pytest airmouse/tests/test_full_pipeline.py -v
```

### Benchmarks
```bash
python3 benchmark_full.py       # Full pipeline benchmark
python3 benchmark_video.py      # Video processing benchmark
```

### Debug Utilities
```bash
python3 test_hand_tracker_gui.py    # Hand/face preview with skeleton
python3 debug_observe_cursor.py     # Cursor position logging
```

---

## Architecture Philosophy

RED separates **vision**, **interpretation**, and **input**:

```
┌────────────────────────────────────┐
│              RED                   │
│                                    │
│  Camera   →  Vision  →  Tracking   │
│                ↓                   │
│            Spatial Mapping         │
│                ↓                   │
│            Gesture Engine          │
│                ↓                   │
│            Input Backend           │
│                                    │
└────────────────────────────────────┘
```

This modular design enables:
- Swapping hand models without touching gesture logic
- Testing input backends independently
- Adding new desktop environments without vision changes
- Benchmarking each subsystem in isolation

---

## Roadmap

### ✅ Completed (Core Tracking)
- [x] Webcam input with V4L2 + format negotiation
- [x] MediaPipe Hand + Face Landmarker integration
- [x] Head-relative 3D coordinate system
- [x] Virtual display plane with ray-plane intersection
- [x] Adaptive smoothing (OneEuroFilter + VelocityLimiter)
- [x] Kalman filter for predictive tracking
- [x] Gesture hysteresis state machines
- [x] Multi-backend Linux input (Wayland, ydotool, X11, uinput)
- [x] System tray (AppIndicator3, SNI, Qt)
- [x] Global hotkeys (X11, Portal, evdev)
- [x] Safety system (corner, velocity, focus, inactivity)
- [x] Auto-brightness control

### 🟡 In Progress (Interaction Polish)
- [ ] Better cursor coordination & jitter reduction
- [ ] Eliminate mirrored movement issues
- [ ] Dynamic sensitivity adjustment
- [ ] Reliable pinch clicking with debouncing
- [ ] Scroll gestures
- [ ] Drag gestures
- [ ] Right-click gesture refinement

### 🟠 Planned (Spatial Control)
- [ ] Improved virtual display mapping
- [ ] Better depth estimation
- [ ] Automatic calibration wizard

### 🔵 Planned (Desktop Experience)
- [ ] Background/daemon mode
- [ ] Systemd service installation
- [ ] Minimal settings UI
- [ ] Sensitivity & gesture configuration UI
- [ ] System startup integration
- [ ] AppImage / .deb packaging

### 🚀 Long-Term
- [ ] Cross-platform input abstraction (Windows, macOS)
- [ ] Application-specific gestures
- [ ] Accessibility features
- [ ] Plugin architecture
- [ ] Advanced spatial interaction

---

## Privacy First

RED runs **entirely locally**. Your camera feed never leaves your machine.

```
📷 Camera
   │
   ▼
💻 Your Computer
   │
   ▼
🧠 Vision Processing (MediaPipe)
   │
   ▼
🖱️ Input Events
```

No cloud services. No telemetry. Your data stays yours.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores, 2.0 GHz | 4+ cores, 3.0 GHz |
| RAM | 4 GB | 8 GB |
| Camera | 720p @ 30 FPS | 1080p @ 60 FPS |
| GPU | Integrated | Discrete (for MediaPipe GPU delegate) |
| Linux Kernel | 5.10+ | 6.0+ (better uinput/Wayland support) |

Performance depends on: CPU, camera resolution/FPS, lighting, desktop environment, tracking configuration.

---

## Contributing

Contributions welcome in:
- 👁️ Computer vision & MediaPipe optimization
- 🖐️ Hand/face tracking improvements
- 🧠 Gesture recognition & hysteresis tuning
- 📐 Spatial mathematics & coordinate mapping
- 🐧 Linux input systems (Wayland, X11, uinput, ydotool)
- 🌊 Wayland protocol expertise
- ⚡ Performance optimization
- 🎨 UI/UX design
- 🧪 Testing & CI/CD
- 📚 Documentation

> For significant architectural changes, please open an issue first to discuss the approach.

---

## Bug Reports

Please include:
- **OS / Distribution:**
- **Desktop Environment:** (GNOME, KDE, Sway, Hyprland, etc.)
- **Session Type:** (Wayland / X11)
- **Python Version:**
- **Camera Model:**
- **RED Commit/Version:**

**What happened:**
**What you expected:**
**Steps to reproduce:**
**Error logs / terminal output:**

For tracking issues, a short screen recording or debug output (`python3 debug_observe_cursor.py`) helps immensely.

---

## Why RED?

Most computer interfaces assume:
```
Hand → Physical Mouse → Computer
```

RED explores:
```
Hand → Camera → Computer
```

The objective isn't to replace every mouse. It's to make **hands-free interaction practical** — for accessibility, presentations, VR/AR adjacency, or simply a different way to work.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>RED</strong><br>
  A camera. A hand. A new way to interact.
  <br><br>
  <a href="https://github.com/wrappingpine/RED">
    ⭐ Star the project on GitHub
  </a>
</p>

<p align="center">
  Made with 🖐️ and 🧠
</p>