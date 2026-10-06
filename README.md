# 🖐️ AirMouse

### Control your Linux desktop with your hand.

**AirMouse** is an open-source, camera-based virtual mouse for Linux that lets you control the desktop using natural hand movements and gestures.

It uses computer vision to track your hand, estimate the palm position, translate that movement into cursor movement, and recognize gestures for actions such as clicking, dragging, and scrolling.

> 🚧 **Status: Active Development**
>
> AirMouse is currently focused on Linux, with Pop!_OS/COSMIC + Wayland as an important development target.

---

## ✨ Features

### 🖐️ Hand-Based Cursor

Control the cursor using the position of your palm instead of a physical mouse.

* Palm-based 2D tracking
* Left and right hand support
* Stable hand selection
* Hand-loss recovery
* Smooth cursor movement
* Adaptive workspace
* Full-screen control
* Configurable cursor sensitivity
* ~20% increased cursor responsiveness target

### 🎯 Smart Cursor Behavior

AirMouse is designed around natural hand movement:

| Movement           | Behavior                   |
| ------------------ | -------------------------- |
| 🐢 Slow movement   | Precision                  |
| ⚡ Fast movement    | Faster cursor response     |
| ✋ Stationary hand  | Stabilization              |
| 🖥️ Workspace edge | Adaptive boundary behavior |

The goal is not simply to make the cursor follow the camera.

The goal is to make it **feel like a mouse**.

---

## 👆 Gesture Control

AirMouse separates cursor movement from gesture recognition.

This allows hand movement to remain responsive even when a gesture is being detected.

Planned/supported gesture actions include:

* 🖱️ Left click
* 🖱️ Right click
* 🖱️ Middle click
* ✋ Drag
* ↕️ Scroll
* 🤏 Additional configurable gestures

Gesture recognition includes debounce and confidence checks to reduce accidental actions.

---

## 🧠 Computer Vision

AirMouse uses hand landmarks to understand hand position and gestures.

### Current architecture

```text
┌──────────────┐
│    Camera    │
└──────┬───────┘
       ↓
┌────────────────────┐
│ Camera Processing  │
└────────┬───────────┘
         ↓
┌────────────────────┐
│ Hand Landmarking   │
└────────┬───────────┘
         ↓
    ┌────┴─────┐
    ↓          ↓
┌─────────┐ ┌──────────────┐
│  Palm   │ │ Hand         │
│ Center  │ │ Landmarks    │
└────┬────┘ └──────┬───────┘
     ↓              ↓
┌─────────────┐ ┌─────────────┐
│   Cursor    │ │   Gesture   │
│   Pipeline  │ │   Pipeline  │
└──────┬──────┘ └──────┬──────┘
       ↓               ↓
       └───────┬───────┘
               ↓
       ┌──────────────┐
       │ Linux Input  │
       └──────────────┘
```

### Why this architecture?

The cursor and gesture systems are intentionally separated.

A gesture detection delay should **not freeze cursor movement**, and temporary cursor tracking loss should **not accidentally trigger a click**.

---

## 🚀 Performance

AirMouse is designed to run on relatively low-power hardware.

The project prioritizes:

* Low latency
* Low CPU usage
* Efficient camera processing
* Lightweight hand tracking
* Minimal frame copying
* Adaptive smoothing
* Responsive input

Target performance:

```text
Minimum target:   ~20 FPS
Preferred target: 25–30 FPS
```

Performance diagnostics can measure:

```text
FPS
Camera processing
Hand inference
Cursor processing
Gesture processing
Input delivery
Total latency
CPU usage
RAM usage
```

---

# 🖥️ Linux Support

AirMouse is currently **Linux-first**.

Primary development environment:

* Linux
* Pop!_OS
* COSMIC
* Wayland

Other Linux distributions are intended to be supported where their camera and input systems are compatible.

### Current platform direction

```text
Linux       ██████████  Primary
Windows     ░░░░░░░░░░  Future
macOS       ░░░░░░░░░░  Future
```

Cross-platform support is intentionally not the current priority.

---

# 📦 Installation

## Requirements

Recommended:

* Linux
* Python 3
* Webcam
* MediaPipe-compatible environment
* Camera access permissions
* Linux mouse-input backend

Optional/recommended for the current Wayland implementation:

```bash
ydotool
```

---

## Clone the repository

```bash
git clone https://github.com/wrappingpine/RED.git
cd RED
```

---

## Install dependencies

Follow the dependency configuration included in the repository.

If a virtual environment is used:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then install the project dependencies:

```bash
pip install -r requirements.txt
```

> Dependency installation may change while AirMouse is under active development. Check the repository's current configuration before installing manually.

---

# ▶️ Running AirMouse

If the `airmouse` command is installed:

```bash
airmouse
```

Alternatively, run the project's main entry point according to the current repository structure.

---

# ⚙️ Configuration

AirMouse is designed to expose important behavior through configuration rather than requiring source-code modifications.

Important parameters include:

```text
cursor_speed_multiplier = 1.20

workspace_x_min
workspace_x_max

workspace_y_min
workspace_y_max

hand_detection_confidence
hand_presence_confidence
hand_tracking_confidence

smoothing
adaptive_smoothing

hand_switch_delay
hand_loss_grace_period

gesture_debounce
click_confidence
```

The default configuration should work without manual tuning.

Advanced users can tune the workspace and responsiveness for their camera and screen.

---

# 🖱️ Cursor Model

AirMouse does **not** use a traditional virtual mouse image or a physical mouse representation.

Instead, it maps the user's natural palm movement into a configurable 2D workspace.

```text
Camera Workspace
┌─────────────────────────────┐
│                             │
│       ✋ Palm                │
│                             │
│                             │
│                             │
└─────────────────────────────┘
              ↓
        Screen Mapping
              ↓
┌─────────────────────────────┐
│                             │
│                         🖱️  │
│                             │
│                             │
└─────────────────────────────┘
```

The workspace is designed so that the user does not need to move their hand to extreme camera edges to reach the corners of the screen.

---

# 🧩 Architecture Principles

AirMouse follows several important design principles.

### 1. Low latency over unnecessary complexity

Every processing stage should justify its CPU and latency cost.

### 2. Tracking should be independent from gestures

Cursor movement must remain responsive while gestures are being evaluated.

### 3. Smoothness without lag

Filtering should remove jitter without introducing noticeable input delay.

### 4. Fail safely

Temporary tracking loss should hold the cursor rather than generate unpredictable input.

### 5. No accidental actions

Clicks require confidence, stability and proper gesture-state transitions.

### 6. Linux-first

The current implementation should be optimized for Linux desktop environments before expanding to other platforms.

---

# 🔒 Privacy

AirMouse is designed as a **local computer-vision application**.

Camera frames are processed locally by the application.

There is no requirement for:

* Cloud vision APIs
* Remote AI services
* Online image processing
* Voice services

The project aims to keep camera processing local.

---

# 🛠️ Development

The project is actively evolving.

When contributing, prefer focused changes rather than large rewrites.

Before modifying a subsystem, understand its role in:

```text
Camera
  ↓
Hand Tracking
  ↓
Palm / Landmark Processing
  ↓
Cursor / Gesture Pipelines
  ↓
Input Backend
```

---

# 🧪 Testing

Testing should cover both computer-vision behavior and real desktop interaction.

### Hand tracking

* [ ] Left hand
* [ ] Right hand
* [ ] Both hands
* [ ] Different distances
* [ ] Different orientations
* [ ] Temporary occlusion
* [ ] Hand loss
* [ ] Hand reacquisition
* [ ] Hand switching

### Cursor

* [ ] Slow movement
* [ ] Fast movement
* [ ] Precision movement
* [ ] Stationary stabilization
* [ ] Horizontal movement
* [ ] Vertical movement
* [ ] Diagonal movement
* [ ] Screen corners
* [ ] Screen boundaries

### Gestures

* [ ] Left click
* [ ] Right click
* [ ] Middle click
* [ ] Drag
* [ ] Scroll
* [ ] False-click rejection

### Performance

* [ ] FPS measurement
* [ ] CPU measurement
* [ ] RAM measurement
* [ ] Input latency measurement

---

# 🗺️ Roadmap

## Phase 1 — Core Tracking

* [x] Camera input
* [x] Hand landmark tracking
* [x] Linux cursor control
* [x] Gesture framework
* [ ] Improve hand recognition
* [ ] Robust palm tracking
* [ ] Stable hand switching

## Phase 2 — Cursor Quality

* [ ] Adaptive workspace calibration
* [ ] Low-latency smoothing
* [ ] Velocity-aware cursor control
* [ ] Boundary assistance
* [ ] 20% responsiveness improvement
* [ ] Better precision mode

## Phase 3 — Gesture System

* [ ] Reliable click state machine
* [ ] Improved drag
* [ ] Scroll gestures
* [ ] Custom gesture configuration
* [ ] Gesture profiles

## Phase 4 — Application Profiles

* [ ] Per-application sensitivity
* [ ] Per-application gestures
* [ ] Automatic application detection
* [ ] Universal Linux application support

## Phase 5 — Advanced Features

* [ ] Adaptive user calibration
* [ ] Personal tracking profiles
* [ ] Advanced gesture customization
* [ ] Better Wayland integration
* [ ] Additional Linux desktop environments

## Future

* [ ] Windows support
* [ ] macOS support
* [ ] Cross-platform input abstraction
* [ ] Production-ready packaging
* [ ] Optional commercial edition

---

# 🐛 Known Limitations

AirMouse is still under active development.

Current areas receiving significant work:

* Hand recognition quality
* Tracking stability
* Cursor latency
* Gesture reliability
* Wayland input integration
* Automatic calibration
* Multi-hand behavior

Hardware, lighting, camera quality and desktop environment can affect tracking performance.

---

# 🤝 Contributing

Contributions are welcome.

Useful contributions include:

* Hand-tracking improvements
* Performance optimization
* Gesture recognition
* Linux/Wayland compatibility
* Testing on different hardware
* Bug reports
* Documentation
* UX improvements

### Before submitting a change

Please test:

1. Cursor movement
2. Hand recognition
3. Existing gestures
4. CPU usage
5. Input latency
6. Temporary tracking loss

Avoid introducing unnecessary dependencies or cloud services.

---

# 📄 License

See the repository's `LICENSE` file for the current license.

---

# ⭐ Project

**AirMouse**

> Turn your hand into a mouse.

GitHub:

**https://github.com/wrappingpine/RED**

Built for Linux.
Powered by computer vision.
Designed for natural interaction.
