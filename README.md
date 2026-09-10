🖐️ RED — Air Mouse

<p align="center">
  <strong>Turn your camera into a mouse.</strong><br>
  Control your Linux desktop with your hands — no physical mouse required.
</p><p align="center">
  <a href="https://github.com/wrappingpine/RED">
    <img src="https://img.shields.io/badge/GitHub-RED-black?style=for-the-badge&logo=github" alt="GitHub">
  </a>
  <img src="https://img.shields.io/badge/Linux-Supported-success?style=for-the-badge&logo=linux" alt="Linux">
  <img src="https://img.shields.io/badge/OpenCV-Vision-blue?style=for-the-badge&logo=opencv" alt="OpenCV">
  <img src="https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange?style=for-the-badge" alt="MediaPipe">
  <img src="https://img.shields.io/badge/Status-Active%20Development-yellow?style=for-the-badge" alt="Status">
</p>---

⚡ What is RED?

RED is a camera-based, hands-free mouse for Linux.

It uses computer vision to track your hand and translate your movements and gestures into desktop input.

        🖐️
        │
        │  Move / Gesture
        ▼
   📷 Webcam
        │
        ▼
  🧠 Computer Vision
        │
        ▼
 🎯 Spatial Mapping
        │
        ▼
 🖱️ Input Controller
        │
        ▼
   🖥️ Linux Desktop

«No special hardware. No wearable controller. Just a camera and your hand.»

---

🎬 Demo

«🚧 Demo video coming soon»

The goal is simple:

Point  →  Move cursor
Pinch  →  Click
Gesture → Action

RED is being developed toward a natural interaction model where the cursor follows your hand smoothly, accurately, and predictably.

---

✨ Features

<table>
<tr>
<td width="50%">🖐️ Hand Tracking

Real-time hand landmark detection using MediaPipe.

🖱️ Cursor Control

Move the desktop cursor using your hand.

🤏 Gesture Input

Use configurable hand gestures for mouse actions.

👥 Multi-Hand Support

Architecture designed for reliable two-hand interaction.

</td><td width="50%">🎯 Spatial Mapping

Converts camera-space movement into screen coordinates.

🧠 Head-Relative Tracking

Experimental spatial tracking using face/head landmarks.

⚡ Performance Focused

Designed with low-end hardware and low latency in mind.

🐧 Linux First

Built around Linux desktop and input systems.

</td>
</tr>
</table>---

🧠 How RED Works

RED is more than simply mapping your fingertip to the screen.

The processing pipeline is divided into several stages:

┌───────────────────────┐
│       📷 CAMERA       │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│      OpenCV           │
│    Frame Capture      │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│     MediaPipe         │
│   Hand + Face Vision  │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│   Spatial Mapping     │
│  Camera → Screen      │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ Gesture / State Logic │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│    Linux Input        │
└───────────┬───────────┘
            │
            ▼
       🖥️ DESKTOP

---

🎮 Interaction Model

RED is designed around simple, human-readable interactions.

Action| Input
Move cursor| ☝️ Index finger
Click| 🤏 Pinch
Scroll| 🚧 In development
Drag| 🚧 In development
Right click| 🚧 In development
Custom gestures| 🚧 Planned

The gesture system is intentionally being developed to minimize false positives.

---

🎯 The Hard Part: Making It Feel Natural

A webcam can detect your hand.

That's easy.

Making it feel like a real mouse is much harder.

RED focuses heavily on:

        Raw Tracking
             │
             ▼
       Noise Filtering
             │
             ▼
       Motion Smoothing
             │
             ▼
     Coordinate Mapping
             │
             ▼
      Gesture Detection
             │
             ▼
        Input Event

Current engineering priorities

- 🎯 Accurate cursor direction
- 🪶 Smooth movement
- ⚡ Low latency
- 🧹 Jitter reduction
- 🛑 False-click prevention
- 👥 Reliable two-hand detection
- 💡 Lighting robustness
- 🧮 Better coordinate mapping

---

🔬 Experimental Spatial Tracking

RED also explores a more advanced approach than traditional webcam mouse implementations.

Instead of thinking only in terms of:

Camera Pixel → Screen Pixel

the system can reason about:

          Head
           👤
           │
           │
           ▼
      Virtual Plane
    ┌──────────────┐
    │              │
    │   🖐️        │
    │              │
    └──────────────┘
           │
           ▼
       Screen XY

This creates the possibility of a more consistent 3D-aware pointing model.

---

🛠️ Technology

Layer| Technology
Programming| Python
Video| OpenCV
Hand Tracking| MediaPipe
Face Tracking| MediaPipe
Input| Linux input / "uinput"
Platform| Linux
Primary Environment| Pop!_OS

---

🚀 Quick Start

1. Clone

git clone https://github.com/wrappingpine/RED.git
cd RED

2. Create environment

python3 -m venv .venv
source .venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Run RED

./run.sh

If the launcher isn't executable:

chmod +x run.sh
./run.sh

---

🧪 Developer Mode

RED contains multiple testing and benchmarking utilities.

Hand tracking

python test_hand_tracker.py

Coordinate mapping

python test_mapping.py

Head-relative tracking

python test_head_relative.py

General runtime test

python test_run.py

Performance benchmarks

benchmark_face.py
benchmark_full.py
benchmark_image.py
benchmark_video.py
benchmark_video2.py

These tools allow tracking and performance changes to be tested independently.

---

📂 Project Structure

RED/
│
├── airmouse/                 # Core application
│
├── hand_landmarker.task      # Hand tracking model
├── face_landmarker.task      # Face tracking model
│
├── benchmark_*.py            # Performance benchmarks
│
├── test_*.py                 # Test suite
│
├── debug_observe_cursor.py   # Cursor debugging
│
├── cursor_observation.jsonl  # Cursor observations
├── observation_output.txt    # Debug output
│
├── ARCHITECTURE_AUDIT.md     # Architecture notes
│
├── requirements.txt          # Python dependencies
├── setup.py                  # Package configuration
├── install.sh                # Installation helper
└── run.sh                    # Launcher

---

🧩 Architecture Philosophy

RED separates vision, interpretation, and input.

┌────────────────────────────────────┐
│              RED                   │
│                                    │
│  Camera                            │
│    ↓                               │
│  Vision                            │
│    ↓                               │
│  Tracking                          │
│    ↓                               │
│  Spatial Mapping                   │
│    ↓                               │
│  Gesture Engine                    │
│    ↓                               │
│  Input Backend                     │
│                                    │
└────────────────────────────────────┘

This makes it possible to improve one subsystem without rewriting everything else.

For example:

Better hand model
       ↓
Same gesture engine
       ↓
Same input backend

---

🐧 Linux & Wayland

RED is being developed Linux-first, with modern desktop environments in mind.

Particular attention is required for:

- Wayland
- pointer injection
- "/dev/uinput"
- permissions
- compositor behavior
- application focus
- input security

The vision pipeline is kept separate from the OS input layer so that different input backends can be explored without rebuilding the tracking system.

---

📊 Development Status

Computer Vision       █████████░  90%
Hand Tracking         █████████░  90%
Cursor Mapping        ████████░░  80%
Gesture Engine        ███████░░░  70%
Two-Hand Tracking     ██████░░░░  60%
Wayland Integration   ██████░░░░  60%
GUI                   ████░░░░░░  40%
Calibration            ████░░░░░░  40%
Production Polish     ███░░░░░░░  30%

«These are development targets/estimates, not formal release guarantees.»

---

🗺️ Roadmap

🟢 Core Tracking

- [x] Webcam input
- [x] Hand landmark detection
- [x] MediaPipe integration
- [x] Basic cursor mapping
- [x] Tracking tests
- [x] Benchmark tooling

🟡 Interaction

- [ ] Better cursor coordination
- [ ] Eliminate mirrored movement
- [ ] Better smoothing
- [ ] Dynamic sensitivity
- [ ] Reliable pinch clicking
- [ ] Gesture debouncing
- [ ] Scroll gestures
- [ ] Drag gestures
- [ ] Right click
- [ ] Custom gestures

🟠 Spatial Control

- [x] Face tracking experiments
- [x] Head-relative tracking experiments
- [ ] Improved virtual display mapping
- [ ] Better depth estimation
- [ ] Automatic calibration

🔵 Desktop Experience

- [ ] Background mode
- [ ] Global activation shortcut
- [ ] Minimal settings UI
- [ ] Sensitivity controls
- [ ] Gesture configuration
- [ ] System startup integration
- [ ] Better Wayland support

🚀 Long-Term

- [ ] Cross-platform input abstraction
- [ ] Windows support
- [ ] macOS support
- [ ] Application-specific gestures
- [ ] Accessibility features
- [ ] Plugin architecture
- [ ] Advanced spatial interaction

---

🔐 Privacy First

RED's core computer-vision pipeline is designed to run locally.

Your camera feed does not need to be uploaded to a cloud computer-vision service for hand tracking.

📷 Camera
   │
   ▼
💻 Your Computer
   │
   ▼
🧠 Vision Processing
   │
   ▼
🖱️ Input

Your camera. Your machine. Your data.

---

💻 Hardware

RED is intentionally designed to work with ordinary hardware.

Minimum concept

💻 Computer
+
📷 Webcam

No specialized motion controller is required.

Performance will depend on:

- CPU
- camera resolution
- camera FPS
- lighting
- number of tracking models
- desktop environment
- tracking configuration

---

🤝 Contributing

RED is an evolving open-source project.

Contributions are especially useful in:

- 👁️ Computer vision
- 🖐️ Hand tracking
- 🧠 Gesture recognition
- 📐 Spatial mathematics
- 🐧 Linux input
- 🌊 Wayland
- ⚡ Performance optimization
- 🎨 UI/UX
- 🧪 Testing
- 📚 Documentation

If you're planning a significant architectural change, open an issue first so the approach can be discussed.

---

🐛 Found a Bug?

Please include:

OS:
Desktop Environment:
Python:
Camera:
RED commit/version:

What happened:

What you expected:

Steps to reproduce:

Error / logs:

For tracking problems, a short screen recording or debug output can make diagnosis much easier.

---

🌟 Why RED?

Most computer interfaces assume:

Hand → Physical Mouse → Computer

RED explores:

Hand → Camera → Computer

The objective isn't to replace every mouse.

It's to make hands-free interaction practical.

---

🖐️ Point

🤏 Gesture

🖥️ Control

---

<p align="center">RED

A camera. A hand. A new way to interact.

<br><a href="https://github.com/wrappingpine/RED">
  ⭐ Star the project on GitHub
</a></p>---

<p align="center">
  Made with 🖐️ and 🧠
</p>
