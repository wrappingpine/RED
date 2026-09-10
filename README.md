# Air Mouse 🖱️

> Control your computer with your hands — no physical mouse required.

**Air Mouse** is a computer-vision-based input system that turns a webcam into a virtual mouse. It tracks your hand in real time using **MediaPipe** and **OpenCV**, interprets gestures, and sends mouse input through Linux's `uinput` interface.

Designed primarily for **Linux desktops**, Air Mouse aims to provide a lightweight, responsive and accessible alternative to traditional mouse input.

---

## ✨ Features

* 🖐️ Real-time hand tracking
* 🖱️ Control the cursor using hand movement
* 👆 Gesture-based mouse interaction
* 📷 Webcam-based — no special hardware required
* ⚡ Real-time computer vision processing
* 🖥️ PySide6 graphical interface
* 🐧 Native Linux input through `uinput`
* 🔧 Modular architecture for vision, input, UI and control
* 🧪 Built-in camera, hand-tracking and mouse diagnostics
* 📦 Automated Linux installation script
* 🚀 Simple launcher for everyday use

---

## 🧠 How It Works

Air Mouse processes the webcam feed through a simple pipeline:

```text
┌─────────────┐
│   Webcam    │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ OpenCV Capture  │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ MediaPipe Hand  │
│    Tracking     │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Gesture / Hand  │
│    Analysis     │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Mouse Controller│
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Linux / uinput  │
└─────────────────┘
```

The camera provides frames, MediaPipe detects the hand and landmarks, the application interprets those landmarks, and the resulting actions are sent to the operating system as mouse input.

---

## 🛠️ Technology Stack

| Technology       | Purpose                             |
| ---------------- | ----------------------------------- |
| **Python 3.9+**  | Application runtime                 |
| **OpenCV**       | Camera capture and image processing |
| **MediaPipe**    | Hand landmark detection             |
| **NumPy**        | Numerical and image operations      |
| **PySide6**      | Graphical user interface            |
| **Linux uinput** | Virtual mouse input                 |
| **pytest**       | Testing                             |
| **TOML**         | Configuration                       |

The project currently requires Python `>=3.9` and is packaged as `airmouse`.

---

## 🖥️ Supported Platform

Air Mouse is currently designed for:

* Linux
* Ubuntu / Debian-based distributions
* Pop!_OS
* Desktop environments with webcam and `uinput` support

The included installation script specifically installs Debian/Ubuntu system dependencies and configures the Linux `uinput` device.

> **Note:** Wayland support is an area of ongoing development. X11 environments may currently provide the most predictable experience.

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/wrappingpine/RED.git
cd RED
```

## 2. Run the installer

The repository includes an installation script that sets up the Python environment, installs dependencies, configures `uinput`, and creates a launcher.

```bash
chmod +x install.sh
./install.sh
```

The installer creates a virtual environment at:

```text
~/.airmouse-venv
```

It also configures the `input` group and a udev rule for `/dev/uinput`.

### ⚠️ Important

Run `install.sh` as your **normal user**, not as root.

After installation, you may need to **log out and log back in** for the new `input` group membership to take effect.

---

# ▶️ Running Air Mouse

After installation:

```bash
./run.sh
```

The launcher checks for the virtual environment, verifies `/dev/uinput`, checks available cameras, and starts the graphical Air Mouse application.

You can also launch the installed application directly:

```bash
airmouse
```

---

# 🔍 Diagnostics

The package provides several diagnostic commands.

### Diagnose the installation

```bash
airmouse-diagnose
```

### Test the camera

```bash
airmouse-test-camera
```

### Test hand tracking

```bash
airmouse-test-hand
```

### Test mouse input

```bash
airmouse-test-mouse
```

These commands are exposed through the project's Python package entry points.

---

# 📁 Project Structure

```text
RED/
├── airmouse/
│   ├── app/
│   ├── brightness/
│   ├── camera/
│   ├── control/
│   ├── debug/
│   ├── input/
│   ├── tests/
│   ├── ui/
│   ├── vision/
│   ├── main.py
│   └── __init__.py
│
├── face_landmarker.task
├── hand_landmarker.task
│
├── benchmark_face.py
├── benchmark_full.py
├── benchmark_image.py
├── benchmark_video.py
├── benchmark_video2.py
│
├── test_hand_tracker.py
├── test_hand_tracker_gui.py
│
├── install.sh
├── run.sh
├── requirements.txt
├── setup.py
└── README.md
```

The project is separated into dedicated areas for camera handling, computer vision, input, UI, controls, debugging and tests.

---

# 🖐️ Hand Tracking

Air Mouse uses **MediaPipe Hand Landmarker** for hand detection and landmark tracking.

The repository includes the required model assets:

```text
hand_landmarker.task
face_landmarker.task
```

The hand landmark model is approximately 7.5 MB and is included directly in the repository.

---

# ⚙️ Dependencies

Core Python dependencies include:

```text
opencv-python >= 4.8.0
numpy >= 1.24.0
mediapipe >= 0.10.0
PySide6 >= 6.5.0
tomli
tomli-w
```

Testing dependencies include:

```text
pytest
pytest-asyncio
```

These are defined in `requirements.txt` and `setup.py`.

---

# 🧪 Development

Create a development environment manually if you don't want to use the installation script:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project:

```bash
pip install -r requirements.txt
pip install -e .
```

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest
```

---

# 📊 Performance Testing

The repository contains several benchmarking scripts for evaluating computer-vision performance:

```text
benchmark_face.py
benchmark_full.py
benchmark_image.py
benchmark_video.py
benchmark_video2.py
```

For example:

```bash
python benchmark_video2.py
```

The video benchmark evaluates hand-landmark detection at several resolutions and reports processing time and FPS.

---

# 🔐 Linux `uinput`

Air Mouse uses Linux's **uinput** subsystem to create virtual input events.

The installation script:

1. Loads the `uinput` kernel module.
2. Makes the module persistent.
3. Creates an `input` group.
4. Adds the current user to the group.
5. Creates a udev rule for `/dev/uinput`.

The relevant device is:

```text
/dev/uinput
```

If the application cannot control the mouse, check:

```bash
ls -l /dev/uinput
```

and:

```bash
groups
```

If necessary, add your user to the input group:

```bash
sudo usermod -aG input $USER
```

Then log out and log back in.

---

# 🎯 Project Goals

Air Mouse is intended to become a practical hands-free input system that is:

* **Responsive**
* **Lightweight**
* **Customizable**
* **Accessible**
* **Hardware-independent**
* **Easy to install**
* **Easy to extend**

The architecture is intentionally modular so that vision, gesture recognition, input handling and the UI can evolve independently.

---

# 🗺️ Roadmap

Planned areas of improvement include:

* [ ] More accurate cursor positioning
* [ ] Better cursor smoothing
* [ ] Reduce accidental clicks
* [ ] Improve hand-tracking stability
* [ ] Two-hand tracking
* [ ] Custom gesture configuration
* [ ] Better gesture recognition
* [ ] Lower CPU and RAM usage
* [ ] Improved background operation
* [ ] Improved Wayland compatibility
* [ ] Configuration interface
* [ ] More accessibility-oriented controls
* [ ] Additional mouse actions
* [ ] Expanded automated testing

---

# 🐛 Troubleshooting

### Camera is not detected

Check available video devices:

```bash
ls /dev/video*
```

If `v4l-utils` is installed:

```bash
v4l2-ctl --list-devices
```

The included launcher performs a camera check before starting the application.

### `uinput` is missing

Try:

```bash
sudo modprobe uinput
```

Then verify:

```bash
ls -l /dev/uinput
```

### Permission denied for `/dev/uinput`

Make sure your user belongs to the `input` group:

```bash
sudo usermod -aG input $USER
```

Then log out and back in.

### Application does not start

Reinstall the Python environment:

```bash
./install.sh
```

Then:

```bash
./run.sh
```

---

# 🤝 Contributing

Contributions are welcome.

If you'd like to improve Air Mouse:

1. Fork the repository.
2. Create a feature branch.

```bash
git checkout -b feature/my-feature
```

3. Make your changes.
4. Run the tests.

```bash
pytest
```

5. Commit your changes.

```bash
git commit -m "Add my feature"
```

6. Push your branch.

```bash
git push origin feature/my-feature
```

7. Open a pull request.

For larger changes, opening an issue first can help keep development coordinated.

---

# 📄 License

Air Mouse is currently under active development.

The Python package metadata declares the project as **MIT licensed**. Before publishing a release, make sure the repository contains the corresponding `LICENSE` file and that all bundled model assets and third-party components are used in accordance with their respective licenses.

---

# ⭐ Acknowledgements

This project builds on several excellent open-source technologies:

* [OpenCV](https://opencv.org/)
* [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide)
* [NumPy](https://numpy.org/)
* [PySide6](https://doc.qt.io/qtforpython/)
* Linux `uinput`

---

## 🔗 Repository

**Air Mouse — wrappingpine/RED**

https://github.com/wrappingpine/RED

---

<p align="center">
  Built with Python, computer vision, and a webcam. 🖐️
</p>
