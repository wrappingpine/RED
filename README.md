Air Mouse 🖱️

«Control your computer with your hands — no physical mouse required.»

Air Mouse is a computer-vision-based hands-free mouse that turns a regular camera into an input device. It tracks your hand in real time and translates hand movements and gestures into mouse actions.

Built with Python, OpenCV, MediaPipe, and Linux input/uinput, the project is designed to provide a lightweight and responsive alternative way to interact with a computer.

---

✨ Features

- 🖐️ Real-time hand tracking using a camera
- 🖱️ Cursor control using hand position
- 👆 Gesture-based mouse actions
- ✌️ Support for additional gesture interactions
- 🎥 Camera-based computer vision
- ⚡ Designed for continuous background operation
- 🐧 Built primarily for Linux / Pop!_OS
- 🧩 Modular vision and input architecture
- 🧪 Includes tracking, mapping, playback, and integration tests
- 📊 Includes benchmarking tools for performance evaluation

---

🧠 How It Works

Air Mouse uses a camera to observe your hand and follows a simple processing pipeline:

Camera
   │
   ▼
┌───────────────┐
│ OpenCV        │
│ Video Capture │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ MediaPipe     │
│ Hand Tracking │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Gesture /     │
│ Position      │
│ Processing    │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Coordinate    │
│ Mapping &     │
│ Smoothing     │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Linux Input   │
│ / uinput      │
└───────┬───────┘
        │
        ▼
     🖱️ Cursor

The camera captures your hand, MediaPipe detects hand landmarks, and Air Mouse converts those landmarks into cursor movement and gesture actions.

---

🛠️ Tech Stack

Technology| Purpose
Python| Core application
OpenCV| Camera and image processing
MediaPipe| Hand and landmark detection
Linux uinput| Virtual mouse/input events
Computer Vision| Hand tracking and gesture recognition

---

📋 Requirements

Hardware

- A computer running Linux
- A working webcam or USB camera
- At least one visible hand in the camera frame

Software

- Python 3
- "pip"
- Linux input/uinput support

«The project is currently developed and tested primarily with Linux / Pop!_OS.»

---

🚀 Installation

1. Clone the repository

git clone https://github.com/wrappingpine/RED.git
cd RED

2. Create a virtual environment

python3 -m venv .venv

Activate it:

source .venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Run Air Mouse

You can start the application with:

python3 airmouse/main.py

If your local project configuration provides a run script, you can alternatively use:

./run.sh

---

🖐️ Basic Usage

Once Air Mouse is running:

1. Position yourself in front of the camera.
2. Place your hand inside the camera's field of view.
3. Allow the hand tracker to detect your hand.
4. Move your hand to control the cursor.
5. Use supported gestures to perform mouse actions.

For the best tracking experience:

- Use adequate lighting.
- Keep your hand clearly visible.
- Avoid excessive motion blur.
- Keep the camera at a comfortable distance.
- Use a stable camera position.

---

📁 Project Structure

RED/
├── airmouse/
│   └── ...
│
├── face_landmarker.task
├── hand_landmarker.task
│
├── requirements.txt
├── setup.py
├── install.sh
├── run.sh
│
├── benchmark_face.py
├── benchmark_full.py
├── benchmark_image.py
├── benchmark_video.py
├── benchmark_video2.py
│
├── debug_observe_cursor.py
├── cursor_observation.jsonl
├── observation_output.txt
│
├── test_hand_face_preview.py
├── test_hand_tracker.py
├── test_hand_tracker_gui.py
├── test_head_relative.py
├── test_mapping.py
├── test_run.py
└── test_video_playback.py

«The project structure is actively evolving and may change as development continues.»

---

🧪 Testing

The repository contains several tests covering different parts of the system, including:

- Hand tracking
- Hand/face preview
- Cursor mapping
- Head-relative positioning
- Application execution
- Video playback
- GUI tracking

Run individual tests with Python:

python3 test_hand_tracker.py

or:

python3 test_mapping.py

Run the relevant test suite according to the component you're working on.

---

📊 Benchmarking

Several benchmark scripts are included to evaluate different parts of the computer-vision pipeline:

benchmark_face.py
benchmark_full.py
benchmark_image.py
benchmark_video.py
benchmark_video2.py

These can be used to investigate performance and identify bottlenecks in image processing, face/hand detection, and video processing.

---

🏗️ Architecture

Air Mouse is organized around separate responsibilities:

                 ┌─────────────┐
                 │   Camera    │
                 └──────┬──────┘
                        │
                        ▼
              ┌──────────────────┐
              │ Vision Processing│
              └────────┬─────────┘
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
      ┌─────────────┐     ┌─────────────┐
      │ Hand        │     │ Face / Head │
      │ Tracking    │     │ Tracking    │
      └──────┬──────┘     └──────┬──────┘
             │                   │
             └─────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │ Position /       │
              │ Gesture Mapping  │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Input Controller │
              └────────┬─────────┘
                       │
                       ▼
                  🖱️ Cursor

This separation makes it easier to experiment with different tracking and input strategies without rewriting the entire application.

---

🎯 Project Goals

The long-term goal of Air Mouse is to provide a lightweight, responsive, and practical hands-free computer input system that:

- Requires no specialized hardware
- Works with an ordinary camera
- Feels natural to use
- Minimizes accidental input
- Uses minimal system resources
- Can operate continuously in the background
- Remains modular and easy to extend

---

🔮 Roadmap

Planned improvements include:

- [ ] Better cursor coordination
- [ ] Improved cursor smoothing
- [ ] Reduce false clicks
- [ ] More stable hand detection
- [ ] Two-hand tracking
- [ ] Customizable gestures
- [ ] Lower CPU and RAM usage
- [ ] Improved background operation
- [ ] Better Wayland compatibility
- [ ] Configuration interface
- [ ] More mouse actions
- [ ] Improved calibration
- [ ] More robust tracking under poor lighting

---

🤝 Contributing

Contributions, ideas, bug reports, and experiments are welcome.

A typical workflow:

git clone https://github.com/wrappingpine/RED.git
cd RED

git checkout -b feature/my-feature

Make your changes, test them, and open a pull request.

When contributing, please try to:

- Keep changes focused.
- Add or update tests where appropriate.
- Document new functionality.
- Avoid introducing unnecessary dependencies.
- Test on Linux when possible.

---

⚠️ Current Status

Air Mouse is an active work-in-progress.

The core computer-vision and hand-tracking functionality is being developed toward a more stable and reliable hands-free input experience. APIs, project structure, gesture mappings, and supported environments may change over time.

---

📜 License

This project is currently under development. See the repository for the applicable license information.

---

⭐ Support the Project

If you find Air Mouse interesting or useful:

- ⭐ Star the repository
- 🐛 Report bugs
- 💡 Suggest improvements
- 🔧 Submit pull requests
- 📢 Share the project

Repository:
"wrappingpine/RED on GitHub" (https://reference-url-citation.invalid/1)

---

<p align="center">
  Made with 🖐️, 👁️ and Python
</p>
