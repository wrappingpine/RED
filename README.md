# Air Mouse 🖱️

A computer-vision-based **air mouse** that lets you control your computer cursor using hand gestures in front of a camera.

## ✨ Features

* 🖐️ Hand tracking using a camera
* 🖱️ Cursor movement using hand position
* 👆 Gesture-based clicking
* ✌️ Support for additional mouse actions
* ⚡ Designed to run continuously in the background
* 🐧 Built for Linux / Pop!_OS
* 🔧 Modular vision and input architecture

## 🛠️ Tech Stack

* **Python**
* **OpenCV**
* **MediaPipe**
* **Computer Vision**
* **Linux Input / uinput**

## 📁 Project Structure

```text
airmouse/
├── vision/
│   └── hand_tracker.py
├── input/
├── main.py
├── requirements.txt
└── README.md
```

> The project structure may change as development continues.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/wrappingpine/airmouse.git
cd airmouse
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run

```bash
python3 main.py
```

## 🎯 Goal

The goal of Air Mouse is to create a **lightweight, responsive and practical hands-free mouse** that can be used as an alternative input method without requiring additional hardware.

## 🔮 Planned Improvements

* [ ] Better cursor coordination
* [ ] Reduce false clicks
* [ ] Improve hand detection stability
* [ ] Two-hand tracking
* [ ] Gesture customization
* [ ] Lower CPU and RAM usage
* [ ] Better cursor smoothing
* [ ] Background operation
* [ ] Wayland compatibility
* [ ] Configuration interface

## 📜 License

This project is currently under development.
