# AirMouse — Ultimate Product Specification & Development Constitution

> **Status:** Master Specification
> **Project:** AirMouse
> **Platform:** Linux-first, all major Linux environments
> **Long-term:** Cross-platform commercial application
> **Architecture:** Local-first, background-first, modular, extensible
> **Primary interaction:** Camera-based hand tracking → intelligent 2D computer input
> **Voice:** Not part of AirMouse core
> **AI:** Optional intelligence layer, never allowed to destabilize the real-time input loop

---

# 1. The Ultimate Goal

AirMouse is intended to become a **production-grade human-computer interaction platform** that allows a person to control a computer naturally using their hands.

It must not become merely:

* a webcam mouse
* a collection of unreliable gestures
* a flashy HUD
* an AI chatbot controlling a cursor
* a collection of application-specific hacks
* a CPU-heavy computer-vision demo

The ultimate goal is:

> **Make the computer understand the user's hands naturally, precisely, smoothly, privately, reliably, and comfortably.**

AirMouse should eventually feel less like operating a camera system and more like using a new input device.

The user should be able to:

* move the pointer naturally
* click
* double-click
* drag
* scroll
* switch interaction modes
* use customizable gestures
* control applications
* trigger keyboard shortcuts
* control media
* interact with arbitrary Linux applications
* create application profiles
* adapt the system to their personal movement style

without needing to think about the underlying computer vision.

---

# 2. Product Philosophy

Every feature must be evaluated against these principles:

1. **Smooth**
2. **Fast**
3. **Lightweight**
4. **Smart**
5. **Comfortable**
6. **Reliable**
7. **Predictable**
8. **Private**
9. **Local-first**
10. **Extensible**
11. **Application-agnostic**
12. **Hardware-efficient**
13. **Safe**
14. **Measurable**

A feature is not successful simply because it works in a demonstration.

It must work reliably in real-world conditions.

---

# 3. Development Constitution

## 3.1 Never blindly implement an idea

AirMouse development agents must NOT simply implement every requested feature.

Before implementing a significant feature, the development process must determine:

### Problem

What actual user problem is being solved?

### Evidence

Is there evidence that the problem exists?

### Research

What existing approaches already solve it?

### Alternatives

Are there simpler or more reliable solutions?

### Cost

What are the CPU, RAM, latency, complexity, maintenance, and UX costs?

### Risk

Could the feature create:

* false clicks
* accidental actions
* tracking instability
* latency
* discomfort
* security problems
* application incompatibilities
* excessive resource consumption?

### Prototype

Can the idea be tested independently before integrating it into the core?

### Benchmark

Does it measurably improve the system?

### Real-world test

Does it improve actual interaction rather than only synthetic tests?

### Decision

Only then:

* implement
* modify
* postpone
* isolate as experimental
* or reject

---

# 4. Research-First Development

Before substantial architectural or algorithmic changes, AirMouse development should research current information.

Research should prioritize:

1. Official Linux documentation
2. Wayland/freedesktop specifications
3. Kernel documentation
4. Open-source reference implementations
5. Computer-vision framework documentation
6. Academic HCI research
7. Relevant GitHub repositories
8. Issue trackers
9. Real-world user reports
10. Benchmarks

Do not reinvent existing infrastructure without a measurable reason.

Do not assume an old Linux workaround is still the correct architecture.

Do not blindly copy GitHub implementations.

Every major technical decision should record:

```text
Problem
Research
Options
Trade-offs
Decision
Why
Benchmark
Rollback plan
```

---

# 5. Core Product Definition

AirMouse ultimately has four layers:

```text
HUMAN
  ↓
HAND / BODY MOVEMENT
  ↓
COMPUTER VISION
  ↓
TRACKING
  ↓
MOTION INTERPRETATION
  ↓
INTENT / GESTURE ENGINE
  ↓
INPUT ABSTRACTION
  ↓
LINUX DESKTOP
  ↓
APPLICATIONS
```

The computer should not need to know that the input originated from a webcam.

AirMouse should behave like an input device.

---

# 6. AirMouse Is Primarily a Smart Input Platform

The first intelligence layer is not an LLM.

The core intelligence should come from:

* motion analysis
* hand tracking
* confidence estimation
* velocity estimation
* acceleration estimation
* temporal stability
* gesture state machines
* intent recognition
* context awareness
* application profiles
* adaptive calibration
* user-specific adaptation

Machine-learning or LLM-based intelligence may be added later where it provides measurable value.

The low-latency pointer loop should remain deterministic wherever practical.

---

# 7. No Voice in AirMouse Core

Voice is explicitly **not part of the AirMouse core architecture**.

AirMouse should not depend on:

* speech recognition
* TTS
* cloud AI
* voice assistants
* microphones

for its fundamental operation.

A future external system may combine AirMouse with voice or an AI agent, but that must remain an optional integration.

---

# 8. Background-First Architecture

AirMouse should normally run in the background.

The user should not need to keep a large application window open.

Normal architecture:

```text
AirMouse Service
       │
       ├── Camera
       ├── Vision
       ├── Tracking
       ├── Motion
       ├── Gesture
       ├── Input
       └── Diagnostics
```

Optional UI:

```text
AirMouse Settings
AirMouse Calibration
AirMouse Diagnostics
AirMouse Profiles
```

The UI should appear only when explicitly requested.

Examples:

* configurable keyboard shortcut
* system tray/menu entry where supported
* dedicated activation mechanism

The core tracking/input service must continue independently of the settings UI.

---

# 9. Status Indicator

AirMouse should provide a **small, restrained status indicator**.

It must communicate states such as:

```text
OFF
STARTING
CAMERA ERROR
SEARCHING
TRACKING
PAUSED
GESTURE MODE
CALIBRATION
ERROR
```

The indicator must not become a permanent distracting HUD.

Visual identity:

* professional
* minimal
* futuristic but restrained
* dark interface
* subtle cyan/blue accent
* high readability
* no excessive glow
* no unnecessary animation

---

# 10. Platform Strategy

## Phase 1

Support Linux broadly.

Do not design AirMouse specifically around only:

* Pop!_OS
* Ubuntu
* COSMIC

The architecture must support different Linux environments.

Primary target:

```text
Linux
├── Wayland
├── X11 compatibility
├── GNOME
├── KDE
├── COSMIC
└── other major environments
```

The implementation must detect capabilities instead of assuming one desktop environment.

---

# 11. Linux Input Architecture

Input injection must be abstracted.

Conceptually:

```text
AirMouse Input API
       │
       ├── Wayland backend
       ├── Linux virtual-input backend
       ├── X11 backend
       └── Future platform backends
```

Linux virtual input should investigate `uinput`/`libevdev` rather than making `ydotool` the architectural foundation. Linux documents `uinput` as a userspace mechanism for creating virtual input devices and specifically notes `libevdev` as a less error-prone option for new software.

`ydotool` may remain a compatibility/testing mechanism if useful, but the AirMouse architecture must not depend on a single command-line workaround.

---

# 12. Input Abstraction

AirMouse should expose an internal interface similar to:

```text
move_pointer(x, y)
click(button)
double_click(button)
press(button)
release(button)
scroll(x, y)
key_down(key)
key_up(key)
shortcut(keys)
drag_start()
drag_end()
```

The gesture system must never directly manipulate OS-specific APIs.

Instead:

```text
Gesture
   ↓
Intent
   ↓
AirMouse Input API
   ↓
Platform Backend
```

This is critical for future cross-platform support.

---

# 13. 2D First

The first production architecture should be **2D interaction**.

Do not prematurely build full 3D spatial interaction.

Primary model:

```text
Camera
 ↓
Hand landmarks
 ↓
2D normalized coordinates
 ↓
screen mapping
 ↓
cursor
```

Future 3D/depth interaction can be added later as a separate subsystem.

Potential future technologies:

* depth cameras
* stereo cameras
* monocular depth estimation
* spatial interaction planes
* 3D gestures

But none of these should complicate the first reliable product.

---

# 14. No Multiple-Camera Requirement

AirMouse should initially assume:

```text
1 camera
```

Multiple cameras are not a product requirement.

The architecture should remain extensible enough that future multi-camera support does not require rewriting the entire system.

---

# 15. Hand Tracking

The vision layer must support:

* one-hand tracking
* two-hand tracking
* handedness
* landmark confidence
* tracking confidence
* tracking-loss detection
* temporal continuity
* hand identity persistence

The tracking layer must be independent of the gesture layer.

Conceptually:

```text
Camera Frame
    ↓
Hand Detector
    ↓
Landmarks
    ↓
Tracking State
    ↓
Gesture Engine
```

Do not mix gesture decisions into the raw vision implementation.

---

# 16. Tracking Confidence

Every tracked hand should have a confidence state.

Conceptually:

```text
HIGH
MEDIUM
LOW
LOST
```

The system must never treat a low-confidence landmark as equivalent to a stable landmark.

Confidence should influence:

* cursor movement
* clicking
* dragging
* gesture activation
* scrolling
* mode changes

---

# 17. Tracking Loss Safety

The required behavior when the controlling hand disappears is:

> **Freeze pointer movement and generate no input.**

This means:

```text
Hand lost
   ↓
Stop pointer updates
   ↓
Stop gesture recognition
   ↓
Do NOT generate clicks
   ↓
Do NOT generate keyboard events
   ↓
Wait for stable reacquisition
```

No random movement.

No automatic cursor repositioning.

No action generated from stale landmarks.

No click generated during reacquisition.

---

# 18. Reacquisition

When the hand returns:

1. Detect the hand.
2. Verify confidence.
3. Stabilize tracking.
4. Establish a new motion baseline.
5. Resume control.

The cursor must not jump because the hand reappeared at a different camera position.

---

# 19. Cursor Philosophy

The pointer must feel:

> **stable when I stop, responsive when I move, fast when I move quickly, and controllable when I need precision.**

This is one of the most important product requirements.

---

# 20. Adaptive Cursor Motion

AirMouse should NOT use one fixed sensitivity value for all movement.

The motion engine should adapt according to movement state.

### Slow movement

→ precision mode

### Fast movement

→ acceleration

### Stationary hand

→ stabilization

### Application context

→ application profile

Conceptually:

```text
Hand Motion
     ↓
Velocity Estimation
     ↓
Movement Classification
     │
     ├── Stationary → Stabilize
     ├── Slow → Precision
     ├── Medium → Normal
     └── Fast → Accelerate
```

---

# 21. Motion Filtering

The motion pipeline should investigate and benchmark:

* EMA
* One Euro filtering
* Kalman filtering
* adaptive low-pass filtering
* velocity-dependent filtering
* dead zones
* hysteresis
* prediction
* interpolation
* outlier rejection

Do not assume one algorithm is universally optimal.

A likely architecture is an adaptive filter whose behavior changes according to movement speed.

The 1€ filter is specifically designed as a speed-based low-pass filter for noisy interactive input, making this class of approach worth benchmarking rather than using arbitrary fixed smoothing.

---

# 22. Important Rule: Never Over-Smooth

Smoothing must not become lag.

Bad:

```text
Smooth = more smoothing
```

Desired:

```text
Low speed
→ strong stabilization
→ precision

High speed
→ lower smoothing
→ responsiveness
```

The system must optimize:

```text
jitter ↓
latency ↓
overshoot ↓
control ↑
```

rather than maximizing smoothness alone.

---

# 23. Cursor Vibration

Cursor vibration is a first-class engineering problem.

Measure it.

Possible metrics:

* stationary RMS displacement
* 95th percentile displacement
* micro-movement frequency
* cursor travel during stationary hand
* false gesture activation rate

A fix is not considered successful because the cursor "looks smoother."

It must reduce measurable unwanted motion without introducing unacceptable latency.

---

# 24. Cursor Speed

The system must avoid excessive cursor speed.

Sensitivity should be:

* configurable
* adaptive
* application-aware
* calibrated to screen size
* influenced by hand velocity

The user should be able to tune:

```text
Sensitivity
Acceleration
Precision
Dead zone
Stabilization
Maximum velocity
```

---

# 25. Screen Mapping

The system must support:

* different resolutions
* different aspect ratios
* scaling
* fractional scaling
* multiple monitors
* monitor selection
* monitor arrangement
* coordinate transformations

The core vision system must not assume:

```text
1920 × 1080
```

or any single display geometry.

---

# 26. Calibration

AirMouse must include a calibration system.

Calibration should establish:

* camera framing
* usable hand region
* screen mapping
* sensitivity
* preferred control area
* handedness
* baseline position
* movement scale

Calibration must be quick enough that users will actually use it.

---

# 27. Adaptive Personalization

AirMouse should learn from the user's interaction over time.

However:

> **Adaptation must never silently destroy predictability.**

The system may learn:

* preferred sensitivity
* preferred acceleration
* typical movement speed
* stabilization preference
* gesture thresholds
* gesture timing
* preferred application profiles

Adaptation should have:

* bounds
* confidence
* rollback
* reset
* transparency

---

# 28. Deterministic Core + Adaptive Layer

Use:

```text
Deterministic Core
        +
Adaptive Personalization
```

not:

```text
Black-box learning everywhere
```

The deterministic layer guarantees predictable behavior.

The adaptive layer improves comfort.

---

# 29. Gesture Strategy

Do NOT start with 30–50 gestures.

The recommended initial vocabulary is:

> **8 core gestures**

with an extensible gesture engine.

The first gestures should prioritize reliability over novelty.

---

# 30. Proposed Core Gesture Set

Initial candidate vocabulary:

### 1. Point

Index finger extended.

Purpose:

```text
Pointer control
```

### 2. Pinch

Thumb + index finger.

Purpose:

```text
Left click
```

### 3. Pinch Hold

Thumb + index maintained.

Purpose:

```text
Drag
```

### 4. Two-Finger Scroll

Index + middle finger.

Purpose:

```text
Scroll
```

### 5. Open Palm

Purpose:

```text
Pause / mode control
```

### 6. Fist

Purpose:

```text
Secondary interaction / mode
```

### 7. Thumb Gesture

Purpose:

```text
Configurable action
```

### 8. Two-Hand Gesture

Purpose:

```text
Mode switching / future advanced control
```

These are candidates, not sacred definitions.

Each must be validated experimentally.

---

# 31. Gesture Engine

Gestures must use a state machine.

Example:

```text
UNKNOWN
   ↓
CANDIDATE
   ↓
STABLE
   ↓
ACTIVATED
   ↓
HELD
   ↓
RELEASED
```

The system must avoid instantaneous decisions based on one frame.

---

# 32. Gesture Hysteresis

Gesture activation and deactivation thresholds should not necessarily be identical.

Example:

```text
Activation threshold ≠ Release threshold
```

This prevents:

```text
CLICK
NO CLICK
CLICK
NO CLICK
```

caused by noisy landmarks.

---

# 33. Gesture Timing

The gesture engine must account for:

* dwell time
* transition time
* hold duration
* release duration
* motion velocity
* confidence
* previous state

Avoid arbitrary magic numbers.

Every threshold should have:

* reason
* benchmark
* configuration
* test coverage

---

# 34. Gesture Conflict Resolution

When multiple gestures appear simultaneously, the engine must resolve conflicts deterministically.

Priority should depend on:

* confidence
* stability
* gesture specificity
* current mode
* current application profile
* gesture state

Example:

```text
Pinch + movement
```

should not randomly become:

```text
click
drag
scroll
```

---

# 35. Custom Gestures

Users must eventually be able to create custom gestures.

A custom gesture should map to:

* mouse action
* keyboard shortcut
* application command
* media control
* system action
* AirMouse skill
* external command

Example:

```text
Gesture:
Thumb + index + middle

Action:
Ctrl + Shift + T
```

---

# 36. Universal Application Support

AirMouse must not be built specifically around:

* Chrome
* Spotify
* Blender
* KiCad

Those may receive optimized profiles, but the core system must work universally.

The architecture should be:

```text
Universal Input
       +
Optional Application Profile
```

not:

```text
Application-specific hacks
```

---

# 37. Application Profiles

Profiles may modify:

* sensitivity
* acceleration
* gesture mapping
* scrolling
* gesture thresholds
* pointer behavior
* shortcuts
* available gestures

Example:

```text
Global
Chrome
Blender
KiCad
Media
Presentation
Gaming
Custom
```

The profile system must be extensible.

---

# 38. Profile Detection

The system may detect the active application where the operating system allows it.

But if detection is unavailable or unreliable:

```text
Use Global Profile
```

AirMouse must never stop working simply because application detection failed.

---

# 39. Skills Architecture

AirMouse should eventually support modular skills.

Example:

```text
skills/
├── pointer
├── click
├── drag
├── scroll
├── media
├── presentation
├── browser
├── desktop
├── accessibility
├── custom-shortcuts
└── experimental
```

A skill should expose:

```text
Trigger
Action
Conditions
Permissions
Configuration
Tests
```

---

# 40. Skills Must Not Pollute the Core

The core must remain small.

Bad:

```text
if chrome:
if spotify:
if blender:
if vscode:
if ...
```

Preferred:

```text
Core
 ↓
Profile/Skill Manager
 ↓
Selected Skill
```

---

# 41. AI Integration

AI may eventually be used for:

* configuration
* gesture discovery
* profile generation
* troubleshooting
* natural-language customization
* advanced application actions

But AI must NOT sit unnecessarily inside the high-frequency pointer loop.

Bad:

```text
Camera
 ↓
LLM
 ↓
Cursor
```

Preferred:

```text
Camera
 ↓
Vision
 ↓
Motion
 ↓
Input
```

with AI as an optional higher-level layer.

---

# 42. No AI Coach

An AI coach/diagnostic assistant is not part of the required product.

Diagnostics should primarily be deterministic and measurable.

Future AI diagnostics can be considered only if they solve a demonstrated problem.

---

# 43. Gaming

Gaming support is not a core requirement at this stage.

Gaming should not distort the architecture.

If gaming support is added later, it should be evaluated separately because:

* latency requirements differ
* anti-cheat systems may interfere
* relative input may behave differently
* gesture interpretation can be problematic
* accidental actions have different consequences

Do not promise universal gaming compatibility.

---

# 44. Performance Philosophy

AirMouse must be lightweight.

The system should optimize:

```text
CPU ↓
RAM ↓
GPU usage ↓
latency ↓
power usage ↓
camera bandwidth ↓
background overhead ↓
```

while preserving:

```text
tracking quality
smoothness
responsiveness
reliability
```

---

# 45. Performance Budgets

Performance must be measurable.

Track at minimum:

### CPU

Average CPU usage.

Peak CPU usage.

### RAM

Idle RAM.

Tracking RAM.

Peak RAM.

### Latency

Camera-to-landmark latency.

Landmark-to-pointer latency.

End-to-end latency.

### Frame rate

Camera FPS.

Processing FPS.

Output/input update rate.

### Stability

Dropped frames.

Tracking losses.

False gestures.

False clicks.

---

# 46. Low-Spec Hardware Requirement

AirMouse must remain usable on modest hardware.

The development system should not be treated as the minimum hardware requirement.

Optimization must consider:

* older CPUs
* integrated GPUs
* limited RAM
* thermal throttling
* battery operation

Hardware acceleration should be optional.

---

# 47. Power Management

AirMouse should support intelligent camera/processing behavior.

Possible states:

```text
ACTIVE
IDLE
PAUSED
SCREEN LOCKED
DISPLAY OFF
CAMERA ERROR
```

When interaction is unnecessary, processing should be reduced or suspended where possible.

---

# 48. Camera Management

The camera subsystem must:

* detect available cameras
* select the correct camera
* handle camera disappearance
* recover from camera errors
* detect unsupported formats
* negotiate resolution
* avoid unnecessary resolution
* release resources cleanly

The system must not permanently crash because `/dev/videoX` changes.

---

# 49. Camera Auto Configuration

AirMouse should eventually determine appropriate:

* resolution
* FPS
* pixel format
* exposure
* brightness
* processing mode

based on hardware capability.

Manual overrides must remain available.

---

# 50. Privacy

AirMouse should be local-first.

Camera data should not leave the machine unless the user explicitly enables an external service.

Default:

```text
Camera
 ↓
Local processing
 ↓
Local input
```

No mandatory cloud account.

No mandatory telemetry.

No hidden image uploads.

---

# 51. Data Policy

Any stored data should be minimal.

Potential local data:

* settings
* profiles
* calibration
* performance metrics
* optional anonymized diagnostics

Raw camera recordings must never be stored by default.

---

# 52. Safety

AirMouse must have a reliable emergency disable mechanism.

Example:

```text
Global hotkey
```

that immediately disables input generation.

When disabled:

```text
No pointer movement
No clicks
No keyboard actions
No gestures
```

---

# 53. Accidental Action Prevention

High-risk actions require stronger confidence.

For example:

```text
Pointer movement
→ lower threshold

Click
→ higher threshold

Drag
→ higher stability

Keyboard shortcut
→ high confidence

System command
→ very high confidence / explicit confirmation
```

AirMouse should not prioritize cleverness over safety.

---

# 54. Comfort Is a First-Class Metric

AirMouse must not require the user to keep their arm unnaturally extended for long periods.

The system should investigate:

* hand repositioning
* relative movement modes
* clutch gestures
* resting states
* neutral wrist posture
* movement scaling
* fatigue reduction

A user should be able to reposition their hand without unexpectedly moving the pointer.

---

# 55. Hand Repositioning / Clutch

A clutch mechanism should be investigated.

Concept:

```text
Control
 ↓
Clutch gesture
 ↓
Move hand to comfortable position
 ↓
Release clutch
 ↓
Resume pointer control
```

This is potentially more important for long-term comfort than adding dozens of gestures.

---

# 56. Interaction Modes

AirMouse may eventually have:

```text
Pointer Mode
Click Mode
Scroll Mode
Gesture Mode
Paused
Calibration
```

But modes should not become confusing.

Prefer automatic interpretation where possible.

---

# 57. User Experience Principle

The user should not constantly think:

> "Which mode am I in?"

The system should instead infer intent from:

* hand posture
* motion
* velocity
* duration
* context

while preserving explicit mode controls when necessary.

---

# 58. Application-Aware Adaptation

Application context may influence behavior.

For example:

```text
Text editor
→ precision pointer

Browser
→ scrolling emphasis

Media player
→ media gestures

CAD
→ precision + specialized controls

Presentation
→ large simple gestures
```

However, application profiles must remain optional.

---

# 59. Adaptive Learning

AirMouse should eventually learn:

```text
How fast the user moves
How much jitter they naturally produce
Preferred sensitivity
Preferred acceleration
Gesture timing
Gesture confidence
Preferred application settings
```

Learning must be:

* local
* bounded
* reversible
* inspectable
* optional

---

# 60. No Silent Behavioral Drift

The user should never wake up one day and find that AirMouse behaves completely differently because the system learned something.

Adaptation should be gradual.

Example:

```text
Observed behavior
      ↓
Confidence
      ↓
Small adjustment
      ↓
Evaluation
      ↓
Keep / rollback
```

---

# 61. Diagnostics

AirMouse should include a diagnostics system.

Possible metrics:

```text
Camera FPS
Tracking FPS
CPU
RAM
Latency
Hand confidence
Gesture confidence
Pointer jitter
Dropped frames
Tracking losses
False gesture events
Input backend
Active profile
```

---

# 62. Debug Overlay

The debug overlay should be optional.

Possible visualization:

```text
Hand landmarks
Hand bounding box
Confidence
Velocity
Acceleration
Gesture state
Pointer coordinate
Filter state
Active profile
FPS
Latency
```

It must never be required for normal operation.

---

# 63. Recording and Replay

AirMouse should eventually support recording landmark streams rather than requiring raw camera video.

Example:

```text
Camera
 ↓
Landmarks
 ↓
Record
 ↓
Replay
 ↓
Algorithm testing
```

This enables deterministic testing without requiring a camera every time.

---

# 64. Synthetic Testing

Create synthetic hand-motion sequences:

```text
Stationary
Slow movement
Fast movement
Circular motion
Diagonal motion
Abrupt movement
Hand loss
Hand reacquisition
Pinch
Release
Gesture transition
Two-hand interaction
```

These should be automatically tested.

---

# 65. Regression Testing

Every discovered bug should become a regression test.

Example:

```text
Bug:
Cursor jumps after hand reacquisition.

Test:
test_reacquisition_does_not_jump_cursor()
```

The bug should never be allowed to silently return.

---

# 66. Test Pyramid

AirMouse testing should include:

```text
Unit Tests
    ↓
Algorithm Tests
    ↓
Simulation Tests
    ↓
Recorded Landmark Tests
    ↓
Integration Tests
    ↓
Camera Tests
    ↓
Desktop Tests
    ↓
Real-World Tests
```

---

# 67. Real-World Test Matrix

Test under:

### Lighting

* daylight
* low light
* artificial light
* backlighting
* changing brightness

### Background

* plain
* cluttered
* dark
* bright
* moving

### User

* different hand sizes
* left hand
* right hand
* different skin tones
* different distances
* different seating positions

### Motion

* slow
* fast
* tiny
* large
* abrupt
* shaky

---

# 68. Two-Hand Support

Two-hand support should exist in the architecture.

But two-hand interaction must not be allowed to destabilize single-hand pointer control.

The system must identify:

```text
Primary hand
Secondary hand
```

and avoid accidental switching.

---

# 69. Gesture Arbitration

When two hands are detected:

```text
Hand A → pointer
Hand B → gesture
```

or:

```text
Hand A + Hand B → combined gesture
```

depending on context.

The system must not randomly alternate between hands.

---

# 70. Feature Lifecycle

Every major feature should move through:

```text
IDEA
 ↓
RESEARCH
 ↓
RFC
 ↓
PROTOTYPE
 ↓
BENCHMARK
 ↓
REAL-WORLD TEST
 ↓
EXPERIMENTAL
 ↓
BETA
 ↓
CORE
```

or:

```text
REJECTED
```

---

# 71. Experimental Features

Experimental functionality must be isolated.

Example:

```text
experimental/
```

It must not destabilize the stable core.

Experimental features must be clearly marked.

---

# 72. Feature Acceptance Criteria

A feature should enter the core product only when it:

* solves a real problem
* is sufficiently reliable
* has measurable benefit
* does not introduce unacceptable latency
* does not consume excessive resources
* is comfortable
* has tests
* has failure handling
* has documentation

---

# 73. Feature Rejection Criteria

Reject or postpone features that are:

* gimmicky
* unreliable
* excessively expensive
* difficult to maintain
* redundant
* uncomfortable
* unsafe
* impossible to test
* highly application-specific
* dependent on fragile hacks

---

# 74. "Smart" Does Not Mean "Complicated"

AirMouse should prefer:

```text
Simple + reliable
```

over:

```text
Complex + impressive
```

If a 20-line algorithm solves a problem better than a neural network, use the simpler solution.

---

# 75. Architecture

Recommended high-level architecture:

```text
airmouse/
│
├── core/
│   ├── runtime/
│   ├── configuration/
│   ├── state/
│   └── events/
│
├── vision/
│   ├── camera/
│   ├── detection/
│   ├── landmarks/
│   └── tracking/
│
├── motion/
│   ├── velocity/
│   ├── acceleration/
│   ├── filtering/
│   ├── stabilization/
│   └── mapping/
│
├── gestures/
│   ├── detector/
│   ├── state_machine/
│   ├── arbitration/
│   └── custom/
│
├── intent/
│
├── input/
│   ├── abstraction/
│   ├── linux/
│   ├── wayland/
│   └── x11/
│
├── profiles/
│
├── skills/
│
├── calibration/
│
├── diagnostics/
│
├── ui/
│
├── tests/
│
└── docs/
```

The exact directory structure may evolve.

The architecture matters more than the names.

---

# 76. Event-Driven Architecture

Prefer:

```text
Camera Event
 ↓
Tracking Event
 ↓
Motion Event
 ↓
Gesture Event
 ↓
Intent Event
 ↓
Input Event
```

over tightly coupled components.

This improves:

* testing
* debugging
* modularity
* performance analysis
* future plugins
* future AI integration

---

# 77. No God Module

Avoid a giant:

```text
airmouse.py
```

containing everything.

Vision, motion, gestures, input, UI, configuration, and application logic must remain separated.

---

# 78. Configuration

Configuration should be user-editable.

Potential configuration:

```yaml
pointer:
  sensitivity:
  acceleration:
  smoothing:
  precision:

gestures:
  pinch:
  scroll:
  drag:

camera:
  device:
  resolution:
  fps:

profiles:
  enabled:

adaptive:
  enabled:
```

The actual format may change.

---

# 79. Profiles Must Be Portable

A profile should be exportable/importable.

Example:

```text
MyAirMouseProfile
```

could contain:

* pointer settings
* gesture mappings
* application mappings
* calibration preferences

---

# 80. Accessibility

AirMouse should eventually support users who cannot comfortably use a traditional mouse.

Potential future features:

* slower precision mode
* larger gesture tolerances
* customizable gestures
* dwell activation
* reduced-motion interaction
* one-handed operation
* alternative activation methods

Accessibility should be designed into the architecture rather than bolted on later.

---

# 81. Internationalization

The application UI should eventually support multiple languages.

The core gesture system must remain language-independent.

---

# 82. Security

AirMouse can generate system input, so security matters.

Potential protections:

* explicit permissions
* safe defaults
* no arbitrary command execution by gestures by default
* clear dangerous-action warnings
* configurable skill permissions
* isolated plugins
* local-only defaults

---

# 83. Plugin / Skill Security

Third-party skills should not automatically gain unrestricted system access.

Future permission model:

```text
Skill
 ↓
Requested capability
 ↓
User permission
 ↓
Execution
```

---

# 84. Commercial Direction

AirMouse should remain architecturally capable of becoming a paid application later.

The immediate development philosophy is:

```text
Open development
 ↓
Reliable Linux product
 ↓
Mature architecture
 ↓
Future commercial product
```

Do not compromise the technical architecture prematurely for monetization.

---

# 85. Future Cross-Platform Architecture

Eventually:

```text
Shared Core
├── Vision
├── Tracking
├── Motion
├── Gestures
├── Intent
├── Profiles
└── Skills
```

Platform layer:

```text
Linux
Windows
macOS
```

The shared core should not contain platform-specific code.

---

# 86. No Rewrite Principle

Do not rewrite stable subsystems simply because a newer technology exists.

Replace a subsystem only when the new approach demonstrates a measurable improvement in:

* latency
* reliability
* resource consumption
* compatibility
* maintainability
* user experience

---

# 87. Development Agents

AirMouse can use specialized development agents.

Recommended roles:

1. Architecture Agent
2. Computer Vision Agent
3. Tracking Agent
4. Motion/Filtering Agent
5. Gesture Agent
6. Linux Input Agent
7. Wayland/X11 Agent
8. Performance Agent
9. UX Agent
10. Accessibility Agent
11. Testing Agent
12. Security Agent
13. Research Agent
14. Documentation Agent
15. Release Agent

These agents should not independently rewrite the project.

A coordinating architecture process should arbitrate changes.

---

# 88. Agent Rules

Every development agent must:

1. Inspect the current implementation.
2. Understand existing behavior.
3. Search relevant documentation/research.
4. Identify the actual problem.
5. Propose alternatives.
6. Explain trade-offs.
7. Make the smallest justified change.
8. Run tests.
9. Benchmark where appropriate.
10. Test real-world behavior when necessary.
11. Document important decisions.
12. Avoid unrelated refactoring.

---

# 89. Agent Disagreement

If agents disagree:

```text
Do not average the opinions.
```

Instead:

```text
Identify disagreement
 ↓
Compare evidence
 ↓
Prototype alternatives
 ↓
Benchmark
 ↓
Choose based on evidence
```

---

# 90. Development Rule

> **Build less. Measure more.**

A feature that looks impressive but makes AirMouse worse must be removed.

---

# 91. Documentation Architecture

There should be one authoritative product-goal document:

```text
ULTIMATE_GOAL.md
```

Other documentation should describe implementation details, architecture, APIs, testing, development, or release processes.

Do not maintain multiple competing "ultimate goal" documents.

---

# 92. Documentation Cleanup

Before deleting old goal documents:

1. Search the repository.
2. Find references.
3. Identify unique information.
4. Move useful information into the authoritative document.
5. Update links.
6. Remove obsolete duplicates.
7. Run documentation/build tests.

Never blindly execute:

```bash
rm *.md
```

---

# 93. Important Distinction

The goal is to consolidate **obsolete goal/specification documents**.

Do NOT delete:

* source code
* tests
* useful architecture documentation
* API documentation
* research
* changelogs
* licenses
* contributor documentation

unless there is a separate explicit decision to do so.

---

# 94. Development Priorities

The order of priorities should be:

## Priority 1 — Reliable camera

Camera selection and recovery.

## Priority 2 — Reliable tracking

Stable hand detection.

## Priority 3 — Cursor stability

Eliminate vibration.

## Priority 4 — Cursor responsiveness

Reduce latency.

## Priority 5 — Adaptive motion

Precision + acceleration.

## Priority 6 — Tracking loss safety

Freeze + no input.

## Priority 7 — Reliable click

Extremely low false-click rate.

## Priority 8 — Drag

Stable press/hold/release.

## Priority 9 — Scroll

Reliable and comfortable.

## Priority 10 — Calibration

Fast and repeatable.

## Priority 11 — Gesture engine

State-based and extensible.

## Priority 12 — Application profiles

Universal architecture.

## Priority 13 — Skills

Extensibility.

## Priority 14 — Adaptive personalization

Learn user preferences safely.

## Priority 15 — Advanced features

Only after the core is excellent.

---

# 95. Definition of "Production Ready"

AirMouse is not production-ready when:

```text
"It works on my laptop."
```

It becomes production-ready when:

* camera failures recover
* tracking is stable
* pointer is smooth
* pointer is responsive
* false clicks are rare
* hand loss is safe
* gestures are predictable
* CPU usage is acceptable
* RAM usage is acceptable
* calibration works
* different resolutions work
* different Linux environments are tested
* application profiles work
* settings are reliable
* diagnostics exist
* regression tests exist
* packaging works
* documentation exists
* real users can operate it without technical knowledge

---

# 96. Core Product Metrics

AirMouse should eventually define explicit targets for:

### Pointer

```text
Latency
Jitter
Overshoot
Accuracy
```

### Gesture

```text
True positive rate
False positive rate
False activation rate
Recognition latency
```

### Tracking

```text
Tracking confidence
Tracking loss rate
Reacquisition time
```

### Performance

```text
CPU
RAM
FPS
Power
```

### UX

```text
Calibration time
Learning time
Hand fatigue
User satisfaction
```

---

# 97. The Ultimate UX Test

A successful user should eventually be able to use AirMouse without thinking:

> "I am controlling a webcam."

Instead:

> "I am controlling my computer with my hand."

That is the real product.

---

# 98. Long-Term Possibilities

Future features may include:

* advanced 3D interaction
* depth-camera support
* richer two-hand interaction
* spatial UI
* presentation mode
* accessibility modes
* advanced CAD interaction
* creative software interaction
* gaming experiments
* gesture macros
* user-trained gestures
* plugin ecosystem
* cross-platform releases
* optional AI integration
* optional voice integration through external systems
* wearable/device integration

None of these should be implemented merely because they sound impressive.

Each must pass the research → prototype → benchmark → real-world validation process.

---

# 99. Five-Year Direction

The exact five-year vision is intentionally undefined.

That is acceptable.

The architecture should therefore maximize:

```text
Extensibility
Modularity
Compatibility
Reliability
```

rather than locking the project into one future product concept.

The system should be capable of evolving from:

```text
Gesture Mouse
```

into:

```text
Intelligent Human-Computer Input Platform
```

without requiring a complete rewrite.

---

# 100. Ultimate North Star

Everything ultimately comes back to one statement:

> **AirMouse should make computers understand human hand movement as naturally as they understand a physical mouse.**

Not with gimmicks.

Not with unnecessary AI.

Not with excessive hardware.

Not with unreliable gestures.

Not with cloud dependency.

Instead:

```text
Natural
+
Precise
+
Smooth
+
Fast
+
Lightweight
+
Smart
+
Comfortable
+
Reliable
+
Private
+
Extensible
```

That is AirMouse.

---

# 101. Final Engineering Rule

Before adding anything, ask:

> **Does this make AirMouse genuinely better for a human being using a computer?**

If yes:

```text
Research it.
Prototype it.
Measure it.
Test it.
Keep it if it works.
```

If no:

```text
Do not build it.
```

---

# 102. Final Product Statement

**AirMouse is a local-first, intelligent gesture-based input platform designed to make desktop computers naturally controllable through human hand movement.**

Its first goal is not to have the most features.

Its first goal is to make the fundamental interaction **feel exceptionally good**.

Once that foundation is reliable, everything else can grow on top of it.

> **Smooth first. Reliable second. Smart third. Features fourth.**