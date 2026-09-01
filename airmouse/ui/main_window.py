"""
Main Window GUI for Air Mouse

PySide6-based GUI with:
- Camera preview with hand landmarks overlay
- Real-time stats display
- Settings dialog
- System tray integration
- Start/Stop/Pause controls
"""

import sys
import logging
from typing import Optional, List
from dataclasses import asdict
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QCheckBox, QSpinBox, QDoubleSpinBox,
    QComboBox, QGroupBox, QFormLayout, QTabWidget, QMessageBox,
    QSystemTrayIcon, QMenu, QStyle, QSlider, QProgressBar,
    QTextEdit, QSplitter, QFrame, QSizePolicy
)
from PySide6.QtCore import (
    Qt, QTimer, QThread, Signal, Slot, QSize, QPoint, QRect
)
from PySide6.QtGui import (
    QImage, QPixmap, QIcon, QFont, QColor, QPainter, QPen,
    QAction, QCloseEvent, QResizeEvent, QPalette
)

from ..camera.manager import CameraSettings
from ..vision.hand_tracker import HandTrackerSettings, Hand
from ..vision.gestures import GestureConfig, GestureEvent, GestureType
from ..control.cursor import CursorConfig, SmoothingAlgorithm
from ..input.uinput_mouse import UInputDeviceConfig
from ..control.main_loop import (
    AirMouseController, AirMouseConfig, AirMouseState, PerformanceStats
)

logger = logging.getLogger(__name__)


class CameraPreviewWidget(QLabel):
    """Widget displaying camera feed with hand landmarks overlay."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #2b2b2b; border: 1px solid #555;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._current_frame = None
        self._hands = []
        self._show_landmarks = True
        self._show_connections = True
        self._show_palm = True
        self._show_index_tip = True
        self._mirror = True

    def update_frame(self, frame, hands=None):
        """Update with new frame and hand data."""
        self._current_frame = frame
        self._hands = hands or []
        self.update()

    def set_show_landmarks(self, show: bool):
        self._show_landmarks = show
        self.update()

    def set_show_connections(self, show: bool):
        self._show_connections = show
        self.update()

    def set_mirror(self, mirror: bool):
        self._mirror = mirror
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Fill background
        painter.fillRect(self.rect(), QColor(43, 43, 43))

        if self._current_frame is not None:
            # Convert OpenCV frame to QImage
            h, w = self._current_frame.shape[:2]
            bytes_per_line = 3 * w

            # Create QImage from frame data
            qimg = QImage(
                self._current_frame.data, w, h, bytes_per_line,
                QImage.Format_RGB888
            ).rgbSwapped()

            # Scale to widget size maintaining aspect ratio
            scaled = qimg.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )

            # Center the image
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2

            painter.drawImage(x, y, scaled)

            # Draw hand landmarks overlay
            if self._hands and (self._show_landmarks or self._show_connections):
                self._draw_hands(painter, x, y, scaled.width(), scaled.height(), w, h)

        else:
            # Show placeholder text
            painter.setPen(QColor(150, 150, 150))
            painter.setFont(QFont("Arial", 14))
            painter.drawText(self.rect(), Qt.AlignCenter, "Camera Preview\nWaiting for camera...")

    def _draw_hands(self, painter, offset_x, offset_y, display_w, display_h, frame_w, frame_h):
        """Draw hand landmarks and connections."""
        scale_x = display_w / frame_w
        scale_y = display_h / frame_h

        for hand in self._hands:
            if not hand.landmarks:
                continue

            # Landmark colors
            landmark_color = QColor(0, 255, 0)
            connection_color = QColor(255, 0, 0)
            palm_color = QColor(255, 0, 0)
            index_color = QColor(255, 255, 0)

            # Draw connections
            if self._show_connections:
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),   # thumb
                    (0, 5), (5, 6), (6, 7), (7, 8),   # index
                    (5, 9), (9, 10), (10, 11), (11, 12),  # middle
                    (9, 13), (13, 14), (14, 15), (15, 16),  # ring
                    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
                    (0, 17),  # palm
                ]
                painter.setPen(QPen(connection_color, 2))
                for start_idx, end_idx in connections:
                    if start_idx < len(hand.landmarks) and end_idx < len(hand.landmarks):
                        p1 = hand.landmarks[start_idx]
                        p2 = hand.landmarks[end_idx]
                        x1 = int(offset_x + p1.x * display_w)
                        y1 = int(offset_y + p1.y * display_h)
                        x2 = int(offset_x + p2.x * display_w)
                        y2 = int(offset_y + p2.y * display_h)
                        if self._mirror:
                            x1 = offset_x + display_w - (x1 - offset_x)
                            x2 = offset_x + display_w - (x2 - offset_x)
                        painter.drawLine(x1, y1, x2, y2)

            # Draw landmarks
            if self._show_landmarks:
                painter.setPen(QPen(landmark_color, 1))
                painter.setBrush(landmark_color)
                for lm in hand.landmarks:
                    x = int(offset_x + lm.x * display_w)
                    y = int(offset_y + lm.y * display_h)
                    if self._mirror:
                        x = offset_x + display_w - (x - offset_x)
                    painter.drawEllipse(x - 4, y - 4, 8, 8)

            # Draw palm center
            if self._show_palm and hand.palm_center:
                painter.setPen(QPen(palm_color, 2))
                painter.setBrush(palm_color)
                x = int(offset_x + hand.palm_center.x * display_w)
                y = int(offset_y + hand.palm_center.y * display_h)
                if self._mirror:
                    x = offset_x + display_w - (x - offset_x)
                painter.drawEllipse(x - 8, y - 8, 16, 16)

            # Draw index tip (cursor point)
            if self._show_index_tip and hand.index_tip:
                painter.setPen(QPen(index_color, 3))
                painter.setBrush(Qt.NoBrush)
                x = int(offset_x + hand.index_tip.x * display_w)
                y = int(offset_y + hand.index_tip.y * display_h)
                if self._mirror:
                    x = offset_x + display_w - (x - offset_x)
                painter.drawEllipse(x - 10, y - 10, 20, 20)

            # Draw handedness label
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Arial", 10, QFont.Bold))
            label_x = offset_x + 10
            label_y = offset_y + 25
            painter.drawText(label_x, label_y, f"{hand.handedness} Hand ({hand.confidence:.2f})")


class StatsWidget(QWidget):
    """Widget displaying performance statistics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Title
        title = QLabel("Performance")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title)

        # Stats labels
        self.fps_label = QLabel("FPS: --")
        self.frame_time_label = QLabel("Frame Time: -- ms")
        self.hand_time_label = QLabel("Hand Detection: -- ms")
        self.gesture_time_label = QLabel("Gesture: -- ms")
        self.cursor_time_label = QLabel("Cursor: -- ms")
        self.mouse_time_label = QLabel("Mouse: -- ms")
        self.frames_label = QLabel("Frames: 0")
        self.dropped_label = QLabel("Dropped: 0")

        for label in [
            self.fps_label, self.frame_time_label, self.hand_time_label,
            self.gesture_time_label, self.cursor_time_label, self.mouse_time_label,
            self.frames_label, self.dropped_label
        ]:
            label.setFont(QFont("Monospace", 9))
            layout.addWidget(label)

        layout.addStretch()

    def update_stats(self, stats: PerformanceStats):
        """Update displayed statistics."""
        self.fps_label.setText(f"FPS: {stats.fps:.1f}")
        self.frame_time_label.setText(f"Frame Time: {stats.frame_time_ms:.1f} ms")
        self.hand_time_label.setText(f"Hand Detection: {stats.hand_detection_time_ms:.1f} ms")
        self.gesture_time_label.setText(f"Gesture: {stats.gesture_time_ms:.1f} ms")
        self.cursor_time_label.setText(f"Cursor: {stats.cursor_time_ms:.1f} ms")
        self.mouse_time_label.setText(f"Mouse: {stats.mouse_time_ms:.1f} ms")
        self.frames_label.setText(f"Frames: {stats.frames_processed}")
        self.dropped_label.setText(f"Dropped: {stats.frames_dropped}")


class StatusWidget(QWidget):
    """Widget displaying current state and status."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # State indicator
        self.state_label = QLabel("State: STOPPED")
        self.state_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.state_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.state_label)

        # Status details
        self.details_label = QLabel("Ready to start")
        self.details_label.setAlignment(Qt.AlignCenter)
        self.details_label.setWordWrap(True)
        layout.addWidget(self.details_label)

        # Hand info
        self.hand_info_label = QLabel("No hand detected")
        self.hand_info_label.setAlignment(Qt.AlignCenter)
        self.hand_info_label.setWordWrap(True)
        layout.addWidget(self.hand_info_label)

        layout.addStretch()

    def update_state(self, state: AirMouseState, details: str = ""):
        """Update state display."""
        colors = {
            AirMouseState.STOPPED: "#888",
            AirMouseState.STARTING: "#ffaa00",
            AirMouseState.RUNNING: "#00aa00",
            AirMouseState.PAUSED: "#ffaa00",
            AirMouseState.ERROR: "#ff0000",
        }
        color = colors.get(state, "#888")
        self.state_label.setText(f"State: {state.value.upper()}")
        self.state_label.setStyleSheet(f"color: {color};")

        if details:
            self.details_label.setText(details)

    def update_hand_info(self, hand: Optional[object]):
        """Update hand information display."""
        if hand and hasattr(hand, 'landmarks') and hand.landmarks:
            fingers = []
            if hasattr(hand, 'finger_states'):
                for name, extended in hand.finger_states.items():
                    if extended:
                        fingers.append(name.capitalize())
            finger_str = ", ".join(fingers) if fingers else "None"
            self.hand_info_label.setText(
                f"Hand: {hand.handedness}\n"
                f"Confidence: {hand.confidence:.2f}\n"
                f"Extended: {finger_str}\n"
                f"Pinch: {hand.is_pinch('thumb', 'index'):.3f}\n"
                f"Fist: {hand.is_fist()}"
            )
        else:
            self.hand_info_label.setText("No hand detected")


class ControlWidget(QWidget):
    """Control buttons for air mouse."""

    start_clicked = Signal()
    stop_clicked = Signal()
    pause_clicked = Signal()
    settings_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Main buttons
        self.start_btn = QPushButton("Start")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #2e7d32;
                color: white;
                font-weight: bold;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #388e3c; }
            QPushButton:disabled { background-color: #555; }
        """)
        self.start_btn.clicked.connect(self.start_clicked.emit)
        layout.addWidget(self.start_btn)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setMinimumHeight(40)
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.pause_clicked.emit)
        layout.addWidget(self.pause_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #c62828;
                color: white;
                font-weight: bold;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #d32f2f; }
            QPushButton:disabled { background-color: #555; }
        """)
        self.stop_btn.clicked.connect(self.stop_clicked.emit)
        layout.addWidget(self.stop_btn)

        layout.addSpacing(20)

        # Settings button
        self.settings_btn = QPushButton("Settings")
        self.settings_btn.setMinimumHeight(35)
        self.settings_btn.clicked.connect(self.settings_clicked.emit)
        layout.addWidget(self.settings_btn)

        layout.addStretch()

    def set_state(self, state: AirMouseState):
        """Update button states based on air mouse state."""
        if state == AirMouseState.STOPPED:
            self.start_btn.setEnabled(True)
            self.start_btn.setText("Start")
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
        elif state == AirMouseState.STARTING:
            self.start_btn.setEnabled(False)
            self.start_btn.setText("Starting...")
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
        elif state == AirMouseState.RUNNING:
            self.start_btn.setEnabled(False)
            self.start_btn.setText("Running")
            self.pause_btn.setEnabled(True)
            self.pause_btn.setText("Pause")
            self.stop_btn.setEnabled(True)
        elif state == AirMouseState.PAUSED:
            self.start_btn.setEnabled(False)
            self.start_btn.setText("Paused")
            self.pause_btn.setEnabled(True)
            self.pause_btn.setText("Resume")
            self.stop_btn.setEnabled(True)
        elif state == AirMouseState.ERROR:
            self.start_btn.setEnabled(True)
            self.start_btn.setText("Start")
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)


class SettingsDialog(QWidget):
    """Settings dialog for air mouse configuration."""

    settings_changed = Signal(dict)

    def __init__(self, config: AirMouseConfig, parent=None):
        super().__init__(parent, Qt.Dialog | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.setWindowTitle("Air Mouse Settings")
        self.setMinimumWidth(500)
        self._config = config
        self._widgets = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Tab widget
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Camera tab
        tabs.addTab(self._create_camera_tab(), "Camera")

        # Tracking tab
        tabs.addTab(self._create_tracking_tab(), "Tracking")

        # Cursor tab
        tabs.addTab(self._create_cursor_tab(), "Cursor")

        # Gestures tab
        tabs.addTab(self._create_gestures_tab(), "Gestures")

        # Brightness tab
        tabs.addTab(self._create_brightness_tab(), "Brightness")

        # Advanced tab
        tabs.addTab(self._create_advanced_tab(), "Advanced")

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._apply_settings)
        btn_layout.addWidget(apply_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.close)
        btn_layout.addWidget(cancel_btn)

        layout.addLayout(btn_layout)

    def _create_camera_tab(self) -> QWidget:
        widget = QWidget()
        layout = QFormLayout(widget)

        # Camera index
        self._widgets['camera_index'] = QSpinBox()
        self._widgets['camera_index'].setRange(-1, 10)
        self._widgets['camera_index'].setValue(self._config.camera.device_index)
        self._widgets['camera_index'].setSpecialValueText("Auto")
        layout.addRow("Camera Index:", self._widgets['camera_index'])

        # Resolution
        self._widgets['camera_width'] = QSpinBox()
        self._widgets['camera_width'].setRange(160, 3840)
        self._widgets['camera_width'].setValue(self._config.camera.width)
        layout.addRow("Width:", self._widgets['camera_width'])

        self._widgets['camera_height'] = QSpinBox()
        self._widgets['camera_height'].setRange(120, 2160)
        self._widgets['camera_height'].setValue(self._config.camera.height)
        layout.addRow("Height:", self._widgets['camera_height'])

        # FPS
        self._widgets['camera_fps'] = QSpinBox()
        self._widgets['camera_fps'].setRange(15, 120)
        self._widgets['camera_fps'].setValue(self._config.camera.fps)
        layout.addRow("Target FPS:", self._widgets['camera_fps'])

        return widget

    def _create_tracking_tab(self) -> QWidget:
        widget = QWidget()
        layout = QFormLayout(widget)

        # Max hands
        self._widgets['max_hands'] = QSpinBox()
        self._widgets['max_hands'].setRange(1, 4)
        self._widgets['max_hands'].setValue(self._config.hand_tracker.max_hands)
        layout.addRow("Max Hands:", self._widgets['max_hands'])

        # Detection confidence
        self._widgets['detection_conf'] = QDoubleSpinBox()
        self._widgets['detection_conf'].setRange(0.1, 1.0)
        self._widgets['detection_conf'].setSingleStep(0.05)
        self._widgets['detection_conf'].setValue(self._config.hand_tracker.min_detection_confidence)
        layout.addRow("Detection Confidence:", self._widgets['detection_conf'])

        # Tracking confidence
        self._widgets['tracking_conf'] = QDoubleSpinBox()
        self._widgets['tracking_conf'].setRange(0.1, 1.0)
        self._widgets['tracking_conf'].setSingleStep(0.05)
        self._widgets['tracking_conf'].setValue(self._config.hand_tracker.min_tracking_confidence)
        layout.addRow("Tracking Confidence:", self._widgets['tracking_conf'])

        # Model complexity
        self._widgets['model_complexity'] = QComboBox()
        self._widgets['model_complexity'].addItems(["Lite (0)", "Full (1)"])
        self._widgets['model_complexity'].setCurrentIndex(self._config.hand_tracker.model_complexity)
        layout.addRow("Model:", self._widgets['model_complexity'])

        return widget

    def _create_cursor_tab(self) -> QWidget:
        widget = QWidget()
        layout = QFormLayout(widget)

        # Sensitivity
        self._widgets['sensitivity'] = QDoubleSpinBox()
        self._widgets['sensitivity'].setRange(0.1, 5.0)
        self._widgets['sensitivity'].setSingleStep(0.1)
        self._widgets['sensitivity'].setValue(self._config.cursor.sensitivity)
        layout.addRow("Sensitivity:", self._widgets['sensitivity'])

        # Acceleration
        self._widgets['acceleration'] = QDoubleSpinBox()
        self._widgets['acceleration'].setRange(0.5, 3.0)
        self._widgets['acceleration'].setSingleStep(0.1)
        self._widgets['acceleration'].setValue(self._config.cursor.acceleration)
        layout.addRow("Acceleration:", self._widgets['acceleration'])

        # Dead zone
        self._widgets['dead_zone'] = QDoubleSpinBox()
        self._widgets['dead_zone'].setRange(0.0, 0.2)
        self._widgets['dead_zone'].setSingleStep(0.01)
        self._widgets['dead_zone'].setValue(self._config.cursor.dead_zone_radius)
        layout.addRow("Dead Zone:", self._widgets['dead_zone'])

        # Smoothing algorithm
        self._widgets['smoothing'] = QComboBox()
        for algo in SmoothingAlgorithm:
            self._widgets['smoothing'].addItem(algo.value.capitalize(), algo)
        idx = self._widgets['smoothing'].findData(self._config.cursor.smoothing)
        if idx >= 0:
            self._widgets['smoothing'].setCurrentIndex(idx)
        layout.addRow("Smoothing:", self._widgets['smoothing'])

        # EMA Alpha
        self._widgets['ema_alpha'] = QDoubleSpinBox()
        self._widgets['ema_alpha'].setRange(0.01, 1.0)
        self._widgets['ema_alpha'].setSingleStep(0.05)
        self._widgets['ema_alpha'].setValue(self._config.cursor.ema_alpha)
        layout.addRow("EMA Alpha:", self._widgets['ema_alpha'])

        # Invert axes
        self._widgets['invert_x'] = QCheckBox("Invert X")
        self._widgets['invert_x'].setChecked(self._config.cursor.invert_x)
        layout.addRow("", self._widgets['invert_x'])

        self._widgets['invert_y'] = QCheckBox("Invert Y")
        self._widgets['invert_y'].setChecked(self._config.cursor.invert_y)
        layout.addRow("", self._widgets['invert_y'])

        # Use index tip
        self._widgets['use_index_tip'] = QCheckBox("Use Index Tip (vs Palm Center)")
        self._widgets['use_index_tip'].setChecked(self._config.cursor.use_index_tip)
        layout.addRow("", self._widgets['use_index_tip'])

        return widget

    def _create_gestures_tab(self) -> QWidget:
        widget = QWidget()
        layout = QFormLayout(widget)

        # Pinch threshold
        self._widgets['pinch_threshold'] = QDoubleSpinBox()
        self._widgets['pinch_threshold'].setRange(0.01, 0.2)
        self._widgets['pinch_threshold'].setSingleStep(0.01)
        self._widgets['pinch_threshold'].setValue(self._config.gestures.pinch_threshold)
        layout.addRow("Pinch Threshold:", self._widgets['pinch_threshold'])

        # Pinch release threshold
        self._widgets['pinch_release'] = QDoubleSpinBox()
        self._widgets['pinch_release'].setRange(0.01, 0.3)
        self._widgets['pinch_release'].setSingleStep(0.01)
        self._widgets['pinch_release'].setValue(self._config.gestures.pinch_release_threshold)
        layout.addRow("Release Threshold:", self._widgets['pinch_release'])

        # Scroll sensitivity
        self._widgets['scroll_sens'] = QDoubleSpinBox()
        self._widgets['scroll_sens'].setRange(0.005, 0.1)
        self._widgets['scroll_sens'].setSingleStep(0.005)
        self._widgets['scroll_sens'].setValue(self._config.gestures.scroll_sensitivity)
        layout.addRow("Scroll Sensitivity:", self._widgets['scroll_sens'])

        # Fist hold time
        self._widgets['fist_time'] = QDoubleSpinBox()
        self._widgets['fist_time'].setRange(0.1, 2.0)
        self._widgets['fist_time'].setSingleStep(0.1)
        self._widgets['fist_time'].setValue(self._config.gestures.fist_hold_time)
        layout.addRow("Fist Hold Time (s):", self._widgets['fist_time'])

        # Drag hold time
        self._widgets['drag_time'] = QDoubleSpinBox()
        self._widgets['drag_time'].setRange(0.1, 1.0)
        self._widgets['drag_time'].setSingleStep(0.1)
        self._widgets['drag_time'].setValue(self._config.gestures.drag_hold_time)
        layout.addRow("Drag Hold Time (s):", self._widgets['drag_time'])

        # Click max duration
        self._widgets['click_duration'] = QDoubleSpinBox()
        self._widgets['click_duration'].setRange(0.1, 1.0)
        self._widgets['click_duration'].setSingleStep(0.05)
        self._widgets['click_duration'].setValue(self._config.gestures.click_max_duration)
        layout.addRow("Click Max Duration (s):", self._widgets['click_duration'])

        return widget

    def _create_advanced_tab(self) -> QWidget:
        widget = QWidget()
        layout = QFormLayout(widget)

        # Target FPS
        self._widgets['target_fps'] = QSpinBox()
        self._widgets['target_fps'].setRange(15, 120)
        self._widgets['target_fps'].setValue(self._config.target_fps)
        layout.addRow("Target FPS:", self._widgets['target_fps'])

        # Emergency stop
        self._widgets['emergency_stop'] = QCheckBox("Enable Corner Escape")
        self._widgets['emergency_stop'].setChecked(self._config.emergency_stop_corner)
        layout.addRow("", self._widgets['emergency_stop'])

        # Virtual mouse name
        self._widgets['mouse_name'] = QLabel(self._config.virtual_mouse.name)
        self._widgets['mouse_name'].setStyleSheet("color: #888;")
        layout.addRow("Virtual Mouse:", self._widgets['mouse_name'])

        # Vendor/Product IDs
        self._widgets['vendor_id'] = QSpinBox()
        self._widgets['vendor_id'].setRange(0, 0xFFFF)
        self._widgets['vendor_id'].setValue(self._config.virtual_mouse.vendor_id)
        layout.addRow("Vendor ID:", self._widgets['vendor_id'])

        self._widgets['product_id'] = QSpinBox()
        self._widgets['product_id'].setRange(0, 0xFFFF)
        self._widgets['product_id'].setValue(self._config.virtual_mouse.product_id)
        layout.addRow("Product ID:", self._widgets['product_id'])

        return widget

    def _create_brightness_tab(self) -> QWidget:
        widget = QWidget()
        layout = QFormLayout(widget)

        # Enable auto-brightness
        self._widgets['brightness_enabled'] = QCheckBox("Enable Auto-Brightness")
        self._widgets['brightness_enabled'].setChecked(self._config.brightness.enabled)
        layout.addRow("", self._widgets['brightness_enabled'])

        # Require ALS
        self._widgets['brightness_require_als'] = QCheckBox("Require Ambient Light Sensor (disable if not found)")
        self._widgets['brightness_require_als'].setChecked(self._config.brightness.require_als)
        layout.addRow("", self._widgets['brightness_require_als'])

        # Min brightness
        self._widgets['brightness_min'] = QDoubleSpinBox()
        self._widgets['brightness_min'].setRange(0, 100)
        self._widgets['brightness_min'].setSingleStep(1)
        self._widgets['brightness_min'].setSuffix("%")
        self._widgets['brightness_min'].setValue(int(self._config.brightness.min_brightness * 100))
        layout.addRow("Min Brightness:", self._widgets['brightness_min'])

        # Max brightness
        self._widgets['brightness_max'] = QDoubleSpinBox()
        self._widgets['brightness_max'].setRange(0, 100)
        self._widgets['brightness_max'].setSingleStep(1)
        self._widgets['brightness_max'].setSuffix("%")
        self._widgets['brightness_max'].setValue(int(self._config.brightness.max_brightness * 100))
        layout.addRow("Max Brightness:", self._widgets['brightness_max'])

        # Poll interval
        self._widgets['brightness_poll_interval'] = QDoubleSpinBox()
        self._widgets['brightness_poll_interval'].setRange(0.1, 10.0)
        self._widgets['brightness_poll_interval'].setSingleStep(0.1)
        self._widgets['brightness_poll_interval'].setSuffix(" s")
        self._widgets['brightness_poll_interval'].setValue(self._config.brightness.poll_interval)
        layout.addRow("Poll Interval:", self._widgets['brightness_poll_interval'])

        # Smoothing alpha
        self._widgets['brightness_smoothing_alpha'] = QDoubleSpinBox()
        self._widgets['brightness_smoothing_alpha'].setRange(0.01, 1.0)
        self._widgets['brightness_smoothing_alpha'].setSingleStep(0.05)
        self._widgets['brightness_smoothing_alpha'].setValue(self._config.brightness.smoothing_alpha)
        layout.addRow("Smoothing (EMA α):", self._widgets['brightness_smoothing_alpha'])

        # Hysteresis
        self._widgets['brightness_hysteresis'] = QDoubleSpinBox()
        self._widgets['brightness_hysteresis'].setRange(0, 1000)
        self._widgets['brightness_hysteresis'].setSingleStep(1)
        self._widgets['brightness_hysteresis'].setSuffix(" lux")
        self._widgets['brightness_hysteresis'].setValue(self._config.brightness.hysteresis_lux)
        layout.addRow("Hysteresis:", self._widgets['brightness_hysteresis'])

        # Lux at min brightness
        self._widgets['brightness_lux_min'] = QDoubleSpinBox()
        self._widgets['brightness_lux_min'].setRange(0, 10000)
        self._widgets['brightness_lux_min'].setSingleStep(10)
        self._widgets['brightness_lux_min'].setSuffix(" lux")
        self._widgets['brightness_lux_min'].setValue(self._config.brightness.lux_at_min)
        layout.addRow("Lux at Min Brightness:", self._widgets['brightness_lux_min'])

        # Lux at max brightness
        self._widgets['brightness_lux_max'] = QDoubleSpinBox()
        self._widgets['brightness_lux_max'].setRange(0, 100000)
        self._widgets['brightness_lux_max'].setSingleStep(100)
        self._widgets['brightness_lux_max'].setSuffix(" lux")
        self._widgets['brightness_lux_max'].setValue(self._config.brightness.lux_at_max)
        layout.addRow("Lux at Max Brightness:", self._widgets['brightness_lux_max'])

        # Restore on exit
        self._widgets['brightness_restore_on_exit'] = QCheckBox("Restore Original Brightness on Exit")
        self._widgets['brightness_restore_on_exit'].setChecked(self._config.brightness.restore_on_exit)
        layout.addRow("", self._widgets['brightness_restore_on_exit'])

        # Read-only status display
        self._widgets['brightness_current_lux'] = QLabel("-- lux")
        self._widgets['brightness_current_lux'].setStyleSheet("color: #888; font-family: monospace;")
        layout.addRow("Current Lux:", self._widgets['brightness_current_lux'])

        self._widgets['brightness_current_brightness'] = QLabel("--%")
        self._widgets['brightness_current_brightness'].setStyleSheet("color: #888; font-family: monospace;")
        layout.addRow("Current Brightness:", self._widgets['brightness_current_brightness'])

        self._widgets['brightness_als_status'] = QLabel("Detecting...")
        self._widgets['brightness_als_status'].setStyleSheet("color: #888; font-family: monospace;")
        layout.addRow("ALS Status:", self._widgets['brightness_als_status'])

        self._widgets['brightness_backlight_status'] = QLabel("Detecting...")
        self._widgets['brightness_backlight_status'].setStyleSheet("color: #888; font-family: monospace;")
        layout.addRow("Backlight Status:", self._widgets['brightness_backlight_status'])

        return widget

    def _apply_settings(self):
        """Apply settings to config and emit signal."""
        # Camera
        self._config.camera.device_index = self._widgets['camera_index'].value()
        self._config.camera.width = self._widgets['camera_width'].value()
        self._config.camera.height = self._widgets['camera_height'].value()
        self._config.camera.fps = self._widgets['camera_fps'].value()

        # Tracking
        self._config.hand_tracker.max_hands = self._widgets['max_hands'].value()
        self._config.hand_tracker.min_detection_confidence = self._widgets['detection_conf'].value()
        self._config.hand_tracker.min_tracking_confidence = self._widgets['tracking_conf'].value()
        self._config.hand_tracker.model_complexity = self._widgets['model_complexity'].currentIndex()

        # Cursor
        self._config.cursor.sensitivity = self._widgets['sensitivity'].value()
        self._config.cursor.acceleration = self._widgets['acceleration'].value()
        self._config.cursor.dead_zone_radius = self._widgets['dead_zone'].value()
        self._config.cursor.smoothing = self._widgets['smoothing'].currentData()
        self._config.cursor.ema_alpha = self._widgets['ema_alpha'].value()
        self._config.cursor.invert_x = self._widgets['invert_x'].isChecked()
        self._config.cursor.invert_y = self._widgets['invert_y'].isChecked()
        self._config.cursor.use_index_tip = self._widgets['use_index_tip'].isChecked()

        # Gestures
        self._config.gestures.pinch_threshold = self._widgets['pinch_threshold'].value()
        self._config.gestures.pinch_release_threshold = self._widgets['pinch_release'].value()
        self._config.gestures.scroll_sensitivity = self._widgets['scroll_sens'].value()
        self._config.gestures.fist_hold_time = self._widgets['fist_time'].value()
        self._config.gestures.drag_hold_time = self._widgets['drag_time'].value()
        self._config.gestures.click_max_duration = self._widgets['click_duration'].value()

        # Advanced
        self._config.target_fps = self._widgets['target_fps'].value()
        self._config.emergency_stop_corner = self._widgets['emergency_stop'].isChecked()
        self._config.virtual_mouse.vendor_id = self._widgets['vendor_id'].value()
        self._config.virtual_mouse.product_id = self._widgets['product_id'].value()

        # Brightness
        self._config.brightness.enabled = self._widgets['brightness_enabled'].isChecked()
        self._config.brightness.require_als = self._widgets['brightness_require_als'].isChecked()
        self._config.brightness.min_brightness = self._widgets['brightness_min'].value() / 100.0
        self._config.brightness.max_brightness = self._widgets['brightness_max'].value() / 100.0
        self._config.brightness.poll_interval = self._widgets['brightness_poll_interval'].value()
        self._config.brightness.smoothing_alpha = self._widgets['brightness_smoothing_alpha'].value()
        self._config.brightness.hysteresis_lux = self._widgets['brightness_hysteresis'].value()
        self._config.brightness.lux_at_min = self._widgets['brightness_lux_min'].value()
        self._config.brightness.lux_at_max = self._widgets['brightness_lux_max'].value()
        self._config.brightness.restore_on_exit = self._widgets['brightness_restore_on_exit'].isChecked()

        # Emit signal with config dict
        self.settings_changed.emit(self._config_to_dict())
        self.close()

    def _config_to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            'camera': asdict(self._config.camera),
            'hand_tracker': asdict(self._config.hand_tracker),
            'cursor': {
                'screen_width': self._config.cursor.screen_width,
                'screen_height': self._config.cursor.screen_height,
                'camera_width': self._config.cursor.camera_width,
                'camera_height': self._config.cursor.camera_height,
                'dead_zone_radius': self._config.cursor.dead_zone_radius,
                'sensitivity': self._config.cursor.sensitivity,
                'acceleration': self._config.cursor.acceleration,
                'smoothing': self._config.cursor.smoothing.value,
                'ema_alpha': self._config.cursor.ema_alpha,
                'invert_x': self._config.cursor.invert_x,
                'invert_y': self._config.cursor.invert_y,
                'use_index_tip': self._config.cursor.use_index_tip,
            },
            'gestures': asdict(self._config.gestures),
            'virtual_mouse': asdict(self._config.virtual_mouse),
            'target_fps': self._config.target_fps,
            'emergency_stop_corner': self._config.emergency_stop_corner,
            'brightness': {
                'enabled': self._config.brightness.enabled,
                'min_brightness': self._config.brightness.min_brightness,
                'max_brightness': self._config.brightness.max_brightness,
                'poll_interval': self._config.brightness.poll_interval,
                'smoothing_alpha': self._config.brightness.smoothing_alpha,
                'hysteresis_lux': self._config.brightness.hysteresis_lux,
                'lux_at_min': self._config.brightness.lux_at_min,
                'lux_at_max': self._config.brightness.lux_at_max,
                'preferred_backlight_path': self._config.brightness.preferred_backlight_path,
                'preferred_als_path': self._config.brightness.preferred_als_path,
                'restore_on_exit': self._config.brightness.restore_on_exit,
                'require_als': self._config.brightness.require_als,
            },
        }


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Air Mouse")
        self.setMinimumSize(1000, 700)

        # Air mouse controller
        self.controller: Optional[AirMouseController] = None
        self.config = AirMouseConfig()

        # Settings dialog
        self.settings_dialog: Optional[SettingsDialog] = None

        # System tray
        self.tray_icon: Optional[QSystemTrayIcon] = None

        # Update timer for GUI - stats only, frames come via callback
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_stats_gui)
        self.update_timer.start(33)  # ~30 FPS GUI updates

        self._setup_ui()
        self._setup_tray()
        self._setup_controller()

    def _setup_ui(self):
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)

        # Main layout
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Left side: Camera preview
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.preview = CameraPreviewWidget()
        left_layout.addWidget(self.preview)

        # Preview options
        preview_opts = QHBoxLayout()
        self.show_landmarks_cb = QCheckBox("Landmarks")
        self.show_landmarks_cb.setChecked(True)
        self.show_landmarks_cb.toggled.connect(self.preview.set_show_landmarks)
        preview_opts.addWidget(self.show_landmarks_cb)

        self.show_connections_cb = QCheckBox("Connections")
        self.show_connections_cb.setChecked(True)
        self.show_connections_cb.toggled.connect(self.preview.set_show_connections)
        preview_opts.addWidget(self.show_connections_cb)

        self.mirror_cb = QCheckBox("Mirror")
        self.mirror_cb.setChecked(True)
        self.mirror_cb.toggled.connect(self.preview.set_mirror)
        preview_opts.addWidget(self.mirror_cb)

        preview_opts.addStretch()
        left_layout.addLayout(preview_opts)

        main_layout.addWidget(left_widget, 3)

        # Right side: Controls and stats
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(15)

        # Control buttons
        self.controls = ControlWidget()
        self.controls.start_clicked.connect(self._on_start)
        self.controls.stop_clicked.connect(self._on_stop)
        self.controls.pause_clicked.connect(self._on_pause)
        self.controls.settings_clicked.connect(self._show_settings)
        right_layout.addWidget(self.controls)

        # Status
        self.status = StatusWidget()
        right_layout.addWidget(self.status)

        # Stats
        self.stats = StatsWidget()
        right_layout.addWidget(self.stats)

        main_layout.addWidget(right_widget, 1)

        # Status bar
        self.statusBar().showMessage("Ready")

    def _setup_tray(self):
        """Setup system tray icon."""
        if not QSystemTrayIcon.isSystemTrayAvailable():
            logger.warning("System tray not available")
            return

        self.tray_icon = QSystemTrayIcon(self)
        # Use standard icon
        self.tray_icon.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))

        # Tray menu
        tray_menu = QMenu()

        show_action = QAction("Show", self)
        show_action.triggered.connect(self.show)
        tray_menu.addAction(show_action)

        self.tray_start_action = QAction("Start", self)
        self.tray_start_action.triggered.connect(self._on_start)
        tray_menu.addAction(self.tray_start_action)

        self.tray_pause_action = QAction("Pause", self)
        self.tray_pause_action.triggered.connect(self._on_pause)
        tray_menu.addAction(self.tray_pause_action)

        self.tray_stop_action = QAction("Stop", self)
        self.tray_stop_action.triggered.connect(self._on_stop)
        tray_menu.addAction(self.tray_stop_action)

        tray_menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(QApplication.quit)
        tray_menu.addAction(quit_action)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self._on_tray_activated)
        self.tray_icon.show()

    def _setup_controller(self):
        """Create and configure air mouse controller."""
        def status_callback(status, data):
            logger.info(f"Status: {status}, {data}")

        def error_callback(msg):
            QMessageBox.critical(self, "Error", msg)

        def brightness_callback(state):
            """Handle brightness state updates from controller."""
            # Update UI on GUI thread
            QTimer.singleShot(0, lambda: self._update_brightness_display(state))

        self.controller = AirMouseController(self.config, status_callback)
        self.controller.on_hand_detected = self._on_hand_detected
        self.controller.on_gesture = self._on_gesture
        self.controller.on_error = error_callback
        self.controller.on_stats_update = self._on_stats_update
        self.controller.on_frame_processed = self._on_frame_processed
        self.controller.on_brightness_update = brightness_callback

    @Slot()
    def _on_start(self):
        """Start air mouse."""
        if self.controller:
            if self.controller.start():
                self.controls.set_state(AirMouseState.RUNNING)
                self.status.update_state(AirMouseState.RUNNING, "Tracking active")
                self.statusBar().showMessage("Air Mouse running")
                self._update_tray_actions(AirMouseState.RUNNING)

    @Slot()
    def _on_stop(self):
        """Stop air mouse."""
        if self.controller:
            self.controller.stop()
            self.controls.set_state(AirMouseState.STOPPED)
            self.status.update_state(AirMouseState.STOPPED, "Stopped")
            self.status.update_hand_info(None)
            self.statusBar().showMessage("Stopped")
            self._update_tray_actions(AirMouseState.STOPPED)

    @Slot()
    def _on_pause(self):
        """Pause/resume air mouse."""
        if self.controller:
            state = self.controller.get_state()
            if state == AirMouseState.RUNNING:
                self.controller.pause()
                self.controls.set_state(AirMouseState.PAUSED)
                self.status.update_state(AirMouseState.PAUSED, "Paused - make fist to resume")
                self._update_tray_actions(AirMouseState.PAUSED)
            elif state == AirMouseState.PAUSED:
                self.controller.resume()
                self.controls.set_state(AirMouseState.RUNNING)
                self.status.update_state(AirMouseState.RUNNING, "Tracking active")
                self._update_tray_actions(AirMouseState.RUNNING)

    def _on_hand_detected(self, hand: Hand):
        """Handle hand detection callback."""
        # Update preview with hand data
        # This is called from worker thread, use QTimer.singleShot for thread safety
        QTimer.singleShot(0, lambda: self._update_preview_hand(hand))

    def _update_preview_hand(self, hand: Hand):
        """Update preview widget with hand data (GUI thread)."""
        self.status.update_hand_info(hand)

    def _on_gesture(self, event: GestureEvent):
        """Handle gesture event."""
        logger.debug(f"Gesture: {event.gesture_type.name}")

    def _on_stats_update(self, stats: PerformanceStats):
        """Handle stats update."""
        QTimer.singleShot(0, lambda: self.stats.update_stats(stats))

    def _on_frame_processed(self, frame: np.ndarray, hands: List[Hand]):
        """Handle processed frame from controller (worker thread)."""
        # Use QTimer.singleShot to update GUI from worker thread
        QTimer.singleShot(0, lambda: self.preview.update_frame(frame, hands))

    def _update_brightness_display(self, state):
        """Update brightness display in settings dialog (GUI thread)."""
        if self.settings_dialog and hasattr(self.settings_dialog, '_widgets'):
            widgets = self.settings_dialog._widgets
            if 'brightness_current_lux' in widgets:
                if state.current_lux is not None:
                    widgets['brightness_current_lux'].setText(f"{state.current_lux:.0f} lux")
                else:
                    widgets['brightness_current_lux'].setText("-- lux")

            if 'brightness_current_brightness' in widgets:
                if state.current_brightness is not None:
                    widgets['brightness_current_brightness'].setText(f"{state.current_brightness:.0%}")
                else:
                    widgets['brightness_current_brightness'].setText("--%")

            if 'brightness_als_status' in widgets:
                if state.current_lux is not None:
                    widgets['brightness_als_status'].setText("Active")
                    widgets['brightness_als_status'].setStyleSheet("color: #4CAF50; font-family: monospace;")
                else:
                    widgets['brightness_als_status'].setText("No sensor")
                    widgets['brightness_als_status'].setStyleSheet("color: #F44336; font-family: monospace;")

            if 'brightness_backlight_status' in widgets:
                if state.current_brightness is not None:
                    widgets['brightness_backlight_status'].setText("Active")
                    widgets['brightness_backlight_status'].setStyleSheet("color: #4CAF50; font-family: monospace;")
                else:
                    widgets['brightness_backlight_status'].setText("No backlight")
                    widgets['brightness_backlight_status'].setStyleSheet("color: #F44336; font-family: monospace;")

    def _update_stats_gui(self):
        """Periodic GUI update for stats only."""
        # Frame updates now come via on_frame_processed callback
        pass

    def _show_settings(self):
        """Show settings dialog."""
        if self.settings_dialog is None:
            self.settings_dialog = SettingsDialog(self.config, self)
            self.settings_dialog.settings_changed.connect(self._on_settings_changed)

        self.settings_dialog.show()
        self.settings_dialog.raise_()
        self.settings_dialog.activateWindow()

    def _on_settings_changed(self, config_dict: dict):
        """Handle settings change."""
        logger.info("Settings changed, restart required for some changes")
        self.statusBar().showMessage("Settings updated (restart required for some changes)")

    def _update_tray_actions(self, state: AirMouseState):
        """Update tray menu actions based on state."""
        if not self.tray_icon:
            return

        if state == AirMouseState.RUNNING:
            self.tray_start_action.setEnabled(False)
            self.tray_pause_action.setEnabled(True)
            self.tray_pause_action.setText("Pause")
            self.tray_stop_action.setEnabled(True)
        elif state == AirMouseState.PAUSED:
            self.tray_start_action.setEnabled(False)
            self.tray_pause_action.setEnabled(True)
            self.tray_pause_action.setText("Resume")
            self.tray_stop_action.setEnabled(True)
        else:
            self.tray_start_action.setEnabled(True)
            self.tray_pause_action.setEnabled(False)
            self.tray_pause_action.setText("Pause")
            self.tray_stop_action.setEnabled(False)

    def _on_tray_activated(self, reason):
        """Handle tray icon activation."""
        if reason == QSystemTrayIcon.DoubleClick:
            self.show()
            self.raise_()
            self.activateWindow()

    def closeEvent(self, event: QCloseEvent):
        """Handle window close - minimize to tray."""
        if self.tray_icon and self.tray_icon.isVisible():
            self.hide()
            event.ignore()
            self.tray_icon.showMessage(
                "Air Mouse",
                "Application minimized to tray. Double-click to restore.",
                QSystemTrayIcon.Information,
                2000
            )
        else:
            self._on_stop()
            event.accept()

    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key.Key_P:
            if self.controller:
                enabled = self.controller.toggle_debug_overlay()
                self.statusBar().showMessage(f"Debug overlay: {'ON' if enabled else 'OFF'}")
                logger.info(f"Debug overlay toggled: {'ON' if enabled else 'OFF'}")
        super().keyPressEvent(event)

    def resizeEvent(self, event: QResizeEvent):
        """Handle window resize."""
        super().resizeEvent(event)


def run_gui():
    """Run the GUI application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Air Mouse")
    app.setOrganizationName("AirMouse")
    app.setQuitOnLastWindowClosed(False)  # Keep running in tray

    # Set dark theme
    app.setStyle("Fusion")
    palette = app.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(palette)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    sys.exit(run_gui())