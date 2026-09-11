"""
System Tray Integration for Air Mouse

Provides cross-desktop system tray support:
- X11: Uses AppIndicator3 (libappindicator) or legacy XEmbed
- Wayland: Uses StatusNotifierItem via DBus (KDE, GNOME with extension, COSMIC)
- Fallback: Uses plain PySide6 QSystemTrayIcon (works on both but limited on Wayland)

Automatically detects the best available method.
"""

import os
import logging
import subprocess
from enum import Enum
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtWidgets import (
    QSystemTrayIcon, QMenu, QApplication, QWidget, QStyle
)
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor, QFont, QAction
from PySide6.QtCore import QObject, Signal, QTimer, Qt

logger = logging.getLogger(__name__)


class TrayBackend(Enum):
    """System tray backend types."""
    APPI_INDICATOR = "appindicator"  # AppIndicator3 (Ubuntu, KDE, XFCE)
    STATUS_NOTIFIER = "status_notifier"  # KDE StatusNotifierItem (Wayland)
    QSYSTEM_TRAY = "qsystem_tray"  # Qt built-in (fallback)
    NONE = "none"


@dataclass
class TrayMenuItem:
    """Menu item configuration."""
    text: str
    callback: Callable
    checkable: bool = False
    checked: bool = False
    enabled: bool = True
    icon: Optional[str] = None  # Icon name or path
    separator_before: bool = False
    separator_after: bool = False


class SystemTrayManager(QObject):
    """
    Cross-platform system tray manager for Linux desktop.

    Supports multiple backends with automatic detection:
    1. AppIndicator3 (X11, works on Ubuntu, KDE, XFCE)
    2. StatusNotifierItem (Wayland, KDE, GNOME with extension, COSMIC)
    3. QSystemTrayIcon (fallback, limited on Wayland)
    """

    # Signals
    tray_activated = Signal(str)  # "left_click", "right_click", "middle_click"
    menu_action_triggered = Signal(str)  # action_id

    def __init__(self, app_id: str = "airmouse", parent: Optional[QObject] = None):
        super().__init__(parent)
        self._app_id = app_id
        self._backend: TrayBackend = TrayBackend.NONE
        self._tray_icon: Optional[QSystemTrayIcon] = None
        self._app_indicator = None
        self._menu: Optional[QMenu] = None
        self._menu_items: Dict[str, QAction] = {}
        self._icon_cache: Dict[str, QIcon] = {}
        self._default_icon: Optional[QIcon] = None
        self._is_visible = False
        self._callbacks: Dict[str, Callable] = {}

        # Try to detect and initialize best backend
        self._detect_backend()

    def _detect_backend(self) -> TrayBackend:
        """Detect the best available system tray backend."""
        # Check for AppIndicator3 (most widely supported on X11)
        if self._check_appindicator():
            self._backend = TrayBackend.APPI_INDICATOR
            logger.info("Using AppIndicator3 backend")
            return self._backend

        # Check for KDE StatusNotifierItem (Wayland native)
        if self._check_status_notifier():
            self._backend = TrayBackend.STATUS_NOTIFIER
            logger.info("Using StatusNotifierItem backend")
            return self._backend

        # Fallback to QSystemTrayIcon
        if QSystemTrayIcon.isSystemTrayAvailable():
            self._backend = TrayBackend.QSYSTEM_TRAY
            logger.info("Using QSystemTrayIcon backend")
            return self._backend

        logger.warning("No system tray backend available")
        self._backend = TrayBackend.NONE
        return self._backend

    def _check_appindicator(self) -> bool:
        """Check if AppIndicator3 is available."""
        # Check if we're on X11 or XWayland
        if os.environ.get('WAYLAND_DISPLAY') and not os.environ.get('DISPLAY'):
            # Pure Wayland - AppIndicator may not work
            return False

        try:
            import gi
            gi.require_version('AyatanaAppIndicator3', '1.0')
            from gi.repository import AyatanaAppIndicator3 as AppIndicator3
            return True
        except Exception:
            try:
                import gi
                gi.require_version('AppIndicator3', '0.1')
                from gi.repository import AppIndicator3
                return True
            except Exception:
                pass
        return False

    def _check_status_notifier(self) -> bool:
        """Check if KDE StatusNotifierItem is available via DBus."""
        # Check for KDE or other Wayland compositors supporting StatusNotifierItem
        try:
            import dbus
            bus = dbus.SessionBus()
            # Check if StatusNotifierWatcher is registered
            try:
                bus.get_object('org.kde.StatusNotifierWatcher', '/StatusNotifierWatcher')
                return True
            except dbus.DBusException:
                pass
            # Also check for org.freedesktop.StatusNotifierWatcher
            try:
                bus.get_object('org.freedesktop.StatusNotifierWatcher', '/StatusNotifierWatcher')
                return True
            except dbus.DBusException:
                pass
        except Exception:
            pass
        return False

    def _create_default_icon(self) -> QIcon:
        """Create a default icon programmatically."""
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw a simple mouse cursor icon
        painter.setBrush(QColor(0, 120, 215))
        painter.setPen(QColor(0, 80, 180))
        painter.drawEllipse(4, 4, 24, 24)

        # Draw cursor arrow
        painter.setBrush(QColor(255, 255, 255))
        painter.setPen(Qt.PenStyle.NoPen)
        points = [
            (16, 8),   # tip
            (12, 16),  # left
            (14, 16),  # inner left
            (14, 20),  # down
            (18, 20),  # down
            (18, 16),  # inner right
            (20, 16),  # right
        ]
        from PySide6.QtGui import QPolygon
        from PySide6.QtCore import QPoint
        polygon = QPolygon([QPoint(x, y) for x, y in points])
        painter.drawPolygon(polygon)

        painter.end()
        return QIcon(pixmap)

    def _get_icon(self, name: str) -> QIcon:
        """Get icon by name, creating default if needed."""
        if name in self._icon_cache:
            return self._icon_cache[name]

        # Try to load from system theme
        icon = QIcon.fromTheme(name)
        if not icon.isNull():
            self._icon_cache[name] = icon
            return icon

        # Fallback to default
        if self._default_icon is None:
            self._default_icon = self._create_default_icon()

        self._icon_cache[name] = self._default_icon
        return self._default_icon

    def initialize(self) -> bool:
        """Initialize the system tray."""
        if self._backend == TrayBackend.NONE:
            logger.error("Cannot initialize: no backend available")
            return False

        if self._backend == TrayBackend.APPI_INDICATOR:
            return self._init_appindicator()
        elif self._backend == TrayBackend.STATUS_NOTIFIER:
            return self._init_status_notifier()
        else:
            return self._init_qsystem_tray()

    def _init_appindicator(self) -> bool:
        """Initialize AppIndicator3 backend."""
        try:
            import gi
            try:
                gi.require_version('AyatanaAppIndicator3', '1.0')
                from gi.repository import AyatanaAppIndicator3 as AppIndicator3
            except Exception:
                gi.require_version('AppIndicator3', '0.1')
                from gi.repository import AppIndicator3

            # Create indicator
            self._app_indicator = AppIndicator3.Indicator.new(
                self._app_id,
                "input-mouse",  # Icon name
                AppIndicator3.IndicatorCategory.APPLICATION_STATUS
            )
            self._app_indicator.set_status(AppIndicator3.IndicatorStatus.ACTIVE)
            self._app_indicator.set_title("Air Mouse")

            # Create menu
            self._menu = self._create_gtk_menu()
            self._app_indicator.set_menu(self._menu)

            self._is_visible = True
            logger.info("AppIndicator3 initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize AppIndicator3: {e}")
            return False

    def _init_status_notifier(self) -> bool:
        """Initialize KDE StatusNotifierItem via DBus."""
        try:
            import dbus
            import dbus.service
            from dbus.mainloop.glib import DBusGMainLoop

            # This is a simplified version - full implementation would need
            # proper DBus service registration
            logger.warning("StatusNotifierItem DBus backend not fully implemented, falling back to QSystemTray")
            return self._init_qsystem_tray()

        except Exception as e:
            logger.error(f"Failed to initialize StatusNotifierItem: {e}")
            return self._init_qsystem_tray()

    def _init_qsystem_tray(self) -> bool:
        """Initialize Qt's built-in system tray."""
        if not QSystemTrayIcon.isSystemTrayAvailable():
            logger.error("QSystemTrayIcon not available")
            return False

        self._tray_icon = QSystemTrayIcon()
        self._tray_icon.setToolTip("Air Mouse")

        # Set default icon
        self._default_icon = self._create_default_icon()
        self._tray_icon.setIcon(self._default_icon)

        # Create context menu
        self._menu = QMenu()
        self._tray_icon.setContextMenu(self._menu)

        # Connect signals
        self._tray_icon.activated.connect(self._on_tray_activated)

        self._is_visible = True
        logger.info("QSystemTrayIcon initialized successfully")
        return True

    def _create_gtk_menu(self):
        """Create GTK menu for AppIndicator3."""
        try:
            import gi
            gi.require_version('Gtk', '3.0')
            from gi.repository import Gtk

            menu = Gtk.Menu()

            # We'll need to populate this dynamically
            # For now, create a basic structure
            return menu
        except Exception as e:
            logger.error(f"Failed to create GTK menu: {e}")
            return None

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason):
        """Handle tray icon activation."""
        reason_map = {
            QSystemTrayIcon.ActivationReason.Trigger: "left_click",
            QSystemTrayIcon.ActivationReason.Context: "right_click",
            QSystemTrayIcon.ActivationReason.MiddleClick: "middle_click",
            QSystemTrayIcon.ActivationReason.DoubleClick: "double_click",
        }
        reason_str = reason_map.get(reason, "unknown")
        self.tray_activated.emit(reason_str)

    def set_icon(self, icon_name: str):
        """Set the tray icon by theme name."""
        icon = self._get_icon(icon_name)

        if self._backend == TrayBackend.APPI_INDICATOR and self._app_indicator:
            # AppIndicator uses icon names directly
            self._app_indicator.set_icon_full(icon_name, "Air Mouse")
        elif self._tray_icon:
            self._tray_icon.setIcon(icon)

    def set_tooltip(self, text: str):
        """Set the tray tooltip."""
        if self._backend == TrayBackend.APPI_INDICATOR and self._app_indicator:
            self._app_indicator.set_title(text)
        elif self._tray_icon:
            self._tray_icon.setToolTip(text)

    def set_status(self, status: str):
        """Set status text (shown in tooltip or menu)."""
        self.set_tooltip(f"Air Mouse - {status}")

    def create_menu(self, items: List[TrayMenuItem]):
        """Create or update the context menu."""
        if self._backend == TrayBackend.APPI_INDICATOR:
            self._create_appindicator_menu(items)
        elif self._menu:
            self._create_qt_menu(items)

    def _create_appindicator_menu(self, items: List[TrayMenuItem]):
        """Create AppIndicator3 menu."""
        try:
            import gi
            gi.require_version('Gtk', '3.0')
            from gi.repository import Gtk

            if self._menu:
                # Clear existing menu
                for child in self._menu.get_children():
                    self._menu.remove(child)
            else:
                self._menu = Gtk.Menu()

            for item in items:
                if item.separator_before:
                    sep = Gtk.SeparatorMenuItem()
                    self._menu.append(sep)

                if item.checkable:
                    menu_item = Gtk.CheckMenuItem.new_with_label(item.text)
                    menu_item.set_active(item.checked)
                else:
                    menu_item = Gtk.MenuItem.new_with_label(item.text)

                menu_item.set_sensitive(item.enabled)

                if item.icon:
                    image = Gtk.Image.new_from_icon_name(item.icon, Gtk.IconSize.MENU)
                    menu_item.set_image(image)
                    menu_item.set_always_show_image(True)

                # Connect callback
                action_id = item.text.lower().replace(' ', '_').replace('&', '')
                self._callbacks[action_id] = item.callback
                menu_item.connect('activate', self._on_gtk_menu_activate, action_id)

                self._menu.append(menu_item)

                if item.separator_after:
                    sep = Gtk.SeparatorMenuItem()
                    self._menu.append(sep)

            self._menu.show_all()
            if self._app_indicator:
                self._app_indicator.set_menu(self._menu)

        except Exception as e:
            logger.error(f"Failed to create AppIndicator menu: {e}")

    def _on_gtk_menu_activate(self, widget, action_id: str):
        """Handle GTK menu activation."""
        if action_id in self._callbacks:
            try:
                self._callbacks[action_id]()
                self.menu_action_triggered.emit(action_id)
            except Exception as e:
                logger.error(f"Menu callback error: {e}")

    def _create_qt_menu(self, items: List[TrayMenuItem]):
        """Create Qt menu."""
        self._menu.clear()
        self._menu_items.clear()

        for item in items:
            if item.separator_before:
                self._menu.addSeparator()

            action = QAction(item.text, self._menu)
            action.setCheckable(item.checkable)
            action.setChecked(item.checked)
            action.setEnabled(item.enabled)

            if item.icon:
                action.setIcon(self._get_icon(item.icon))

            action_id = item.text.lower().replace(' ', '_').replace('&', '')
            self._menu_items[action_id] = action
            self._callbacks[action_id] = item.callback

            action.triggered.connect(
                lambda checked, aid=action_id: self._on_qt_action_triggered(aid)
            )

            self._menu.addAction(action)

            if item.separator_after:
                self._menu.addSeparator()

    def _on_qt_action_triggered(self, action_id: str):
        """Handle Qt menu action."""
        if action_id in self._callbacks:
            try:
                self._callbacks[action_id]()
                self.menu_action_triggered.emit(action_id)
            except Exception as e:
                logger.error(f"Menu callback error: {e}")

    def show(self):
        """Show the tray icon."""
        if self._backend == TrayBackend.APPI_INDICATOR and self._app_indicator:
            import gi
            gi.require_version('AyatanaAppIndicator3', '1.0')
            from gi.repository import AyatanaAppIndicator3 as AppIndicator3
            self._app_indicator.set_status(AppIndicator3.IndicatorStatus.ACTIVE)
            self._is_visible = True
        elif self._tray_icon:
            self._tray_icon.show()
            self._is_visible = True

    def hide(self):
        """Hide the tray icon."""
        if self._backend == TrayBackend.APPI_INDICATOR and self._app_indicator:
            import gi
            gi.require_version('AyatanaAppIndicator3', '1.0')
            from gi.repository import AyatanaAppIndicator3 as AppIndicator3
            self._app_indicator.set_status(AppIndicator3.IndicatorStatus.PASSIVE)
            self._is_visible = False
        elif self._tray_icon:
            self._tray_icon.hide()
            self._is_visible = False

    def is_visible(self) -> bool:
        """Check if tray is visible."""
        return self._is_visible

    def show_message(self, title: str, message: str, icon: str = "info", timeout: int = 5000):
        """Show a notification message."""
        if self._backend == TrayBackend.APPI_INDICATOR and self._app_indicator:
            # AppIndicator3 doesn't have built-in notifications
            # Would need libnotify
            pass
        elif self._tray_icon:
            msg_icon = QSystemTrayIcon.MessageIcon.Information
            if icon == "warning":
                msg_icon = QSystemTrayIcon.MessageIcon.Warning
            elif icon == "critical":
                msg_icon = QSystemTrayIcon.MessageIcon.Critical
            self._tray_icon.showMessage(title, message, msg_icon, timeout)

    def get_backend(self) -> TrayBackend:
        """Get the current backend."""
        return self._backend

    def cleanup(self):
        """Clean up resources."""
        self.hide()

        if self._backend == TrayBackend.APPI_INDICATOR and self._app_indicator:
            import gi
            gi.require_version('AyatanaAppIndicator3', '1.0')
            from gi.repository import AyatanaAppIndicator3 as AppIndicator3
            self._app_indicator.set_status(AppIndicator3.IndicatorStatus.PASSIVE)
            self._app_indicator = None

        if self._tray_icon:
            self._tray_icon.deleteLater()
            self._tray_icon = None

        self._menu = None
        self._menu_items.clear()
        self._callbacks.clear()
        self._is_visible = False

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def create_airmouse_tray_menu(
    on_toggle: Callable,
    on_settings: Callable,
    on_calibrate: Callable,
    on_diagnose: Callable,
    on_quit: Callable,
    is_active: bool = False,
    is_paused: bool = False
) -> List[TrayMenuItem]:
    """Create standard Air Mouse tray menu."""
    return [
        TrayMenuItem(
            text="&Enable Air Mouse" if not is_active else "&Disable Air Mouse",
            callback=on_toggle,
            checkable=True,
            checked=is_active,
            icon="input-mouse"
        ),
        TrayMenuItem(
            text="&Pause Tracking" if not is_paused else "&Resume Tracking",
            callback=lambda: None,  # Will be set by caller
            checkable=True,
            checked=is_paused,
            icon="media-playback-pause"
        ),
        TrayMenuItem(
            text="&Calibrate...",
            callback=on_calibrate,
            icon="preferences-desktop-peripherals"
        ),
        TrayMenuItem(
            text="&Settings...",
            callback=on_settings,
            icon="preferences-system"
        ),
        TrayMenuItem(
            text="&Diagnostics...",
            callback=on_diagnose,
            icon="utilities-system-monitor"
        ),
        TrayMenuItem(
            text="&Quit",
            callback=on_quit,
            icon="application-exit",
            separator_before=True
        ),
    ]


if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)  # Keep running when tray only

    tray = SystemTrayManager("airmouse-test")

    def on_toggle():
        print("Toggle clicked")

    def on_quit():
        print("Quit clicked")
        app.quit()

    if tray.initialize():
        tray.create_menu(create_airmouse_tray_menu(
            on_toggle=on_toggle,
            on_settings=lambda: print("Settings"),
            on_calibrate=lambda: print("Calibrate"),
            on_diagnose=lambda: print("Diagnose"),
            on_quit=on_quit,
            is_active=True
        ))
        tray.show()
        print(f"Tray initialized with backend: {tray.get_backend().value}")
        print("Right-click tray icon to see menu")
        sys.exit(app.exec())
    else:
        print("Failed to initialize tray")
        sys.exit(1)