import pytest
from airmouse.debug.diagnostics import DiagnosticsCollector, DiagnosticSnapshot


def test_diagnostics_collector_offline():
    collector = DiagnosticsCollector(controller=None)
    snapshot = collector.collect_all()

    assert isinstance(snapshot, DiagnosticSnapshot)
    assert isinstance(snapshot.timestamp, float)
    assert "python_version" in snapshot.system
    assert "uinput_available" in snapshot.system
    assert "video_devices" in snapshot.camera or "active" in snapshot.camera
    assert "tracking_mode" in snapshot.tracking
    assert "backend" in snapshot.input
    assert isinstance(snapshot.to_dict(), dict)
