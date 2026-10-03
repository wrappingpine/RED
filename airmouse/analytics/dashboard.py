"""
Real-time performance dashboard for Air Mouse.

Implements P4: Real-time Analytics with:
- Real-time performance visualization
- Metrics collection and display
- Export functionality for analysis
"""

import time
from typing import Optional, Dict, Any
from airmouse.debug.performance_monitor import PerformanceMonitor
from airmouse.analytics.metrics import (
    MetricsCollector, AggregationConfig, PerformanceDashboard
)
class PerformanceDashboardUI:
    """
    Performance dashboard UI for Air Mouse.
    
    Provides text-based dashboard output and JSON reports
    for real-time performance monitoring.
    """
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None,
                 performance_monitor: Optional[PerformanceMonitor] = None):
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.performance_monitor = performance_monitor
        self.dashboard = PerformanceDashboard(self.metrics_collector)
        
    def render_text_dashboard(self) -> str:
        """Render the text-based performance dashboard."""
        return self.dashboard.render_text_dashboard()
        
    def render_json_report(self) -> Dict[str, Any]:
        """Render the JSON-formatted performance report."""
        return self.dashboard.render_json_report()
        
    def export_metrics(self, output_path: str, format: str = "json"):
        """Export metrics to a file."""
        self.metrics_collector.export_time_series(output_path, format)
        
    def update_system_metrics(self):
        """Update system metrics from performance monitor if available."""
        if self.performance_monitor:
            stats = self.performance_monitor.get_stats()
            if stats:
                self.metrics_collector.record_system_metrics(
                    stats.get('cpu_percent', 0),
                    stats.get('memory_mb', 0)
                )
                
    def update_performance_stats(self, fps: float, frame_time_ms: float):
        """Update performance statistics."""
        self.metrics_collector.record_frame(frame_time_ms, fps)
        
    def log_error(self, error: Exception, component: str = "general"):
        """Log an error using the metrics collector."""
        error_type = type(error).__name__
        self.metrics_collector.record_error(error_type)
        
    def log_dropped_frame(self):
        """Log a dropped frame."""
        self.metrics_collector.record_dropped_frame()
        
    def log_tracking_loss(self):
        """Log a tracking loss."""
        self.metrics_collector.record_error('tracking_loss')
        
    def log_false_gesture(self):
        """Log a false gesture."""
        self.metrics_collector.record_error('false_gesture')
        
    def log_false_click(self):
        """Log a false click."""
        self.metrics_collector.record_error('false_click')
        
    def get_dashboard_status(self) -> Dict[str, Any]:
        """Get dashboard status and health metrics."""
        summary = self.metrics_collector.get_summary()
        dashboard_text = self.render_text_dashboard()
        
        return {
            'health_score': (100 - (1000 / max(summary.fps, 1))),
            'total_frames': summary.total_frames,
            'dropped_frames': summary.dropped_frames,
            'tracking_losses': summary.tracking_losses,
            'false_gestures': summary.false_gestures,
            'false_clicks': summary.false_clicks,
            'cpu_percent': summary.cpu_percent,
            'memory_mb': summary.memory_mb,
            'fps': summary.fps,
            'frame_time_ms': summary.avg_frame_time_ms,
            'dashboard_text': dashboard_text,
            'last_update': summary.timestamp,
            'status': 'healthy' if self._is_healthy(summary) else 'degraded'
        }
        
    def _is_healthy(self, summary) -> bool:
        """Check if the system is healthy based on metrics."""
        health_score = self.dashboard._compute_health_score(summary)
        return health_score >= 70
        
    def reset_statistics(self):
        """Reset all statistics."""
        self.metrics_collector.reset_counters()
        
    def clear_history(self):
        """Clear all historical metrics."""
        self.metrics_collector.clear_history()
        
    def add_callback(self, callback):
        """Add a callback for metrics updates."""
        self.metrics_collector.add_callback(callback)
class RealTimePerformanceMonitor:
    """
    Real-time performance monitoring for Air Mouse.
    
    Integrates with the existing performance monitor and metrics collector
    to provide comprehensive performance analytics.
    """
    
    def __init__(self, enable_text_dashboard: bool = True):
        self.metrics_collector = MetricsCollector()
        self.dashboard = PerformanceDashboard(self.metrics_collector)
        self.ui = PerformanceDashboardUI(self.metrics_collector)
        self.enable_text_dashboard = enable_text_dashboard
        self.last_dashboard_update = time.time()
        self.dashboard_update_interval = 5000  # milliseconds
        
    def start_monitoring(self):
        """Start real-time performance monitoring."""
        pass  # Would be started by the main AirMouseController
        
    def stop_monitoring(self):
        """Stop performance monitoring."""
        pass  # Would be stopped by the main AirMouseController
        
    def update_metrics(self, fps: float = 0.0, frame_time_ms: float = 0.0,
                      cpu_percent: float = 0.0, memory_mb: float = 0.0,
                      dropped_frame: bool = False, tracking_loss: bool = False,
                      false_gesture: bool = False, false_click: bool = False):
        """
        Update all performance metrics in real-time.
        
        Args:
            fps: Current FPS
            frame_time_ms: Average frame time in milliseconds
            cpu_percent: CPU usage percentage
            memory_mb: Memory usage in MB
            dropped_frame: Whether a frame was dropped
            tracking_loss: Whether a tracking loss occurred
            false_gesture: Whether a false gesture was detected
            false_click: Whether a false click was detected
        """
        # Update metrics
        self.metrics_collector.record_frame(frame_time_ms, fps)
        self.metrics_collector.record_system_metrics(cpu_percent, memory_mb)
        
        if dropped_frame:
            self.metrics_collector.record_dropped_frame()
        if tracking_loss:
            self.metrics_collector.record_error('tracking_loss')
        if false_gesture:
            self.metrics_collector.record_error('false_gesture')
        if false_click:
            self.metrics_collector.record_error('false_click')
            
    def get_dashboard_text(self) -> str:
        """Get the text-based dashboard."""
        return self.ui.render_text_dashboard()
        
    def get_dashboard_json(self) -> Dict[str, Any]:
        """Get the JSON-based dashboard report."""
        return self.ui.render_json_report()
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get the system status from the dashboard."""
        return self.ui.get_dashboard_status()
        
    def export_metrics(self, output_path: str, format: str = "json"):
        """Export metrics to a file."""
        self.ui.export_metrics(output_path, format)
        
    def reset(self):
        """Reset all metrics and statistics."""
        self.metrics_collector.reset_counters()
        
    def clear_history(self):
        """Clear all historical data."""
        self.metrics_collector.clear_history()
        
    def get_health_score(self) -> float:
        """Get the current health score."""
        summary = self.metrics_collector.get_summary()
        return self.dashboard._compute_health_score(summary)
        
    def is_healthy(self) -> bool:
        """Check if the system is healthy."""
        return self.get_health_score() >= 70


# Global real-time performance monitor instance
_global_monitor: Optional[RealTimePerformanceMonitor] = None

def get_performance_monitor() -> RealTimePerformanceMonitor:
    """Get the global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = RealTimePerformanceMonitor()
    return _global_monitor

def setup_performance_monitoring(enable_text_dashboard: bool = True):
    """Setup global performance monitoring."""
    global _global_monitor
    _global_monitor = RealTimePerformanceMonitor(enable_text_dashboard)

def update_performance_metrics(**kwargs):
    """Update performance metrics using the global monitor."""
    get_performance_monitor().update_metrics(**kwargs)

def get_dashboard_text() -> str:
    """Get the dashboard text from the global monitor."""
    return get_performance_monitor().get_dashboard_text()

def get_dashboard_json() -> Dict[str, Any]:
    """Get the dashboard JSON report from the global monitor."""
    return get_performance_monitor().get_dashboard_json()

def get_system_status() -> Dict[str, Any]:
    """Get the system status from the global monitor."""
    return get_performance_monitor().get_system_status()