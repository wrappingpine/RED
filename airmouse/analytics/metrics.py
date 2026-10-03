"""
Real-time analytics and metrics visualization for Air Mouse.

Implements P4: Real-time Analytics with:
- Metrics collection and aggregation
- Time-series data storage
- Dashboard components for performance monitoring
- Statistics reporting and export
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Deque
from collections import deque
from datetime import datetime
import json
from pathlib import Path


@dataclass
class TimeSeriesPoint:
    """Single data point in a time series."""
    timestamp: float
    value: float
    label: Optional[str] = None


@dataclass
class AggregationConfig:
    """Configuration for metric aggregation."""
    window_size: int = 100  # Number of points in sliding window
    aggregation_interval_ms: int = 1000  # Aggregation interval in milliseconds
    

@dataclass  
class MetricsSummary:
    """Aggregated metrics summary."""
    fps: float = 0.0
    avg_frame_time_ms: float = 0.0
    min_frame_time_ms: float = 1000.0
    max_frame_time_ms: float = 0.0
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    gpu_percent: Optional[float] = None
    dropped_frames: int = 0
    total_frames: int = 0
    tracking_losses: int = 0
    false_gestures: int = 0
    false_clicks: int = 0
    timestamp: float = 0.0


class MetricsCollector:
    """
    Collects and aggregates real-time metrics from the Air Mouse system.
    
    Features:
    - Time-series data collection
    - Sliding window aggregation
    - Export to JSON/CSV formats
    - Real-time statistics computation
    """
    
    def __init__(self, config: Optional[AggregationConfig] = None):
        self.config = config or AggregationConfig()
        
        # Time series storage using deque for efficient sliding windows
        self._fps_history: Deque[TimeSeriesPoint] = deque(maxlen=self.config.window_size)
        self._frame_time_history: Deque[TimeSeriesPoint] = deque(maxlen=self.config.window_size)
        self._cpu_history: Deque[TimeSeriesPoint] = deque(maxlen=self.config.window_size)
        self._memory_history: Deque[TimeSeriesPoint] = deque(maxlen=self.config.window_size)
        
        # Aggregated metrics
        self._current_summary = MetricsSummary()
        
        # Counters
        self._total_frames = 0
        self._dropped_frames = 0
        self._tracking_losses = 0
        self._false_gestures = 0
        self._false_clicks = 0
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Callbacks
        self._update_callbacks: List[Callable[[MetricsSummary], None]] = []
        
    def record_latency(self, latency_ms: float, stage: str = "total"):
        """Record latency for a specific pipeline stage.
        
        Implements spec §44: Latency tracking with end-to-end and stage breakdown.
        This enables fine-grained performance analysis and helps identify
        bottlenecks in the hand-tracking pipeline.
        """
        current_time = time.time()
        
        with self._lock:
            # Store latency per stage for detailed analysis
            if not hasattr(self, '_stage_latencies'):
                self._stage_latencies = {}
            if stage not in self._stage_latencies:
                self._stage_latencies[stage] = deque(maxlen=self.config.window_size)
            
            self._stage_latencies[stage].append(TimeSeriesPoint(current_time, latency_ms))
            
            # Update overall summary with key metrics
            self._update_summary()
            self._notify_callbacks()

    def get_latency_stats(self) -> Dict[str, Any]:
        """Get detailed latency statistics for all pipeline stages.
        
        Returns:
            Dictionary with mean, max, and p95 latency for each stage,
            plus overall end-to-end latency metrics.
        """
        stats = {
            'total': self._get_stage_stats('total'),
            'stages': {},
            'dropped_frames': self._dropped_frames,
            'tracking_losses': self._tracking_losses,
            'false_gestures': self._false_gestures,
            'false_clicks': self._false_clicks,
            'total_frames': self._total_frames
        }
        
        if hasattr(self, '_stage_latencies'):
            for stage, points in self._stage_latencies.items():
                stats['stages'][stage] = self._get_stage_stats(stage)
        
        return stats

    def _get_stage_stats(self, stage: str) -> Dict[str, float]:
        """Get statistics for a specific stage."""
        if not hasattr(self, '_stage_latencies') or stage not in self._stage_latencies:
            return {'mean': 0.0, 'max': 0.0, 'p95': 0.0, 'count': 0}
        
        points = self._stage_latencies[stage]
        if not points:
            return {'mean': 0.0, 'max': 0.0, 'p95': 0.0, 'count': 0}
        
        values = [p.value for p in points]
        return {
            'mean': sum(values) / len(values),
            'max': max(values),
            'p95': sorted(values)[-max(1, int(len(values) * 0.95))],
            'count': len(values)
        }
            
    def record_system_metrics(self, cpu_percent: float, memory_mb: float):
        """Record system resource metrics."""
        current_time = time.time()
        
        with self._lock:
            self._cpu_history.append(TimeSeriesPoint(current_time, cpu_percent))
            self._memory_history.append(TimeSeriesPoint(current_time, memory_mb))
            self._update_summary()
            self._notify_callbacks()
            
    def record_error(self, error_type: str):
        """Record an error event."""
        with self._lock:
            if 'tracking_loss' in error_type:
                self._tracking_losses += 1
            elif 'false_gesture' in error_type:
                self._false_gestures += 1
            elif 'false_click' in error_type:
                self._false_clicks += 1
                
    def record_dropped_frame(self):
        """Record a dropped frame."""
        with self._lock:
            self._dropped_frames += 1
            
    def _update_summary(self):
        """Update the aggregated metrics summary."""
        summary = self._current_summary
        
        # FPS
        if self._fps_history:
            summary.fps = sum(p.value for p in self._fps_history) / len(self._fps_history)
        else:
            summary.fps = 0.0
            
        # Frame times
        if self._frame_time_history:
            frame_times = [p.value for p in self._frame_time_history]
            summary.avg_frame_time_ms = sum(frame_times) / len(frame_times)
            summary.min_frame_time_ms = min(frame_times)
            summary.max_frame_time_ms = max(frame_times)
        else:
            summary.avg_frame_time_ms = 0.0
            summary.min_frame_time_ms = 0.0
            summary.max_frame_time_ms = 0.0
            
        # CPU
        if self._cpu_history:
            summary.cpu_percent = sum(p.value for p in self._cpu_history) / len(self._cpu_history)
        else:
            summary.cpu_percent = 0.0
            
        # Memory
        if self._memory_history:
            summary.memory_mb = sum(p.value for p in self._memory_history) / len(self._memory_history)
        else:
            summary.memory_mb = 0.0
            
        # Counters
        summary.dropped_frames = self._dropped_frames
        summary.total_frames = self._total_frames
        summary.tracking_losses = self._tracking_losses
        summary.false_gestures = self._false_gestures
        summary.false_clicks = self._false_clicks
        
        # Timestamp
        summary.timestamp = time.time()
        
    def _notify_callbacks(self):
        """Notify registered callbacks of metric updates."""
        for callback in self._update_callbacks:
            try:
                callback(self._current_summary)
            except Exception:
                pass  # Don't let callback errors propagate
                
    def add_callback(self, callback: Callable[[MetricsSummary], None]):
        """Register a callback to be notified on metric updates."""
        self._update_callbacks.append(callback)
        
    def get_summary(self) -> MetricsSummary:
        """Get the current aggregated metrics summary."""
        with self._lock:
            return self._current_summary
            
    def get_fps_history(self) -> List[TimeSeriesPoint]:
        """Get FPS time series."""
        with self._lock:
            return list(self._fps_history)
            
    def get_frame_time_history(self) -> List[TimeSeriesPoint]:
        """Get frame time time series."""
        with self._lock:
            return list(self._frame_time_history)
            
    def get_cpu_history(self) -> List[TimeSeriesPoint]:
        """Get CPU usage time series."""
        with self._lock:
            return list(self._cpu_history)
            
    def get_memory_history(self) -> List[TimeSeriesPoint]:
        """Get memory usage time series."""
        with self._lock:
            return list(self._memory_history)
            
    def export_time_series(self, output_path: str, format: str = "json"):
        """
        Export time series data to a file.
        
        Args:
            output_path: Path to export file
            format: Export format ("json" or "csv")
        """
        output_file = Path(output_path)
        
        if format.lower() == "json":
            data = {
                'fps': [
                    {'timestamp': p.timestamp, 'value': p.value}
                    for p in self.get_fps_history()
                ],
                'frame_time_ms': [
                    {'timestamp': p.timestamp, 'value': p.value}
                    for p in self.get_frame_time_history()
                ],
                'cpu_percent': [
                    {'timestamp': p.timestamp, 'value': p.value}
                    for p in self.get_cpu_history()
                ],
                'memory_mb': [
                    {'timestamp': p.timestamp, 'value': p.value}
                    for p in self.get_memory_history()
                ]
            }
            
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        elif format.lower() == "csv":
            import csv
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'fps', 'frame_time_ms', 'cpu_percent', 'memory_mb'])
                
                fps_data = {p.timestamp: p.value for p in self.get_fps_history()}
                ft_data = {p.timestamp: p.value for p in self.get_frame_time_history()}
                cpu_data = {p.timestamp: p.value for p in self.get_cpu_history()}
                mem_data = {p.timestamp: p.value for p in self.get_memory_history()}
                
                all_timestamps = sorted(set(fps_data.keys()) | set(ft_data.keys()) | 
                                       set(cpu_data.keys()) | set(mem_data.keys()))
                
                for ts in all_timestamps:
                    writer.writerow([
                        ts,
                        fps_data.get(ts, 0),
                        ft_data.get(ts, 0),
                        cpu_data.get(ts, 0),
                        mem_data.get(ts, 0)
                    ])
                    
    def reset_counters(self):
        """Reset all counters (keeps time series history)."""
        with self._lock:
            self._total_frames = 0
            self._dropped_frames = 0
            self._tracking_losses = 0
            self._false_gestures = 0
            self._false_clicks = 0
            
    def clear_history(self):
        """Clear all time series history."""
        with self._lock:
            self._fps_history.clear()
            self._frame_time_history.clear()
            self._cpu_history.clear()
            self._memory_history.clear()
            self._total_frames = 0
            self._dropped_frames = 0


class PerformanceDashboard:
    """
    Performance dashboard for real-time metrics visualization.
    
    Provides formatted text output and structured data for display
    in GUI or CLI interfaces.
    """
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        self._width = 80
        
    def render_text_dashboard(self) -> str:
        """Render a text-based dashboard."""
        summary = self.collector.get_summary()
        
        lines = []
        lines.append("=" * self._width)
        lines.append("AIR MOUSE PERFORMANCE DASHBOARD".center(self._width))
        lines.append("=" * self._width)
        lines.append("")
        
        # Frame metrics
        lines.append("FRAME METRICS")
        lines.append("-" * 40)
        lines.append(f"  FPS: {summary.fps:6.1f}")
        lines.append(f"  Avg Frame Time: {summary.avg_frame_time_ms:6.2f} ms")
        lines.append(f"  Frame Time Range: {summary.min_frame_time_ms:5.2f} - {summary.max_frame_time_ms:5.2f} ms")
        lines.append(f"  Target: 30 FPS (33.3 ms)")
        status = "✓ PASS" if summary.fps >= 25 else "✗ FAIL"
        lines.append(f"  Status: {status}")
        lines.append("")
        
        # System metrics
        lines.append("SYSTEM RESOURCES")
        lines.append("-" * 40)
        lines.append(f"  CPU Usage:  {summary.cpu_percent:5.1f}%")
        lines.append(f"  Memory:     {summary.memory_mb:6.1f} MB")
        lines.append(f"  Target:     <50% CPU, <200 MB RAM")
        status = "✓ PASS" if summary.cpu_percent < 50 and summary.memory_mb < 200 else "✗ FAIL"
        lines.append(f"  Status:     {status}")
        lines.append("")
        
        # Stability metrics
        lines.append("STABILITY METRICS")
        lines.append("-" * 40)
        lines.append(f"  Total Frames:     {summary.total_frames:10d}")
        lines.append(f"  Dropped Frames:   {summary.dropped_frames:10d} ({100*summary.dropped_frames/max(1,summary.total_frames):.1f}%)")
        lines.append(f"  Tracking Losses:  {summary.tracking_losses:10d}")
        lines.append(f"  False Gestures:   {summary.false_gestures:10d}")
        lines.append(f"  False Clicks:     {summary.false_clicks:10d}")
        lines.append("")
        
        # Health score
        health_score = self._compute_health_score(summary)
        lines.append(f"HEALTH SCORE: {health_score:.1f}/100")
        lines.append("=" * self._width)
        
        return "\n".join(lines)
        
    def _compute_health_score(self, summary: MetricsSummary) -> float:
        """Compute overall health score (0-100)."""
        score = 100.0
        
        # FPS penalty
        if summary.fps < 20:
            score -= 30
        elif summary.fps < 25:
            score -= 15
        elif summary.fps < 30:
            score -= 5
            
        # Frame time penalty
        if summary.avg_frame_time_ms > 50:
            score -= 20
        elif summary.avg_frame_time_ms > 40:
            score -= 10
        elif summary.avg_frame_time_ms > 33.3:
            score -= 5
            
        # CPU penalty
        if summary.cpu_percent > 75:
            score -= 20
        elif summary.cpu_percent > 50:
            score -= 10
        elif summary.cpu_percent > 40:
            score -= 5
            
        # Memory penalty
        if summary.memory_mb > 300:
            score -= 20
        elif summary.memory_mb > 200:
            score -= 10
        elif summary.memory_mb > 150:
            score -= 5
            
        # Error penalty
        if summary.dropped_frames > summary.total_frames * 0.1:
            score -= 15
        if summary.tracking_losses > 10:
            score -= 10
        if summary.false_gestures > 5:
            score -= 10
        if summary.false_clicks > 3:
            score -= 15
            
        return max(0.0, min(100.0, score))
        
    def render_json_report(self) -> Dict[str, Any]:
        """Render a JSON-formatted performance report."""
        summary = self.collector.get_summary()
        
        return {
            'timestamp': datetime.fromtimestamp(summary.timestamp).isoformat(),
            'metrics': {
                'fps': summary.fps,
                'frame_time_ms': {
                    'average': summary.avg_frame_time_ms,
                    'min': summary.min_frame_time_ms,
                    'max': summary.max_frame_time_ms
                },
                'cpu_percent': summary.cpu_percent,
                'memory_mb': summary.memory_mb
            },
            'counters': {
                'total_frames': summary.total_frames,
                'dropped_frames': summary.dropped_frames,
                'tracking_losses': summary.tracking_losses,
                'false_gestures': summary.false_gestures,
                'false_clicks': summary.false_clicks
            },
            'health_score': self._compute_health_score(summary),
            'status': 'pass' if self._compute_health_score(summary) > 70 else 'warning'
        }


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None

def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector

def setup_metrics_collection(config: Optional[AggregationConfig] = None):
    """Setup global metrics collection with custom configuration."""
    global _global_collector
    _global_collector = MetricsCollector(config)

def record_frame_metrics(frame_time_ms: float, fps: float):
    """Record frame metrics using the global collector."""
    get_metrics_collector().record_frame(frame_time_ms, fps)

def record_system_metrics(cpu_percent: float, memory_mb: float):
    """Record system metrics using the global collector."""
    get_metrics_collector().record_system_metrics(cpu_percent, memory_mb)