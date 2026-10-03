"""
Comprehensive error logging system for Air Mouse.

Implements P3.3: Enhanced error logging and recovery with:
- Structured error information with severity levels
- Automatic error categorization and classification
- Recovery action suggestions
- Error history and statistics tracking
- Export capabilities for analysis
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import traceback
import json
import os
from pathlib import Path

class ErrorSeverity(Enum):
    """Error severity levels for error categorization."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ErrorInfo:
    """Information about an error event."""
    timestamp: str
    severity: ErrorSeverity
    error_type: str
    message: str
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    component: str = "general"
    actionable: bool = False
    recovery_action: Optional[str] = None

class ErrorLogger:
    def __init__(self, log_file: str = "airmouse.log", max_file_size_mb: int = 10):
        self.log_file = log_file
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        
        # Setup logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.error_history: List[ErrorInfo] = []
        self.error_counts: Dict[str, int] = {}
        
    def log_error(self, error: Exception, severity: ErrorSeverity = ErrorSeverity.ERROR,
                   error_type: Optional[str] = None, component: str = "general",
                   context: Optional[Dict[str, Any]] = None, actionable: bool = False,
                   recovery_action: Optional[str] = None) -> ErrorInfo:
        """
        Log an error with structured information.
        
        Args:
            error: The exception or error object
            severity: Severity level of the error
            error_type: Type/category of error (defaults to exception type name)
            component: Which component generated the error
            context: Additional context about the error
            actionable: Whether this error can be automatically recovered from
            recovery_action: Suggested recovery action
            
        Returns:
            ErrorInfo object with error details
        """
        timestamp = datetime.now().isoformat()
        
        if error_type is None:
            error_type = type(error).__name__ if not isinstance(error, str) else "Error"
            
        message = str(error)
        stack_trace = traceback.format_exc() if severity in [ErrorSeverity.ERROR, ErrorSeverity.CRITICAL] else None
        
        # Handle severity parameter that might be a string
        if isinstance(severity, str):
            try:
                severity = ErrorSeverity(severity.lower())
            except ValueError:
                severity = ErrorSeverity.ERROR
        
        error_info = ErrorInfo(
            timestamp=timestamp,
            severity=severity,
            error_type=error_type,
            message=message,
            stack_trace=stack_trace,
            context=context or {},
            component=component,
            actionable=actionable,
            recovery_action=recovery_action
        )
        
        # Add to history
        self.error_history.append(error_info)
        
        # Update error counts
        error_key = f"{component}.{error_type}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Log based on severity
        log_message = f"[{severity.value.upper()}] {component}: {error_type}: {message}"
        if context:
            log_message += f" Context: {json.dumps(context)}"
            
        if severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif severity == ErrorSeverity.ERROR:
            self.logger.error(log_message)
        elif severity == ErrorSeverity.WARNING:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
            
        return error_info
    
    def log_info(self, message: str, component: str = "general", context: Optional[Dict[str, Any]] = None):
        """Log an informational message."""
        return self.log_error(
            Exception(message),
            severity=ErrorSeverity.INFO,
            component=component,
            context=context
        )
    
    def log_warning(self, warning: str, component: str = "general", 
                    context: Optional[Dict[str, Any]] = None):
        """Log a warning message."""
        return self.log_error(
            Exception(warning),
            severity=ErrorSeverity.WARNING,
            component=component,
            context=context
        )
    
    def get_recent_errors(self, since: Optional[datetime] = None, 
                         component: Optional[str] = None,
                         severity: Optional[ErrorSeverity] = None) -> List[ErrorInfo]:
        """
        Get recent errors, optionally filtered by time, component, and severity.
        
        Args:
            since: Only return errors after this datetime
            component: Filter by component name
            severity: Filter by error severity
            
        Returns:
            List of ErrorInfo objects
        """
        errors = self.error_history
        
        if since:
            errors = [e for e in errors if datetime.fromisoformat(e.timestamp) >= since]
            
        if component:
            errors = [e for e in errors if e.component == component]
            
        if severity:
            errors = [e for e in errors if e.severity == severity]
            
        return errors
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get a summary of errors from the last N hours.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Dictionary with error statistics
        """
        since = datetime.now().replace(hour=datetime.now().hour - hours)
        recent_errors = self.get_recent_errors(since=since)
        
        summary = {
            'total_errors': len(recent_errors),
            'errors_by_severity': {sev.value: 0 for sev in ErrorSeverity},
            'errors_by_component': {},
            'errors_by_type': {},
            'actionable_errors': 0,
            'recovery_actions_needed': 0
        }
        
        for error in recent_errors:
            # Count by severity
            summary['errors_by_severity'][error.severity.value] += 1
            
            # Count by component
            summary['errors_by_component'][error.component] = \
                summary['errors_by_component'].get(error.component, 0) + 1
                
            # Count by type
            summary['errors_by_type'][error.error_type] = \
                summary['errors_by_type'].get(error.error_type, 0) + 1
                
            # Actionable errors
            if error.actionable:
                summary['actionable_errors'] += 1
                
            # Recovery actions needed
            if error.recovery_action:
                summary['recovery_actions_needed'] += 1
        
        return summary
    
    def get_error_counts(self) -> Dict[str, int]:
        """Get counts of each error type."""
        return self.error_counts.copy()
    
    def export_errors(self, output_file: str, format: str = "json"):
        """
        Export error history to a file.
        
        Args:
            output_file: Path to output file
            format: Export format ("json", "csv", "txt")
        """
        output_path = Path(output_file)
        
        if format.lower() == "json":
            data = []
            for error in self.error_history:
                data.append({
                    'timestamp': error.timestamp,
                    'severity': error.severity.value,
                    'error_type': error.error_type,
                    'message': error.message,
                    'stack_trace': error.stack_trace,
                    'component': error.component,
                    'context': error.context,
                    'actionable': error.actionable,
                    'recovery_action': error.recovery_action
                })
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        elif format.lower() == "csv":
            import csv
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'severity', 'error_type', 'message',
                    'component', 'context', 'actionable', 'recovery_action'
                ])
                
                for error in self.error_history:
                    writer.writerow([
                        error.timestamp,
                        error.severity.value,
                        error.error_type,
                        error.message,
                        error.component,
                        json.dumps(error.context),
                        error.actionable,
                        error.recovery_action or ''
                    ])
                    
        elif format.lower() == "txt":
            with open(output_path, 'w') as f:
                for error in self.error_history:
                    f.write(f"[{error.timestamp}] {error.severity.value.upper()}: ")
                    f.write(f"{error.component}: {error.error_type}: {error.message}\n")
                    f.write(f"Context: {json.dumps(error.context)}\n")
                    if error.recovery_action:
                        f.write(f"Recovery: {error.recovery_action}\n")
                    f.write("---\n")
    
    def cleanup_old_errors(self, days: int = 30):
        """
        Remove errors older than specified days.
        
        Args:
            days: Number of days to keep
        """
        cutoff_date = datetime.now().replace(day=datetime.now().day - days)
        
        self.error_history = [
            error for error in self.error_history
            if datetime.fromisoformat(error.timestamp) >= cutoff_date
        ]


# Global error logger instance
_global_error_logger: Optional[ErrorLogger] = None

def get_error_logger() -> ErrorLogger:
    """Get the global error logger instance."""
    global _global_error_logger
    if _global_error_logger is None:
        _global_error_logger = ErrorLogger()
    return _global_error_logger

def setup_error_logging(log_file: str = "airmouse.log"):
    """Setup global error logging."""
    global _global_error_logger
    _global_error_logger = ErrorLogger(log_file)

def log_error(error: Exception, **kwargs):
    """Global function to log errors using the global logger."""
    return get_error_logger().log_error(error, **kwargs)

def log_warning(warning: str, **kwargs):
    """Global function to log warnings using the global logger."""
    return get_error_logger().log_warning(warning, **kwargs)

def log_info(info: str, **kwargs):
    """Global function to log info using the global logger."""
    return get_error_logger().log_info(info, **kwargs)