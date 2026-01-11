# analytics/__init__.py - Analytics Package Initialization
"""
Crop Disease Prediction System - Analytics Module

This package provides comprehensive analytics and monitoring capabilities
for the Crop Disease Prediction System, including real-time metrics,
user behavior analytics, and model performance tracking.
"""

from .dashboard import (
    AnalyticsDashboard,
    analytics_bp,
    log_prediction_event,
    log_user_action,
    log_model_metrics,
    log_error_event
)

__version__ = "1.0.0"
__author__ = "Crop Disease Prediction System Team"
__description__ = "Advanced analytics dashboard for monitoring system performance"

__all__ = [
    'AnalyticsDashboard',
    'analytics_bp',
    'log_prediction_event',
    'log_user_action',
    'log_model_metrics',
    'log_error_event'
]