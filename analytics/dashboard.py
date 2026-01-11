# analytics/dashboard.py - Analytics Dashboard for Crop Disease Prediction System
"""
Advanced Analytics Dashboard with Real-time Metrics, User Behavior Analytics, and Model Performance Tracking
"""

import os
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import threading
# import schedule  # Commented out for now
from flask import Flask, render_template, jsonify, request, Blueprint
import psutil
import sqlite3
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64


class AnalyticsDashboard:
    """Advanced analytics dashboard for monitoring system performance"""

    def __init__(self, db_path='../backend/data/analytics.db', start_monitoring=True):
        self.db_path = db_path
        self.metrics = defaultdict(list)
        self.alerts = []
        self.is_monitoring = False
        self.monitoring_thread = None
        self.db_connection = None  # Keep connection open for in-memory databases

        # Initialize database first
        self.init_database()

        # Start background monitoring after database is ready (optional)
        if start_monitoring:
            self.start_monitoring()

    def init_database(self):
        """Initialize analytics database"""
        # Skip directory creation for in-memory databases
        if self.db_path != ':memory:':
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # For in-memory databases, keep connection open
        if self.db_path == ':memory:':
            self.db_connection = sqlite3.connect(self.db_path)
            cursor = self.db_connection.cursor()
        else:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                timestamp REAL,
                cpu_percent REAL,
                memory_percent REAL,
                disk_usage REAL,
                network_connections INTEGER,
                active_requests INTEGER
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prediction_analytics (
                timestamp REAL,
                session_id TEXT,
                user_id TEXT,
                crop_type TEXT,
                disease_predicted TEXT,
                confidence REAL,
                processing_time REAL,
                user_agent TEXT,
                ip_address TEXT,
                feedback_rating INTEGER,
                feedback_comment TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_behavior (
                timestamp REAL,
                user_id TEXT,
                action TEXT,
                page TEXT,
                duration REAL,
                device_type TEXT,
                browser TEXT,
                referrer TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                timestamp REAL,
                model_name TEXT,
                accuracy REAL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                inference_time REAL,
                model_version TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS error_logs (
                timestamp REAL,
                error_type TEXT,
                error_message TEXT,
                stack_trace TEXT,
                user_id TEXT,
                endpoint TEXT,
                request_data TEXT
            )
        ''')

        if self.db_path != ':memory:':
            conn.commit()
            conn.close()
        else:
            self.db_connection.commit()

    def get_db_connection(self):
        """Get database connection (persistent for in-memory, new for file-based)"""
        if self.db_path == ':memory:' and self.db_connection:
            return self.db_connection
        else:
            return sqlite3.connect(self.db_path)

    def start_monitoring(self):
        """Start background monitoring"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()

    def stop_monitoring(self):
        """Stop background monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()

    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.is_monitoring:
            try:
                # Verify database connection and tables exist
                conn = self.get_db_connection()
                try:
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='system_metrics'")
                    if not cursor.fetchone():
                        print("Database tables not found, reinitializing...")
                        self.init_database()
                finally:
                    if self.db_path != ':memory:':
                        conn.close()

                self.collect_system_metrics()
                time.sleep(30)  # Collect metrics every 30 seconds
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(60)

    def collect_system_metrics(self):
        """Collect current system metrics"""
        timestamp = time.time()

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage = disk.percent

        # Network connections
        network_connections = len(psutil.net_connections())

        # Active requests (placeholder - would need to integrate with Flask)
        active_requests = 0

        # Store metrics
        conn = self.get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO system_metrics
                (timestamp, cpu_percent, memory_percent, disk_usage, network_connections, active_requests)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (timestamp, cpu_percent, memory_percent, disk_usage, network_connections, active_requests))
            if self.db_path != ':memory:':
                conn.commit()
        finally:
            if self.db_path != ':memory:':
                conn.close()

    def log_prediction(self, session_id: str, user_id: str = None, crop_type: str = None,
                      disease_predicted: str = None, confidence: float = None,
                      processing_time: float = None, user_agent: str = None,
                      ip_address: str = None):
        """Log prediction analytics"""
        timestamp = time.time()

        conn = self.get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO prediction_analytics
                (timestamp, session_id, user_id, crop_type, disease_predicted, confidence,
                 processing_time, user_agent, ip_address)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, session_id, user_id, crop_type, disease_predicted, confidence,
                  processing_time, user_agent, ip_address))
            if self.db_path != ':memory:':
                conn.commit()
        finally:
            if self.db_path != ':memory:':
                conn.close()

    def log_user_behavior(self, user_id: str, action: str, page: str, duration: float = None,
                         device_type: str = None, browser: str = None, referrer: str = None):
        """Log user behavior analytics"""
        timestamp = time.time()

        conn = self.get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO user_behavior
                (timestamp, user_id, action, page, duration, device_type, browser, referrer)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, user_id, action, page, duration, device_type, browser, referrer))
            if self.db_path != ':memory:':
                conn.commit()
        finally:
            if self.db_path != ':memory:':
                conn.close()

    def log_model_performance(self, model_name: str, accuracy: float, precision: float,
                             recall: float, f1_score: float, inference_time: float,
                             model_version: str = None):
        """Log model performance metrics"""
        timestamp = time.time()

        conn = self.get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO model_performance
                (timestamp, model_name, accuracy, precision, recall, f1_score, inference_time, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, model_name, accuracy, precision, recall, f1_score, inference_time, model_version))
            if self.db_path != ':memory:':
                conn.commit()
        finally:
            if self.db_path != ':memory:':
                conn.close()

    def log_error(self, error_type: str, error_message: str, stack_trace: str = None,
                  user_id: str = None, endpoint: str = None, request_data: str = None):
        """Log error events"""
        timestamp = time.time()

        conn = self.get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO error_logs
                (timestamp, error_type, error_message, stack_trace, user_id, endpoint, request_data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, error_type, error_message, stack_trace, user_id, endpoint, request_data))
            if self.db_path != ':memory:':
                conn.commit()
        finally:
            if self.db_path != ':memory:':
                conn.close()

    def get_system_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get system metrics for the last N hours"""
        cutoff_time = time.time() - (hours * 3600)

        conn = self.get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp, cpu_percent, memory_percent, disk_usage, network_connections, active_requests
                FROM system_metrics
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            ''', (cutoff_time,))

            rows = cursor.fetchall()
        finally:
            if self.db_path != ':memory:':
                conn.close()

        if not rows:
            return {}

        # Convert to time series data
        timestamps = [row[0] for row in rows]
        cpu_data = [row[1] for row in rows]
        memory_data = [row[2] for row in rows]
        disk_data = [row[3] for row in rows]
        network_data = [row[4] for row in rows]
        requests_data = [row[5] for row in rows]

        return {
            'timestamps': timestamps,
            'cpu_percent': cpu_data,
            'memory_percent': memory_data,
            'disk_usage': disk_data,
            'network_connections': network_data,
            'active_requests': requests_data,
            'current': {
                'cpu_percent': cpu_data[0] if cpu_data else 0,
                'memory_percent': memory_data[0] if memory_data else 0,
                'disk_usage': disk_data[0] if disk_data else 0,
                'network_connections': network_data[0] if network_data else 0,
                'active_requests': requests_data[0] if requests_data else 0
            }
        }

    def get_prediction_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get prediction analytics for the last N days"""
        cutoff_time = time.time() - (days * 24 * 3600)

        conn = self.get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp, crop_type, disease_predicted, confidence, processing_time
                FROM prediction_analytics
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            ''', (cutoff_time,))

            rows = cursor.fetchall()
        finally:
            if self.db_path != ':memory:':
                conn.close()

        if not rows:
            return {}

        # Process analytics
        total_predictions = len(rows)
        avg_confidence = np.mean([row[3] for row in rows if row[3] is not None])
        avg_processing_time = np.mean([row[4] for row in rows if row[4] is not None])

        # Crop distribution
        crop_counts = Counter(row[1] for row in rows if row[1])
        crop_distribution = dict(crop_counts.most_common(10))

        # Disease distribution
        disease_counts = Counter(row[2] for row in rows if row[2])
        disease_distribution = dict(disease_counts.most_common(10))

        # Confidence distribution
        confidence_ranges = {
            'high': len([r for r in rows if r[3] and r[3] >= 0.8]),
            'medium': len([r for r in rows if r[3] and 0.6 <= r[3] < 0.8]),
            'low': len([r for r in rows if r[3] and r[3] < 0.6])
        }

        # Daily predictions
        daily_predictions = defaultdict(int)
        for row in rows:
            date = datetime.fromtimestamp(row[0]).date()
            daily_predictions[str(date)] += 1

        return {
            'total_predictions': total_predictions,
            'avg_confidence': avg_confidence,
            'avg_processing_time': avg_processing_time,
            'crop_distribution': crop_distribution,
            'disease_distribution': disease_distribution,
            'confidence_ranges': confidence_ranges,
            'daily_predictions': dict(daily_predictions)
        }

    def get_user_behavior_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get user behavior analytics"""
        cutoff_time = time.time() - (days * 24 * 3600)

        conn = self.get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp, action, page, duration, device_type, browser
                FROM user_behavior
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            ''', (cutoff_time,))

            rows = cursor.fetchall()
        finally:
            if self.db_path != ':memory:':
                conn.close()

        if not rows:
            return {}

        # Process behavior analytics
        total_actions = len(rows)

        # Page views
        page_views = Counter(row[2] for row in rows if row[2])
        top_pages = dict(page_views.most_common(10))

        # Actions
        actions = Counter(row[1] for row in rows if row[1])
        top_actions = dict(actions.most_common(10))

        # Device types
        devices = Counter(row[4] for row in rows if row[4])
        device_distribution = dict(devices)

        # Browsers
        browsers = Counter(row[5] for row in rows if row[5])
        browser_distribution = dict(browsers)

        # Session duration
        durations = [row[3] for row in rows if row[3] and row[3] > 0]
        avg_session_duration = np.mean(durations) if durations else 0

        # Hourly activity
        hourly_activity = defaultdict(int)
        for row in rows:
            hour = datetime.fromtimestamp(row[0]).hour
            hourly_activity[hour] += 1

        return {
            'total_actions': total_actions,
            'top_pages': top_pages,
            'top_actions': top_actions,
            'device_distribution': device_distribution,
            'browser_distribution': browser_distribution,
            'avg_session_duration': avg_session_duration,
            'hourly_activity': dict(hourly_activity)
        }

    def get_model_performance_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get model performance analytics"""
        cutoff_time = time.time() - (days * 24 * 3600)

        conn = self.get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp, model_name, accuracy, precision, recall, f1_score, inference_time
                FROM model_performance
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            ''', (cutoff_time,))

            rows = cursor.fetchall()
        finally:
            if self.db_path != ':memory:':
                conn.close()

        if not rows:
            return {}

        # Process performance analytics
        model_performance = defaultdict(list)

        for row in rows:
            model_name = row[1]
            model_performance[model_name].append({
                'timestamp': row[0],
                'accuracy': row[2],
                'precision': row[3],
                'recall': row[4],
                'f1_score': row[5],
                'inference_time': row[6]
            })

        # Calculate averages for each model
        model_summaries = {}
        for model_name, performances in model_performance.items():
            accuracies = [p['accuracy'] for p in performances if p['accuracy']]
            precisions = [p['precision'] for p in performances if p['precision']]
            recalls = [p['recall'] for p in performances if p['recall']]
            f1_scores = [p['f1_score'] for p in performances if p['f1_score']]
            inference_times = [p['inference_time'] for p in performances if p['inference_time']]

            model_summaries[model_name] = {
                'avg_accuracy': np.mean(accuracies) if accuracies else 0,
                'avg_precision': np.mean(precisions) if precisions else 0,
                'avg_recall': np.mean(recalls) if recalls else 0,
                'avg_f1_score': np.mean(f1_scores) if f1_scores else 0,
                'avg_inference_time': np.mean(inference_times) if inference_times else 0,
                'performance_count': len(performances)
            }

        return {
            'model_summaries': model_summaries,
            'performance_history': dict(model_performance)
        }

    def get_error_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get error analytics"""
        cutoff_time = time.time() - (days * 24 * 3600)

        conn = self.get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp, error_type, error_message, endpoint
                FROM error_logs
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            ''', (cutoff_time,))

            rows = cursor.fetchall()
        finally:
            if self.db_path != ':memory:':
                conn.close()

        if not rows:
            return {}

        # Process error analytics
        total_errors = len(rows)

        # Error types
        error_types = Counter(row[1] for row in rows if row[1])
        error_type_distribution = dict(error_types.most_common(10))

        # Error endpoints
        error_endpoints = Counter(row[3] for row in rows if row[3])
        error_endpoint_distribution = dict(error_endpoints.most_common(10))

        # Daily errors
        daily_errors = defaultdict(int)
        for row in rows:
            date = datetime.fromtimestamp(row[0]).date()
            daily_errors[str(date)] += 1

        return {
            'total_errors': total_errors,
            'error_type_distribution': error_type_distribution,
            'error_endpoint_distribution': error_endpoint_distribution,
            'daily_errors': dict(daily_errors)
        }

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        return {
            'system_metrics': self.get_system_metrics(),
            'prediction_analytics': self.get_prediction_analytics(),
            'user_behavior': self.get_user_behavior_analytics(),
            'model_performance': self.get_model_performance_analytics(),
            'error_analytics': self.get_error_analytics(),
            'generated_at': datetime.now().isoformat()
        }

    def create_chart(self, data: Dict, chart_type: str = 'line', title: str = '') -> str:
        """Create base64 encoded chart image"""
        plt.figure(figsize=(10, 6))

        if chart_type == 'line':
            for key, values in data.items():
                if key != 'timestamps':
                    plt.plot(data.get('timestamps', []), values, label=key)
        elif chart_type == 'bar':
            plt.bar(data.keys(), data.values())
        elif chart_type == 'pie':
            plt.pie(data.values(), labels=data.keys(), autopct='%1.1f%%')

        plt.title(title)
        plt.legend()
        plt.tight_layout()

        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

        return f"data:image/png;base64,{image_base64}"


# Global analytics instance
analytics_dashboard = AnalyticsDashboard()


# Flask Blueprint for analytics dashboard
analytics_bp = Blueprint('analytics', __name__, url_prefix='/analytics')


@analytics_bp.route('/')
def dashboard():
    """Main analytics dashboard page"""
    return render_template('analytics/dashboard.html')


@analytics_bp.route('/api/metrics')
def get_metrics():
    """Get real-time system metrics"""
    return jsonify(analytics_dashboard.get_system_metrics())


@analytics_bp.route('/api/predictions')
def get_predictions():
    """Get prediction analytics"""
    return jsonify(analytics_dashboard.get_prediction_analytics())


@analytics_bp.route('/api/users')
def get_users():
    """Get user behavior analytics"""
    return jsonify(analytics_dashboard.get_user_behavior_analytics())


@analytics_bp.route('/api/models')
def get_models():
    """Get model performance analytics"""
    return jsonify(analytics_dashboard.get_model_performance_analytics())


@analytics_bp.route('/api/errors')
def get_errors():
    """Get error analytics"""
    return jsonify(analytics_dashboard.get_error_analytics())


@analytics_bp.route('/api/report')
def get_report():
    """Get comprehensive performance report"""
    return jsonify(analytics_dashboard.generate_performance_report())


@analytics_bp.route('/api/chart/<chart_type>')
def get_chart(chart_type):
    """Generate and return chart images"""
    if chart_type == 'cpu':
        data = analytics_dashboard.get_system_metrics()
        chart = analytics_dashboard.create_chart(
            {'CPU Usage': data.get('cpu_percent', []), 'timestamps': data.get('timestamps', [])},
            'line', 'CPU Usage Over Time'
        )
    elif chart_type == 'predictions':
        data = analytics_dashboard.get_prediction_analytics()
        chart = analytics_dashboard.create_chart(
            data.get('daily_predictions', {}),
            'bar', 'Daily Predictions'
        )
    else:
        return jsonify({'error': 'Unknown chart type'})

    return jsonify({'chart': chart})


# Analytics logging functions for integration
def log_prediction_event(session_id, crop_type=None, disease=None, confidence=None, processing_time=None):
    """Log prediction event for analytics"""
    analytics_dashboard.log_prediction(
        session_id=session_id,
        crop_type=crop_type,
        disease_predicted=disease,
        confidence=confidence,
        processing_time=processing_time
    )


def log_user_action(user_id, action, page, duration=None, device_type=None, browser=None):
    """Log user action for analytics"""
    analytics_dashboard.log_user_behavior(
        user_id=user_id,
        action=action,
        page=page,
        duration=duration,
        device_type=device_type,
        browser=browser
    )


def log_model_metrics(model_name, accuracy, precision, recall, f1_score, inference_time):
    """Log model performance metrics"""
    analytics_dashboard.log_model_performance(
        model_name=model_name,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        inference_time=inference_time
    )


def log_error_event(error_type, error_message, user_id=None, endpoint=None, stack_trace=None):
    """Log error event for analytics"""
    analytics_dashboard.log_error(
        error_type=error_type,
        error_message=error_message,
        user_id=user_id,
        endpoint=endpoint,
        stack_trace=stack_trace
    )


if __name__ == '__main__':
    # Test the analytics dashboard
    print("Testing Analytics Dashboard...")

    # Log some test data
    analytics_dashboard.log_prediction(
        session_id='test_session_1',
        crop_type='tomato',
        disease_predicted='Leaf Blight',
        confidence=0.85,
        processing_time=0.5
    )

    analytics_dashboard.log_user_behavior(
        user_id='test_user_1',
        action='page_view',
        page='/',
        duration=30.5,
        device_type='mobile',
        browser='Chrome'
    )

    analytics_dashboard.log_model_performance(
        model_name='MobileNetV2',
        accuracy=0.92,
        precision=0.89,
        recall=0.91,
        f1_score=0.90,
        inference_time=0.15
    )

    # Get analytics
    system_metrics = analytics_dashboard.get_system_metrics()
    prediction_analytics = analytics_dashboard.get_prediction_analytics()
    user_behavior = analytics_dashboard.get_user_behavior_analytics()
    model_performance = analytics_dashboard.get_model_performance_analytics()

    print("System Metrics:", system_metrics.get('current', {}))
    print("Prediction Analytics:", prediction_analytics)
    print("User Behavior:", user_behavior)
    print("Model Performance:", model_performance)

    print("Analytics Dashboard test completed!")