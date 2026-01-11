# analytics/test_analytics.py - Test Analytics Dashboard
"""
Test script for analytics dashboard functionality
"""

import sys
import os
import time
import random
import unittest
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.dashboard import AnalyticsDashboard


class TestAnalyticsDashboard(unittest.TestCase):
    """Test cases for analytics dashboard"""

    def setUp(self):
        """Set up test environment"""
        self.test_db = ':memory:'  # Use in-memory database for testing
        self.analytics = AnalyticsDashboard(db_path=self.test_db, start_monitoring=False)

    def tearDown(self):
        """Clean up after tests"""
        self.analytics.stop_monitoring()

    def test_log_prediction_event(self):
        """Test logging prediction events"""
        self.analytics.log_prediction(
            session_id='test_session_1',
            crop_type='tomato',
            disease_predicted='Leaf Blight',
            confidence=0.85,
            processing_time=0.5
        )

        # Verify data was logged
        analytics = self.analytics.get_prediction_analytics()
        self.assertEqual(analytics['total_predictions'], 1)
        self.assertAlmostEqual(analytics['avg_confidence'], 0.85, places=2)
        self.assertAlmostEqual(analytics['avg_processing_time'], 0.5, places=2)

    def test_log_user_behavior(self):
        """Test logging user behavior"""
        self.analytics.log_user_behavior(
            user_id='test_user_1',
            action='page_view',
            page='/',
            duration=30.5,
            device_type='mobile',
            browser='Chrome'
        )

        # Verify data was logged
        behavior = self.analytics.get_user_behavior_analytics()
        self.assertEqual(behavior['total_actions'], 1)
        self.assertAlmostEqual(behavior['avg_session_duration'], 30.5, places=1)

    def test_log_model_performance(self):
        """Test logging model performance"""
        self.analytics.log_model_performance(
            model_name='MobileNetV2',
            accuracy=0.92,
            precision=0.89,
            recall=0.91,
            f1_score=0.90,
            inference_time=0.15
        )

        # Verify data was logged
        performance = self.analytics.get_model_performance_analytics()
        self.assertIn('MobileNetV2', performance['model_summaries'])
        model = performance['model_summaries']['MobileNetV2']
        self.assertAlmostEqual(model['avg_accuracy'], 0.92, places=2)
        self.assertAlmostEqual(model['avg_precision'], 0.89, places=2)
        self.assertAlmostEqual(model['avg_inference_time'], 0.15, places=2)

    def test_log_error_event(self):
        """Test logging error events"""
        self.analytics.log_error(
            error_type='ValueError',
            error_message='Invalid input data',
            endpoint='/predict',
            stack_trace='Traceback...'
        )

        # Verify data was logged
        errors = self.analytics.get_error_analytics()
        self.assertEqual(errors['total_errors'], 1)
        self.assertIn('ValueError', errors['error_type_distribution'])

    def test_system_metrics_collection(self):
        """Test system metrics collection"""
        # Wait a moment for metrics to be collected
        time.sleep(1)

        metrics = self.analytics.get_system_metrics()
        self.assertIsInstance(metrics, dict)

        # Check that current metrics exist
        if metrics.get('current'):
            current = metrics['current']
            self.assertIsInstance(current.get('cpu_percent', 0), (int, float))
            self.assertIsInstance(current.get('memory_percent', 0), (int, float))
            self.assertIsInstance(current.get('disk_usage', 0), (int, float))

    def test_bulk_data_generation(self):
        """Test analytics with bulk data"""
        # Generate sample prediction data
        crops = ['tomato', 'potato', 'corn', 'wheat', 'rice']
        diseases = ['Leaf Blight', 'Bacterial Spot', 'Fusarium Wilt', 'Powdery Mildew', 'Healthy']

        for i in range(100):
            self.analytics.log_prediction(
                session_id=f'session_{i}',
                crop_type=random.choice(crops),
                disease_predicted=random.choice(diseases),
                confidence=random.uniform(0.5, 0.95),
                processing_time=random.uniform(0.1, 1.0)
            )

        # Generate sample user behavior data
        actions = ['page_view', 'button_click', 'form_submit', 'download']
        pages = ['/', '/predict', '/analytics', '/about']
        devices = ['mobile', 'desktop', 'tablet']
        browsers = ['Chrome', 'Firefox', 'Safari', 'Edge']

        for i in range(200):
            self.analytics.log_user_behavior(
                user_id=f'user_{i % 20}',  # 20 unique users
                action=random.choice(actions),
                page=random.choice(pages),
                duration=random.uniform(5, 300),
                device_type=random.choice(devices),
                browser=random.choice(browsers)
            )

        # Verify bulk data analytics
        predictions = self.analytics.get_prediction_analytics()
        behavior = self.analytics.get_user_behavior_analytics()

        self.assertEqual(predictions['total_predictions'], 100)
        self.assertEqual(behavior['total_actions'], 200)

        # Check distributions
        self.assertGreater(len(predictions['crop_distribution']), 0)
        self.assertGreater(len(predictions['disease_distribution']), 0)
        self.assertGreater(len(behavior['device_distribution']), 0)

    def test_performance_report(self):
        """Test comprehensive performance report generation"""
        # Add some test data
        self.analytics.log_prediction(
            session_id='test_session',
            crop_type='tomato',
            disease_predicted='Leaf Blight',
            confidence=0.85,
            processing_time=0.5
        )

        self.analytics.log_model_performance(
            model_name='TestModel',
            accuracy=0.90,
            precision=0.88,
            recall=0.92,
            f1_score=0.90,
            inference_time=0.20
        )

        # Generate report
        report = self.analytics.generate_performance_report()

        # Verify report structure
        self.assertIn('system_metrics', report)
        self.assertIn('prediction_analytics', report)
        self.assertIn('user_behavior', report)
        self.assertIn('model_performance', report)
        self.assertIn('error_analytics', report)
        self.assertIn('generated_at', report)

        # Verify data in report
        self.assertEqual(report['prediction_analytics']['total_predictions'], 1)
        self.assertIn('TestModel', report['model_performance']['model_summaries'])

    def test_data_retention(self):
        """Test data retention and cleanup"""
        # Add old data (more than 7 days ago)
        old_timestamp = time.time() - (8 * 24 * 3600)  # 8 days ago

        # Manually insert old data (this would normally be done by cleanup job)
        # In a real implementation, you'd have a cleanup method

        # For now, just verify that recent data is retained
        self.analytics.log_prediction(
            session_id='recent_session',
            crop_type='tomato',
            disease_predicted='Healthy',
            confidence=0.95,
            processing_time=0.3
        )

        predictions = self.analytics.get_prediction_analytics(days=1)
        self.assertEqual(predictions['total_predictions'], 1)


def run_performance_test():
    """Run performance test with large dataset"""
    print("Running analytics performance test...")

    analytics = AnalyticsDashboard(db_path=':memory:', start_monitoring=False)

    # Generate large dataset
    print("Generating test data...")
    start_time = time.time()

    for i in range(1000):
        analytics.log_prediction(
            session_id=f'perf_session_{i}',
            crop_type='tomato',
            disease_predicted='Leaf Blight',
            confidence=random.uniform(0.7, 0.95),
            processing_time=random.uniform(0.2, 0.8)
        )

        if i % 100 == 0:
            print(f"Generated {i} prediction records...")

    data_gen_time = time.time() - start_time
    print(".2f")

    # Test analytics queries
    print("Testing analytics queries...")
    query_start = time.time()

    predictions = analytics.get_prediction_analytics()
    behavior = analytics.get_user_behavior_analytics()
    models = analytics.get_model_performance_analytics()
    errors = analytics.get_error_analytics()
    report = analytics.generate_performance_report()

    query_time = time.time() - query_start
    print(".2f")

    # Verify results
    assert predictions['total_predictions'] == 1000
    assert predictions['avg_confidence'] > 0
    assert predictions['avg_processing_time'] > 0

    print("Performance test completed successfully!")
    print(f"Data generation: {data_gen_time:.2f}s")
    print(f"Query execution: {query_time:.2f}s")
    print(f"Total predictions: {predictions['total_predictions']}")
    print(".2f")
    print(".2f")

    analytics.stop_monitoring()


if __name__ == '__main__':
    # Run unit tests
    print("Running analytics dashboard tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)

    # Run performance test
    print("\n" + "="*50)
    run_performance_test()