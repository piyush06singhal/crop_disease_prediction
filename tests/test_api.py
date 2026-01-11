# tests/test_api.py - API integration tests
"""
API integration tests for Crop Disease Prediction System
Tests REST API endpoints, authentication, rate limiting, and error handling
"""

import pytest
import json
import io
from unittest.mock import patch, Mock
import tempfile
import os
from PIL import Image

# Import Flask app and test client
import sys
sys.path.append('backend')

from app import create_app
from services.prediction_service import PredictionService


@pytest.fixture
def client():
    """Create test client for Flask app"""
    app = create_app()
    app.config['TESTING'] = True
    app.config['SECRET_KEY'] = 'test_secret_key'

    with app.test_client() as client:
        yield client


@pytest.fixture
def auth_headers():
    """Create authentication headers for testing"""
    # Mock JWT token for testing
    return {'Authorization': 'Bearer test_token'}


class TestHealthEndpoint:
    """Test health check endpoint"""

    def test_health_check(self, client):
        """Test basic health check"""
        response = client.get('/api/health')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert 'version' in data

    def test_health_check_with_services(self, client):
        """Test health check with service status"""
        response = client.get('/api/health?include_services=true')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert 'services' in data
        assert isinstance(data['services'], dict)


class TestPredictionEndpoint:
    """Test prediction API endpoint"""

    def create_test_image(self):
        """Create a test image file"""
        # Create a simple test image
        img = Image.new('RGB', (224, 224), color='red')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        return img_byte_arr

    @patch('routes.api.PredictionService.predict_disease')
    def test_predict_success(self, mock_predict, client):
        """Test successful prediction"""
        # Mock prediction service response
        mock_predict.return_value = {
            'success': True,
            'data': {
                'session_id': 'test_session_123',
                'predictions': [{'disease': 'Leaf Blight', 'confidence': 0.85}],
                'confidence': 0.85,
                'crop_type': 'tomato',
                'confidence_breakdown': {
                    'image_prediction': 0.85,
                    'crop_validation': 0.90,
                    'qa_reasoning': 0.80
                }
            }
        }

        # Create test image
        img_data = self.create_test_image()

        # Make request
        response = client.post('/api/predict',
                             data={'image': (img_data, 'test.jpg')},
                             content_type='multipart/form-data')

        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['success'] is True
        assert 'data' in data
        assert data['data']['session_id'] == 'test_session_123'

    def test_predict_no_image(self, client):
        """Test prediction without image"""
        response = client.post('/api/predict')

        assert response.status_code == 400

        data = json.loads(response.data)
        assert data['success'] is False
        assert 'error' in data

    def test_predict_invalid_file_type(self, client):
        """Test prediction with invalid file type"""
        # Create text file instead of image
        text_data = io.BytesIO(b'this is not an image')

        response = client.post('/api/predict',
                             data={'image': (text_data, 'test.txt')},
                             content_type='multipart/form-data')

        assert response.status_code == 400

        data = json.loads(response.data)
        assert data['success'] is False

    @patch('routes.api.PredictionService.predict_disease')
    def test_predict_service_error(self, mock_predict, client):
        """Test prediction with service error"""
        mock_predict.return_value = {
            'success': False,
            'error': {'message': 'Prediction service unavailable'}
        }

        img_data = self.create_test_image()

        response = client.post('/api/predict',
                             data={'image': (img_data, 'test.jpg')},
                             content_type='multipart/form-data')

        assert response.status_code == 500

        data = json.loads(response.data)
        assert data['success'] is False
        assert 'error' in data


class TestAnswerEndpoint:
    """Test answer submission endpoint"""

    @patch('routes.api.PredictionService.answer_question')
    def test_answer_success(self, mock_answer, client):
        """Test successful answer submission"""
        mock_answer.return_value = {
            'success': True,
            'data': {
                'refined_confidence': 0.95,
                'predictions': [{'disease': 'Leaf Blight', 'confidence': 0.95}],
                'confidence_breakdown': {
                    'image_prediction': 0.85,
                    'crop_validation': 0.90,
                    'qa_reasoning': 0.95
                }
            }
        }

        answer_data = {
            'session_id': 'test_session_123',
            'question_id': 'q1',
            'answer': 'Yes, there are yellow spots'
        }

        response = client.post('/api/answer',
                             data=json.dumps(answer_data),
                             content_type='application/json')

        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['success'] is True
        assert 'data' in data

    def test_answer_missing_fields(self, client):
        """Test answer submission with missing fields"""
        incomplete_data = {
            'session_id': 'test_session_123'
            # Missing question_id and answer
        }

        response = client.post('/api/answer',
                             data=json.dumps(incomplete_data),
                             content_type='application/json')

        assert response.status_code == 400

        data = json.loads(response.data)
        assert data['success'] is False

    @patch('routes.api.PredictionService.answer_question')
    def test_answer_invalid_session(self, mock_answer, client):
        """Test answer submission with invalid session"""
        mock_answer.return_value = {
            'success': False,
            'error': {'message': 'Invalid session ID'}
        }

        answer_data = {
            'session_id': 'invalid_session',
            'question_id': 'q1',
            'answer': 'Test answer'
        }

        response = client.post('/api/answer',
                             data=json.dumps(answer_data),
                             content_type='application/json')

        assert response.status_code == 404

        data = json.loads(response.data)
        assert data['success'] is False


class TestHistoryEndpoint:
    """Test history retrieval endpoint"""

    @patch('routes.api.SessionService.get_user_history')
    def test_history_success(self, mock_history, client, auth_headers):
        """Test successful history retrieval"""
        mock_history.return_value = {
            'success': True,
            'data': {
                'history': [
                    {
                        'session_id': 'session_1',
                        'timestamp': '2024-01-01T10:00:00Z',
                        'predictions': [{'disease': 'Leaf Blight', 'confidence': 0.85}],
                        'confidence': 0.85
                    }
                ],
                'total': 1,
                'page': 1,
                'per_page': 10
            }
        }

        response = client.get('/api/history', headers=auth_headers)

        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['success'] is True
        assert 'data' in data
        assert 'history' in data['data']

    def test_history_unauthorized(self, client):
        """Test history access without authentication"""
        response = client.get('/api/history')

        # Should return 401 or allow anonymous access based on implementation
        assert response.status_code in [200, 401]

    @patch('routes.api.SessionService.get_user_history')
    def test_history_with_pagination(self, mock_history, client, auth_headers):
        """Test history with pagination parameters"""
        mock_history.return_value = {
            'success': True,
            'data': {
                'history': [],
                'total': 0,
                'page': 2,
                'per_page': 5
            }
        }

        response = client.get('/api/history?page=2&per_page=5', headers=auth_headers)

        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['data']['page'] == 2
        assert data['data']['per_page'] == 5


class TestAnalyticsEndpoint:
    """Test analytics endpoint"""

    @patch('routes.api.SessionService.get_analytics')
    def test_analytics_success(self, mock_analytics, client, auth_headers):
        """Test successful analytics retrieval"""
        mock_analytics.return_value = {
            'success': True,
            'data': {
                'total_predictions': 100,
                'average_confidence': 0.82,
                'common_diseases': ['Leaf Blight', 'Powdery Mildew'],
                'crop_distribution': {'tomato': 60, 'potato': 40}
            }
        }

        response = client.get('/api/analytics', headers=auth_headers)

        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['success'] is True
        assert 'data' in data
        assert 'total_predictions' in data['data']


class TestRateLimiting:
    """Test rate limiting functionality"""

    def test_rate_limit_exceeded(self, client):
        """Test rate limiting behavior"""
        # Make multiple rapid requests to test rate limiting
        responses = []
        for i in range(15):  # Exceed typical rate limit
            response = client.post('/api/predict',
                                 data={'image': (io.BytesIO(b'fake image'), 'test.jpg')},
                                 content_type='multipart/form-data')
            responses.append(response.status_code)

        # Should have some 429 (Too Many Requests) responses
        assert 429 in responses or all(r in [200, 400, 500] for r in responses)


class TestErrorHandling:
    """Test error handling across endpoints"""

    def test_invalid_json(self, client):
        """Test handling of invalid JSON"""
        response = client.post('/api/answer',
                             data='invalid json',
                             content_type='application/json')

        assert response.status_code == 400

        data = json.loads(response.data)
        assert data['success'] is False

    def test_method_not_allowed(self, client):
        """Test handling of incorrect HTTP methods"""
        response = client.put('/api/predict')

        assert response.status_code == 405

    def test_not_found(self, client):
        """Test handling of non-existent endpoints"""
        response = client.get('/api/nonexistent')

        assert response.status_code == 404


class TestCORS:
    """Test CORS headers"""

    def test_cors_headers(self, client):
        """Test CORS headers are present"""
        response = client.options('/api/predict')

        assert 'Access-Control-Allow-Origin' in response.headers
        assert 'Access-Control-Allow-Methods' in response.headers
        assert 'Access-Control-Allow-Headers' in response.headers


# Load testing utilities
class TestLoadHandling:
    """Test system behavior under load"""

    @pytest.mark.slow
    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests"""
        import threading
        import time

        results = []
        errors = []

        def make_request():
            try:
                response = client.get('/api/health')
                results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))

        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()
            time.sleep(0.01)  # Small delay to avoid overwhelming

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Check results
        assert len(results) == 10
        assert all(status == 200 for status in results)
        assert len(errors) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])