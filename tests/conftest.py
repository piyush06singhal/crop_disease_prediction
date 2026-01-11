# tests/conftest.py - Test configuration and fixtures
"""
Shared test configuration and fixtures for all test modules
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test files"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture(scope="session")
def test_data_dir():
    """Get path to test data directory"""
    return Path(__file__).parent / "test_data"


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment variables"""
    # Set test environment
    os.environ['FLASK_ENV'] = 'testing'
    os.environ['SECRET_KEY'] = 'test_secret_key_12345'

    # Disable external API calls in tests
    os.environ['DISABLE_EXTERNAL_APIS'] = 'true'

    yield

    # Cleanup
    test_env_vars = ['FLASK_ENV', 'SECRET_KEY', 'DISABLE_EXTERNAL_APIS']
    for var in test_env_vars:
        if var in os.environ:
            del os.environ[var]


@pytest.fixture
def mock_image():
    """Create a mock image for testing"""
    from PIL import Image
    import io

    # Create a simple test image
    img = Image.new('RGB', (224, 224), color=(255, 128, 0))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr.getvalue()


@pytest.fixture
def sample_prediction_data():
    """Sample prediction data for testing"""
    return {
        'session_id': 'test_session_123',
        'predictions': [
            {'disease': 'Leaf Blight', 'confidence': 0.85},
            {'disease': 'Healthy', 'confidence': 0.15}
        ],
        'confidence': 0.85,
        'crop_type': 'tomato',
        'confidence_breakdown': {
            'image_prediction': 0.85,
            'crop_validation': 0.90,
            'qa_reasoning': 0.80
        },
        'timestamp': '2024-01-01T10:00:00Z'
    }


@pytest.fixture
def sample_question_data():
    """Sample question data for testing"""
    return [
        {
            'id': 'q1',
            'question': 'Are there visible spots on the leaves?',
            'options': ['Yes', 'No', 'Not sure']
        },
        {
            'id': 'q2',
            'question': 'What color are the affected areas?',
            'options': ['Yellow', 'Brown', 'Black', 'White']
        }
    ]


# Custom markers
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: marks tests that require GPU"
    )


# Test data setup
@pytest.fixture(scope="session", autouse=True)
def create_test_data(test_data_dir):
    """Create test data directory and sample files"""
    test_data_dir.mkdir(exist_ok=True)

    # Create sample images for testing
    from PIL import Image

    # Healthy leaf image
    healthy_img = Image.new('RGB', (224, 224), color=(34, 139, 34))  # Green
    healthy_img.save(test_data_dir / "healthy_leaf.jpg")

    # Diseased leaf image
    diseased_img = Image.new('RGB', (224, 224), color=(139, 69, 19))  # Brown spots
    diseased_img.save(test_data_dir / "diseased_leaf.jpg")

    # Low quality image
    low_quality_img = Image.new('RGB', (100, 100), color=(255, 0, 0))
    low_quality_img.save(test_data_dir / "low_quality.jpg")


# Environment detection
@pytest.fixture(scope="session")
def is_ci():
    """Check if running in CI environment"""
    return os.environ.get('CI') == 'true'


@pytest.fixture(scope="session")
def has_gpu():
    """Check if GPU is available"""
    try:
        import tensorflow as tf
        return len(tf.config.list_physical_devices('GPU')) > 0
    except ImportError:
        return False


# Performance testing utilities
@pytest.fixture
def performance_timer():
    """Timer fixture for performance testing"""
    import time

    class Timer:
        def __enter__(self):
            self.start = time.time()
            return self

        def __exit__(self, *args):
            self.end = time.time()
            self.duration = self.end - self.start

    return Timer


# Mock services for testing
@pytest.fixture
def mock_prediction_service():
    """Mock prediction service for testing"""
    from unittest.mock import Mock

    service = Mock()
    service.predict_disease.return_value = {
        'success': True,
        'data': {
            'session_id': 'mock_session_123',
            'predictions': [{'disease': 'Mock Disease', 'confidence': 0.8}],
            'confidence': 0.8,
            'crop_type': 'mock_crop'
        }
    }
    return service


@pytest.fixture
def mock_llm_service():
    """Mock LLM service for testing"""
    from unittest.mock import Mock

    service = Mock()
    service.generate_questions.return_value = [
        {
            'id': 'q1',
            'question': 'Mock question?',
            'options': ['Yes', 'No']
        }
    ]
    service.answer_question.return_value = "Mock answer"
    return service


@pytest.fixture
def mock_session_service():
    """Mock session service for testing"""
    from unittest.mock import Mock

    service = Mock()
    service.create_session.return_value = 'mock_session_123'
    service.get_session.return_value = {'mock': 'data'}
    return service