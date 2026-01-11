# tests/test_services.py - Unit tests for backend services
"""
Unit tests for Crop Disease Prediction System backend services
Tests prediction, model, LLM, session, confidence engine, and image processor services
"""

import pytest
import numpy as np
import cv2
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import io

# Import services (assuming they're in the backend directory)
import sys
sys.path.append('backend')

from services.prediction_service import PredictionService
from services.model_service import ModelService
from services.llm_service import LLMService
from services.session_service import SessionService
from services.confidence_engine import ConfidenceEngine
from services.image_processor import ImageProcessor


class TestPredictionService:
    """Test cases for PredictionService"""

    @pytest.fixture
    def prediction_service(self):
        """Create PredictionService instance with mocked dependencies"""
        with patch('services.prediction_service.ModelService'), \
             patch('services.prediction_service.LLMService'), \
             patch('services.prediction_service.SessionService'), \
             patch('services.prediction_service.ConfidenceEngine'), \
             patch('services.prediction_service.ImageProcessor'):
            service = PredictionService()
            yield service

    def test_initialization(self, prediction_service):
        """Test service initialization"""
        assert prediction_service is not None
        assert hasattr(prediction_service, 'model_service')
        assert hasattr(prediction_service, 'llm_service')
        assert hasattr(prediction_service, 'session_service')
        assert hasattr(prediction_service, 'confidence_engine')
        assert hasattr(prediction_service, 'image_processor')

    @patch('services.prediction_service.cv2.imread')
    def test_predict_disease_success(self, mock_imread, prediction_service):
        """Test successful disease prediction"""
        # Mock image reading
        mock_image = np.random.rand(224, 224, 3).astype(np.uint8)
        mock_imread.return_value = mock_image

        # Mock dependencies
        prediction_service.model_service.predict.return_value = {
            'predictions': [{'disease': 'Leaf Blight', 'confidence': 0.85}],
            'crop_type': 'tomato'
        }
        prediction_service.confidence_engine.calculate_confidence.return_value = {
            'overall_confidence': 0.85,
            'breakdown': {
                'image_prediction': 0.85,
                'crop_validation': 0.90,
                'qa_reasoning': 0.80
            }
        }
        prediction_service.session_service.create_session.return_value = 'session_123'

        # Test prediction
        result = prediction_service.predict_disease('test_image.jpg')

        assert result['success'] is True
        assert 'session_id' in result['data']
        assert 'predictions' in result['data']
        assert 'confidence' in result['data']
        assert result['data']['crop_type'] == 'tomato'

    def test_predict_disease_invalid_image(self, prediction_service):
        """Test prediction with invalid image"""
        with patch('services.prediction_service.cv2.imread', return_value=None):
            result = prediction_service.predict_disease('invalid_image.jpg')

            assert result['success'] is False
            assert 'error' in result

    def test_predict_disease_with_questions(self, prediction_service):
        """Test prediction that generates follow-up questions"""
        # Mock successful prediction
        prediction_service.model_service.predict.return_value = {
            'predictions': [{'disease': 'Leaf Blight', 'confidence': 0.75}],
            'crop_type': 'tomato'
        }
        prediction_service.confidence_engine.calculate_confidence.return_value = {
            'overall_confidence': 0.75,
            'breakdown': {
                'image_prediction': 0.75,
                'crop_validation': 0.80,
                'qa_reasoning': 0.70
            }
        }
        prediction_service.llm_service.generate_questions.return_value = [
            {
                'id': 'q1',
                'question': 'Are there yellow spots on the leaves?',
                'options': ['Yes', 'No', 'Not sure']
            }
        ]

        result = prediction_service.predict_disease('test_image.jpg')

        assert result['success'] is True
        assert 'questions' in result['data']
        assert len(result['data']['questions']) > 0


class TestModelService:
    """Test cases for ModelService"""

    @pytest.fixture
    def model_service(self):
        """Create ModelService instance"""
        service = ModelService()
        yield service

    def test_initialization(self, model_service):
        """Test service initialization"""
        assert model_service is not None
        assert hasattr(model_service, 'models')

    @patch('tensorflow.keras.models.load_model')
    def test_load_model(self, mock_load_model, model_service):
        """Test model loading"""
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        model_service.load_model('test_model')

        assert 'test_model' in model_service.models
        assert model_service.models['test_model'] == mock_model

    def test_predict_with_mock_model(self, model_service):
        """Test prediction with mock model"""
        # Create mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.1, 0.8, 0.1]])
        model_service.models['test_model'] = mock_model

        # Mock image preprocessing
        with patch.object(model_service, 'preprocess_image', return_value=np.random.rand(1, 224, 224, 3)):
            result = model_service.predict(np.random.rand(224, 224, 3), 'test_model')

            assert 'predictions' in result
            assert len(result['predictions']) > 0
            assert 'crop_type' in result


class TestLLMService:
    """Test cases for LLMService"""

    @pytest.fixture
    def llm_service(self):
        """Create LLMService instance"""
        service = LLMService()
        yield service

    def test_initialization(self, llm_service):
        """Test service initialization"""
        assert llm_service is not None

    @patch('services.llm_service.genai.GenerativeModel')
    def test_generate_questions(self, mock_model_class, llm_service):
        """Test question generation"""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = '''
        Based on the analysis, here are some questions:
        1. Are there visible spots on the leaves?
        2. What is the color of the affected areas?
        '''
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        questions = llm_service.generate_questions('Leaf spots detected', 0.7)

        assert isinstance(questions, list)
        assert len(questions) > 0

    @patch('services.llm_service.genai.GenerativeModel')
    def test_answer_question(self, mock_model_class, llm_service):
        """Test question answering"""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = 'Based on the symptoms, this appears to be fungal infection.'
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        answer = llm_service.answer_question(
            'What disease is this?',
            'Yellow spots on tomato leaves',
            'User observed yellow spots'
        )

        assert isinstance(answer, str)
        assert len(answer) > 0


class TestSessionService:
    """Test cases for SessionService"""

    @pytest.fixture
    def session_service(self):
        """Create SessionService instance"""
        service = SessionService()
        yield service

    def test_initialization(self, session_service):
        """Test service initialization"""
        assert session_service is not None

    def test_create_session(self, session_service):
        """Test session creation"""
        session_id = session_service.create_session()

        assert isinstance(session_id, str)
        assert len(session_id) > 0
        assert session_id in session_service.sessions

    def test_store_prediction(self, session_service):
        """Test storing prediction in session"""
        session_id = session_service.create_session()
        prediction_data = {
            'predictions': [{'disease': 'Test Disease', 'confidence': 0.8}],
            'confidence': 0.8
        }

        session_service.store_prediction(session_id, prediction_data)

        assert session_service.sessions[session_id]['predictions'] == prediction_data

    def test_get_session(self, session_service):
        """Test retrieving session data"""
        session_id = session_service.create_session()
        prediction_data = {'test': 'data'}
        session_service.store_prediction(session_id, prediction_data)

        session = session_service.get_session(session_id)

        assert session is not None
        assert session['predictions'] == prediction_data


class TestConfidenceEngine:
    """Test cases for ConfidenceEngine"""

    @pytest.fixture
    def confidence_engine(self):
        """Create ConfidenceEngine instance"""
        service = ConfidenceEngine()
        yield service

    def test_initialization(self, confidence_engine):
        """Test service initialization"""
        assert confidence_engine is not None

    def test_calculate_confidence(self, confidence_engine):
        """Test confidence calculation"""
        image_confidence = 0.8
        crop_confidence = 0.9
        qa_confidence = 0.7

        result = confidence_engine.calculate_confidence(
            image_confidence, crop_confidence, qa_confidence
        )

        assert 'overall_confidence' in result
        assert 'breakdown' in result
        assert isinstance(result['overall_confidence'], float)
        assert 0 <= result['overall_confidence'] <= 1

    def test_update_weights(self, confidence_engine):
        """Test weight updates"""
        # Test with feedback
        confidence_engine.update_weights(True, 0.8, 0.9)

        # Weights should be updated
        assert hasattr(confidence_engine, 'weights')


class TestImageProcessor:
    """Test cases for ImageProcessor"""

    @pytest.fixture
    def image_processor(self):
        """Create ImageProcessor instance"""
        service = ImageProcessor()
        yield service

    def test_initialization(self, image_processor):
        """Test service initialization"""
        assert image_processor is not None

    def test_preprocess_image(self, image_processor):
        """Test image preprocessing"""
        # Create test image
        test_image = np.random.rand(300, 400, 3).astype(np.uint8)

        processed = image_processor.preprocess_image(test_image)

        assert processed.shape[0] == 1  # Batch dimension
        assert processed.shape[1] == 224  # Height
        assert processed.shape[2] == 224  # Width
        assert processed.shape[3] == 3    # Channels

    def test_enhance_image_quality(self, image_processor):
        """Test image quality enhancement"""
        # Create low-quality test image
        test_image = np.random.rand(100, 100, 3).astype(np.uint8)

        enhanced = image_processor.enhance_image_quality(test_image)

        assert enhanced.shape == test_image.shape
        assert enhanced.dtype == test_image.dtype

    def test_detect_crop_type(self, image_processor):
        """Test crop type detection"""
        # Create test image
        test_image = np.random.rand(224, 224, 3).astype(np.uint8)

        crop_type = image_processor.detect_crop_type(test_image)

        assert isinstance(crop_type, str)
        assert len(crop_type) > 0


# Integration tests
class TestServiceIntegration:
    """Integration tests for service interactions"""

    @pytest.fixture
    def services(self):
        """Create all services with mocked external dependencies"""
        with patch('services.prediction_service.ModelService'), \
             patch('services.prediction_service.LLMService'), \
             patch('services.prediction_service.SessionService'), \
             patch('services.prediction_service.ConfidenceEngine'), \
             patch('services.prediction_service.ImageProcessor'):
            prediction_service = PredictionService()
            yield {
                'prediction': prediction_service,
                'model': prediction_service.model_service,
                'llm': prediction_service.llm_service,
                'session': prediction_service.session_service,
                'confidence': prediction_service.confidence_engine,
                'image': prediction_service.image_processor
            }

    def test_full_prediction_workflow(self, services):
        """Test complete prediction workflow"""
        # Mock all dependencies for successful workflow
        services['model'].predict.return_value = {
            'predictions': [{'disease': 'Healthy', 'confidence': 0.95}],
            'crop_type': 'tomato'
        }
        services['confidence'].calculate_confidence.return_value = {
            'overall_confidence': 0.95,
            'breakdown': {
                'image_prediction': 0.95,
                'crop_validation': 1.0,
                'qa_reasoning': 0.9
            }
        }
        services['session'].create_session.return_value = 'test_session_123'

        result = services['prediction'].predict_disease('test_image.jpg')

        assert result['success'] is True
        assert result['data']['session_id'] == 'test_session_123'
        assert result['data']['confidence'] == 0.95

    def test_error_handling_workflow(self, services):
        """Test error handling in workflow"""
        # Mock model failure
        services['model'].predict.side_effect = Exception("Model prediction failed")

        result = services['prediction'].predict_disease('test_image.jpg')

        assert result['success'] is False
        assert 'error' in result


if __name__ == '__main__':
    pytest.main([__file__])