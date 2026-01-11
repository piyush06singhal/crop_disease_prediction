# services/prediction_service.py - Core prediction service
"""
Prediction Service - Orchestrates disease prediction workflow.

Responsibilities:
- Coordinate ML inference and LLM reasoning
- Manage confidence calculation and refinement
- Handle follow-up question generation
- Integrate explainability features

Architecture:
- Uses ML service for model inference
- Uses LLM service for intelligent questioning
- Implements progressive confidence system
"""

import os
import logging
from typing import Dict, List, Optional, Tuple
from ml.model_service import ModelService
from services.llm_service import LLMService
from utils.confidence_engine import ConfidenceEngine
from utils.image_processor import ImageProcessor
from services.session_service import SessionService
from flask import current_app

logger = logging.getLogger(__name__)

class PredictionService:
    """
    Main service for handling disease prediction requests.
    Implements the core business logic for the prediction pipeline.
    """

    def __init__(self):
        """Initialize prediction service with required components"""
        self.model_service = ModelService()
        self.llm_service = LLMService()
        self.confidence_engine = ConfidenceEngine()
        self.image_processor = ImageProcessor()
        self.session_service = SessionService()

        # Supported crops and diseases (would be loaded from config/dataset)
        self.supported_crops = [
            'tomato', 'potato', 'corn', 'pepper', 'apple', 'grape',
            'orange', 'peach', 'strawberry', 'cherry', 'wheat', 'rice'
        ]

        # Disease classes (PlantVillage dataset has 38 classes)
        self.disease_classes = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
            'Apple___healthy', 'Blueberry___healthy', 'Cherry___Powdery_mildew',
            'Cherry___healthy', 'Corn___Cercospora_leaf_spot', 'Corn___Common_rust',
            'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)',
            'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot',
            'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
            'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
            'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
            'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
            'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]  # PlantVillage dataset classes

    def is_model_loaded(self) -> bool:
        """Check if ML model is loaded and ready"""
        return self.model_service.is_loaded()

    def is_llm_available(self) -> bool:
        """Check if LLM service is available"""
        return self.llm_service.is_available()

    def get_supported_crops(self) -> List[str]:
        """Get list of supported crop types"""
        return self.supported_crops

    def predict_disease(self, image_path: str, crop_type: Optional[str],
                       session_id: str) -> Dict:
        """
        Perform initial disease prediction from image.

        Args:
            image_path: Path to input image
            crop_type: Optional crop type hint
            session_id: Unique session identifier

        Returns:
            Dict containing prediction results and next steps
        """
        try:
            # Preprocess image
            processed_image = self.image_processor.preprocess(image_path)

            # Get ML predictions
            predictions = self.model_service.predict(processed_image)

            # Infer crop type if not provided
            if not crop_type:
                crop_type = self._infer_crop_type(predictions)

            # Calculate initial confidence
            initial_confidence = self.confidence_engine.calculate_initial_confidence(
                predictions, crop_type
            )

            # Generate follow-up questions if confidence is low
            questions = []
            if initial_confidence < 0.8:  # Threshold for questioning
                questions = self.llm_service.generate_questions(
                    predictions, crop_type, session_id
                )

            # Prepare response
            result = {
                'session_id': session_id,
                'crop_type': crop_type,
                'predictions': predictions[:5],  # Top 5 predictions
                'confidence': initial_confidence,
                'confidence_breakdown': {
                    'image_prediction': initial_confidence,
                    'crop_validation': 0.0,
                    'qa_reasoning': 0.0
                },
                'questions': questions,
                'status': 'initial_prediction'
            }

            logger.info(f"Initial prediction completed for session {session_id}")
            return result

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def process_answer(self, session_id: str, question_id: str, answer: str) -> Dict:
        """
        Process user answer to follow-up question and refine prediction.

        Args:
            session_id: Session identifier
            question_id: Question identifier
            answer: User's answer

        Returns:
            Dict with refined prediction
        """
        try:
            # Get current session data
            session_data = self.session_service.get_prediction(session_id)
            if not session_data:
                raise ValueError(f"Session {session_id} not found")

            # Use LLM to analyze answer and refine confidence
            refinement = self.llm_service.analyze_answer(
                session_id, question_id, answer
            )

            # Update confidence based on answer
            updated_confidence = self.confidence_engine.refine_confidence(
                refinement, 'qa_reasoning'
            )

            # Generate next question if confidence still low
            next_questions = []
            if updated_confidence < 0.9:
                next_questions = self.llm_service.generate_next_question(
                    session_id, refinement
                )

            result = {
                'session_id': session_id,
                'refined_confidence': updated_confidence,
                'confidence_breakdown': refinement.get('confidence_breakdown', {}),
                'next_questions': next_questions,
                'reasoning': refinement.get('reasoning', ''),
                'status': 'refined'
            }

            return result

        except Exception as e:
            logger.error(f"Answer processing failed: {str(e)}")
            raise

    def refine_prediction(self, session_id: str, additional_data: Dict) -> Dict:
        """
        Refine prediction with additional data.

        Args:
            session_id: Session identifier
            additional_data: Additional information for refinement

        Returns:
            Dict with refined prediction
        """
        # Implementation for additional data refinement
        # Could include weather data, location, etc.
        pass

    def get_explanation(self, session_id: str) -> Dict:
        """
        Generate explanation for prediction including Grad-CAM.

        Args:
            session_id: Session identifier

        Returns:
            Dict with explanation data
        """
        try:
            # Get original prediction data
            session_data = self.session_service.get_prediction(session_id)
            if not session_data:
                raise ValueError(f"Session {session_id} not found")

            # Generate Grad-CAM heatmap
            heatmap_data = self.model_service.generate_gradcam(session_id)

            explanation = {
                'session_id': session_id,
                'heatmap': heatmap_data,
                'feature_importance': {},  # Would include feature explanations
                'reasoning': 'Explanation based on visual features detected'
            }

            return explanation

        except Exception as e:
            logger.error(f"Explanation generation failed: {str(e)}")
            raise

    def _infer_crop_type(self, predictions: List[Dict]) -> str:
        """
        Infer crop type from prediction results.

        Args:
            predictions: List of prediction dictionaries

        Returns:
            Inferred crop type
        """
        # Simple inference based on disease patterns
        # In production, would use a separate crop classification model
        disease_names = [pred['disease'] for pred in predictions]

        crop_mappings = {
            'tomato': ['Tomato_', 'tomato'],
            'potato': ['Potato_', 'potato'],
            'corn': ['Corn_', 'corn'],
            'apple': ['Apple_', 'apple'],
            'grape': ['Grape_', 'grape'],
            # Add more mappings
        }

        for crop, diseases in crop_mappings.items():
            if any(prefix.lower() in disease_names[0].lower() for prefix in diseases):
                return crop

        return 'unknown'  # Default fallback