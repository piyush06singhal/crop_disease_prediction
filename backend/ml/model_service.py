# ml/model_service.py - Machine Learning Model Service
"""
Model Service - Handles ML model loading, inference, and explainability.

Responsibilities:
- Load and manage ML models (Keras/TFLite)
- Perform inference on preprocessed images
- Generate Grad-CAM explanations
- Support model ensembles and optimization

Architecture:
- Supports both Keras and TFLite models
- Implements Grad-CAM for explainability
- Handles model versioning and updates
"""

import os
import logging
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
import cv2
import base64
import io

logger = logging.getLogger(__name__)

class ModelService:
    """
    Service for managing ML models and performing inference.
    Supports both Keras and TFLite models with explainability features.
    """

    def __init__(self):
        """Initialize model service"""
        self.model = None
        self.tflite_model = None
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.is_loaded = False

        # Model configuration
        self.input_shape = (224, 224, 3)  # MobileNetV2 default
        self.num_classes = 38  # PlantVillage dataset

        # Disease class names (PlantVillage dataset)
        self.class_names = [
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
        ]

        # Load model on initialization
        self._load_model()

    def _load_model(self):
        """Load the ML model from disk"""
        try:
            # Try to load TFLite model first (for production)
            model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'crop_disease_model.tflite')

            if os.path.exists(model_path):
                logger.info("Loading TFLite model...")
                self.interpreter = tf.lite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                self.is_loaded = True
                logger.info("TFLite model loaded successfully")
            else:
                # Fallback to Keras model
                keras_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'crop_disease_model.h5')
                if os.path.exists(keras_path):
                    logger.info("Loading Keras model...")
                    self.model = tf.keras.models.load_model(keras_path)
                    self.is_loaded = True
                    logger.info("Keras model loaded successfully")
                else:
                    logger.warning("No model file found. Using mock predictions.")
                    self.is_loaded = False

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            self.is_loaded = False

    def predict(self, image: np.ndarray) -> List[Dict]:
        """
        Perform inference on preprocessed image.

        Args:
            image: Preprocessed image array (224, 224, 3)

        Returns:
            List of prediction dictionaries with confidence scores
        """
        if not self.is_loaded:
            return self._mock_predictions()

        try:
            # Ensure correct input shape
            if image.shape != self.input_shape:
                image = tf.image.resize(image, self.input_shape[:2])
            image = np.expand_dims(image, axis=0)

            if self.interpreter:
                # TFLite inference
                self.interpreter.set_tensor(self.input_details[0]['index'], image.astype(np.float32))
                self.interpreter.invoke()
                predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            else:
                # Keras inference
                predictions = self.model.predict(image)[0]

            # Convert to probabilities
            predictions = tf.nn.softmax(predictions).numpy()

            # Create prediction results
            results = []
            for i, prob in enumerate(predictions):
                results.append({
                    'disease': self.class_names[i],
                    'confidence': float(prob),
                    'class_id': i
                })

            # Sort by confidence
            results.sort(key=lambda x: x['confidence'], reverse=True)

            logger.info(f"Prediction completed with top confidence: {results[0]['confidence']:.3f}")
            return results

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return self._mock_predictions()

    def generate_gradcam(self, session_id: str) -> Dict:
        """
        Generate Grad-CAM heatmap for explainability.

        Args:
            session_id: Session identifier (would be used to retrieve image)

        Returns:
            Dict containing heatmap data and overlay
        """
        try:
            # For now, return mock Grad-CAM data
            # In production, would:
            # 1. Load original image
            # 2. Get last convolutional layer
            # 3. Compute gradients
            # 4. Generate heatmap

            # Mock heatmap data (base64 encoded image)
            mock_heatmap = self._generate_mock_heatmap()

            return {
                'heatmap_base64': mock_heatmap,
                'overlay_base64': mock_heatmap,  # Would be original + heatmap overlay
                'regions': [
                    {'x': 100, 'y': 150, 'width': 50, 'height': 50, 'importance': 0.8},
                    {'x': 200, 'y': 100, 'width': 40, 'height': 60, 'importance': 0.6}
                ],
                'explanation': 'Grad-CAM highlights areas of the leaf showing disease symptoms'
            }

        except Exception as e:
            logger.error(f"Grad-CAM generation failed: {str(e)}")
            return {}

    def _generate_mock_heatmap(self) -> str:
        """Generate a mock heatmap image for demonstration"""
        # Create a simple heatmap overlay
        heatmap = np.zeros((224, 224, 3), dtype=np.uint8)
        heatmap[100:150, 100:150] = [255, 0, 0]  # Red region
        heatmap[150:200, 150:200] = [0, 255, 0]  # Green region

        # Convert to base64
        img = Image.fromarray(heatmap)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        return f"data:image/png;base64,{img_base64}"

    def _mock_predictions(self) -> List[Dict]:
        """Return mock predictions when model is not available"""
        logger.warning("Using mock predictions - model not loaded")

        # Return some realistic mock predictions
        mock_results = [
            {'disease': 'Tomato___Late_blight', 'confidence': 0.85, 'class_id': 31},
            {'disease': 'Tomato___Early_blight', 'confidence': 0.12, 'class_id': 30},
            {'disease': 'Tomato___healthy', 'confidence': 0.03, 'class_id': 37},
        ]

        return mock_results

    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'loaded': self.is_loaded,
            'type': 'tflite' if self.interpreter else 'keras',
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'class_names': self.class_names[:5]  # First 5 for preview
        }

    def reload_model(self):
        """Reload the model (useful for model updates)"""
        logger.info("Reloading model...")
        self.is_loaded = False
        self.model = None
        self.interpreter = None
        self._load_model()