# services/offline_inference.py - Offline inference with TensorFlow Lite
"""
Offline Inference Service - Enables local model inference without server calls
- TensorFlow Lite model loading and inference
- Image preprocessing for mobile models
- Caching and optimization for offline use
- Fallback mechanisms for different model formats
"""

import os
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
import io
import base64
import structlog

# TensorFlow Lite imports
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tensorflow as tflite
        TFLITE_AVAILABLE = True
    except ImportError:
        TFLITE_AVAILABLE = False
        logger = structlog.get_logger(__name__)
        logger.warning("TensorFlow Lite not available, offline inference disabled")

logger = structlog.get_logger(__name__)

class OfflineInferenceService:
    """Service for offline model inference using TensorFlow Lite"""

    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or os.path.join(os.path.dirname(__file__), '..', 'models')
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.class_names = []
        self.model_loaded = False

        if TFLITE_AVAILABLE:
            self.load_model()
        else:
            logger.error("TensorFlow Lite not available")

    def load_model(self) -> bool:
        """
        Load TensorFlow Lite model and metadata

        Returns:
            bool: True if model loaded successfully
        """
        try:
            model_path = os.path.join(self.model_dir, 'model.tflite')

            if not os.path.exists(model_path):
                logger.warning(f"TFLite model not found at {model_path}")
                return False

            # Load TFLite model
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()

            # Get input/output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            # Load class names
            self._load_class_names()

            self.model_loaded = True
            logger.info("TFLite model loaded successfully",
                       input_shape=self.input_details[0]['shape'],
                       output_shape=self.output_details[0]['shape'])

            return True

        except Exception as e:
            logger.error("Failed to load TFLite model", error=str(e))
            return False

    def _load_class_names(self):
        """Load class names from metadata file"""
        try:
            metadata_path = os.path.join(self.model_dir, 'class_names.txt')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.class_names = [line.strip() for line in f.readlines()]
            else:
                # Default class names
                self.class_names = [
                    'Bacterial Blight', 'Fungal Infection', 'Viral Disease',
                    'Nutrient Deficiency', 'Pest Damage', 'Healthy'
                ]

            logger.info(f"Loaded {len(self.class_names)} class names")

        except Exception as e:
            logger.warning("Failed to load class names", error=str(e))
            self.class_names = ['Unknown'] * 6

    def preprocess_image(self, image_data: bytes, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Preprocess image for model inference

        Args:
            image_data: Raw image bytes
            target_size: Target image size (height, width)

        Returns:
            Preprocessed image array
        """
        try:
            # Open image
            image = Image.open(io.BytesIO(image_data))

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Resize image
            image = image.resize(target_size, Image.Resampling.LANCZOS)

            # Convert to numpy array
            img_array = np.array(image, dtype=np.float32)

            # Normalize to [0, 1]
            img_array = img_array / 255.0

            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)

            return img_array

        except Exception as e:
            logger.error("Image preprocessing failed", error=str(e))
            raise ValueError(f"Failed to preprocess image: {str(e)}")

    def predict(self, image_data: bytes) -> Dict[str, Any]:
        """
        Perform offline inference on image

        Args:
            image_data: Raw image bytes

        Returns:
            Dict with prediction results
        """
        if not self.model_loaded or not TFLITE_AVAILABLE:
            raise RuntimeError("Offline inference not available")

        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_data)

            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], processed_image)

            # Run inference
            self.interpreter.invoke()

            # Get output
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

            # Process results
            predictions = self._process_predictions(output_data[0])

            result = {
                "predictions": predictions,
                "model_type": "tflite_offline",
                "inference_time": None,  # Could be measured if needed
                "offline": True
            }

            logger.info("Offline inference completed", top_prediction=predictions[0]['class'])
            return result

        except Exception as e:
            logger.error("Offline inference failed", error=str(e))
            raise RuntimeError(f"Offline inference failed: {str(e)}")

    def _process_predictions(self, output_array: np.ndarray) -> List[Dict[str, Any]]:
        """
        Process model output into prediction results

        Args:
            output_array: Raw model output

        Returns:
            List of prediction dictionaries
        """
        try:
            # Apply softmax if needed
            if len(output_array.shape) == 1:
                # Single output vector - assume logits
                exp_scores = np.exp(output_array - np.max(output_array))
                probabilities = exp_scores / np.sum(exp_scores)
            else:
                # Already probabilities
                probabilities = output_array

            # Create prediction results
            predictions = []
            for i, (class_name, probability) in enumerate(zip(self.class_names, probabilities)):
                predictions.append({
                    "class": class_name,
                    "confidence": float(probability),
                    "index": i
                })

            # Sort by confidence
            predictions.sort(key=lambda x: x['confidence'], reverse=True)

            return predictions

        except Exception as e:
            logger.error("Prediction processing failed", error=str(e))
            # Return fallback predictions
            return [
                {"class": "Unknown", "confidence": 0.5, "index": 0},
                {"class": "Healthy", "confidence": 0.3, "index": 1}
            ]

    def predict_from_base64(self, base64_image: str) -> Dict[str, Any]:
        """
        Perform inference from base64 encoded image

        Args:
            base64_image: Base64 encoded image string

        Returns:
            Dict with prediction results
        """
        try:
            # Decode base64
            image_data = base64.b64decode(base64_image.split(',')[1] if ',' in base64_image else base64_image)
            return self.predict(image_data)

        except Exception as e:
            logger.error("Base64 prediction failed", error=str(e))
            raise ValueError(f"Invalid base64 image: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model

        Returns:
            Dict with model information
        """
        if not self.model_loaded:
            return {"loaded": False, "error": "Model not loaded"}

        return {
            "loaded": True,
            "input_shape": self.input_details[0]['shape'].tolist() if self.input_details else None,
            "output_shape": self.output_details[0]['shape'].tolist() if self.output_details else None,
            "classes": self.class_names,
            "num_classes": len(self.class_names),
            "tflite_available": TFLITE_AVAILABLE
        }

    def is_available(self) -> bool:
        """Check if offline inference is available"""
        return self.model_loaded and TFLITE_AVAILABLE

    def cache_model_for_offline(self, model_data: bytes, metadata: Dict[str, Any] = None) -> bool:
        """
        Cache model data for offline use (for PWA)

        Args:
            model_data: TFLite model binary data
            metadata: Optional metadata

        Returns:
            bool: True if cached successfully
        """
        try:
            # Save model
            model_path = os.path.join(self.model_dir, 'cached_model.tflite')
            with open(model_path, 'wb') as f:
                f.write(model_data)

            # Save metadata
            if metadata:
                metadata_path = os.path.join(self.model_dir, 'cached_metadata.json')
                with open(metadata_path, 'w') as f:
                    import json
                    json.dump(metadata, f)

            logger.info("Model cached for offline use")
            return True

        except Exception as e:
            logger.error("Model caching failed", error=str(e))
            return False

    def get_cached_model_data(self) -> Optional[bytes]:
        """
        Get cached model data for PWA

        Returns:
            Model data bytes or None
        """
        try:
            model_path = os.path.join(self.model_dir, 'cached_model.tflite')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    return f.read()
            return None

        except Exception as e:
            logger.error("Failed to read cached model", error=str(e))
            return None