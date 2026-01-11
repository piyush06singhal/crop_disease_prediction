# services/continual_learning.py - Continual learning for model improvement
"""
Continual Learning Service - Implements continuous model improvement
- Background model retraining with new user data
- Incremental learning techniques
- Model versioning and rollback
- Performance monitoring and automated updates
- Data quality validation and filtering
"""

import os
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import structlog

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger = structlog.get_logger(__name__)
    logger.warning("TensorFlow not available, continual learning disabled")

logger = structlog.get_logger(__name__)

class ContinualLearningService:
    """Service for continuous model improvement with new data"""

    def __init__(self, model_dir: str = None, data_dir: str = None):
        self.model_dir = model_dir or os.path.join(os.path.dirname(__file__), '..', 'models')
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), '..', 'continual_learning_data')
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="continual-learning")

        # Learning parameters
        self.min_samples_for_retraining = 100
        self.retraining_interval_days = 7
        self.performance_threshold = 0.85  # Minimum accuracy to keep new model
        self.max_retraining_history = 10

        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, 'versions'), exist_ok=True)

        # Load learning state
        self.learning_state = self._load_learning_state()

        # Start background learning if enabled
        if TF_AVAILABLE:
            self._start_background_learning()

    def _load_learning_state(self) -> Dict[str, Any]:
        """Load continual learning state from disk"""
        state_file = os.path.join(self.data_dir, 'learning_state.json')

        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error("Failed to load learning state", error=str(e))

        # Default state
        return {
            "last_retraining": None,
            "total_samples": 0,
            "model_versions": [],
            "current_model_version": "v1.0.0",
            "performance_history": [],
            "data_quality_metrics": {
                "avg_confidence": 0.0,
                "class_distribution": {},
                "feedback_accuracy": 0.0
            }
        }

    def _save_learning_state(self):
        """Save learning state to disk"""
        state_file = os.path.join(self.data_dir, 'learning_state.json')

        try:
            with open(state_file, 'w') as f:
                json.dump(self.learning_state, f, indent=2, default=str)
        except Exception as e:
            logger.error("Failed to save learning state", error=str(e))

    def _start_background_learning(self):
        """Start background learning thread"""
        def background_worker():
            while True:
                try:
                    self._check_and_retrain()
                    time.sleep(3600)  # Check every hour
                except Exception as e:
                    logger.error("Background learning error", error=str(e))
                    time.sleep(300)  # Wait 5 minutes on error

        thread = threading.Thread(target=background_worker, daemon=True, name="continual-learning")
        thread.start()
        logger.info("Background continual learning started")

    def add_training_sample(self, image_path: str, true_label: str, predicted_label: str,
                          confidence: float, user_feedback: str = None) -> bool:
        """
        Add a new training sample from user feedback

        Args:
            image_path: Path to the image file
            true_label: Correct disease label
            predicted_label: Model's prediction
            confidence: Model confidence score
            user_feedback: Optional user feedback text

        Returns:
            bool: True if sample added successfully
        """
        try:
            # Validate input
            if not os.path.exists(image_path):
                logger.warning("Image file not found", path=image_path)
                return False

            if not true_label or true_label == predicted_label:
                # Skip if no correction provided or prediction was correct
                return False

            # Create sample record
            sample = {
                "image_path": image_path,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "user_feedback": user_feedback,
                "timestamp": datetime.now().isoformat(),
                "added_to_training": False
            }

            # Save sample
            samples_file = os.path.join(self.data_dir, 'training_samples.jsonl')
            with open(samples_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

            # Update state
            self.learning_state["total_samples"] += 1
            self._update_data_quality_metrics(sample)
            self._save_learning_state()

            logger.info("Training sample added",
                       true_label=true_label,
                       predicted_label=predicted_label,
                       total_samples=self.learning_state["total_samples"])

            return True

        except Exception as e:
            logger.error("Failed to add training sample", error=str(e))
            return False

    def _update_data_quality_metrics(self, sample: Dict[str, Any]):
        """Update data quality metrics with new sample"""
        metrics = self.learning_state["data_quality_metrics"]

        # Update average confidence
        total_samples = self.learning_state["total_samples"]
        current_avg = metrics["avg_confidence"]
        new_confidence = sample["confidence"]
        metrics["avg_confidence"] = (current_avg * (total_samples - 1) + new_confidence) / total_samples

        # Update class distribution
        true_label = sample["true_label"]
        metrics["class_distribution"][true_label] = metrics["class_distribution"].get(true_label, 0) + 1

    def _check_and_retrain(self):
        """Check if retraining is needed and trigger if appropriate"""
        try:
            # Check if enough time has passed since last retraining
            last_retraining = self.learning_state.get("last_retraining")
            if last_retraining:
                last_retraining_date = datetime.fromisoformat(last_retraining)
                days_since_retraining = (datetime.now() - last_retraining_date).days

                if days_since_retraining < self.retraining_interval_days:
                    return  # Not enough time has passed

            # Check if we have enough samples
            total_samples = self.learning_state["total_samples"]
            if total_samples < self.min_samples_for_retraining:
                logger.info("Not enough samples for retraining",
                           samples=total_samples,
                           required=self.min_samples_for_retraining)
                return

            # Check data quality
            if not self._validate_data_quality():
                logger.warning("Data quality validation failed, skipping retraining")
                return

            # Trigger retraining
            logger.info("Triggering model retraining", samples=total_samples)
            self.executor.submit(self._retrain_model)

        except Exception as e:
            logger.error("Error in retraining check", error=str(e))

    def _validate_data_quality(self) -> bool:
        """Validate quality of accumulated training data"""
        try:
            metrics = self.learning_state["data_quality_metrics"]

            # Check minimum confidence threshold
            if metrics["avg_confidence"] < 0.6:
                logger.warning("Average confidence too low", confidence=metrics["avg_confidence"])
                return False

            # Check class distribution balance
            class_counts = metrics["class_distribution"]
            if len(class_counts) < 3:  # Need at least 3 classes
                logger.warning("Insufficient class diversity", classes=len(class_counts))
                return False

            # Check for class imbalance
            counts = list(class_counts.values())
            max_count = max(counts)
            min_count = min(counts)
            imbalance_ratio = max_count / max_count if min_count == 0 else max_count / min_count

            if imbalance_ratio > 5:  # More than 5x imbalance
                logger.warning("Severe class imbalance detected", ratio=imbalance_ratio)
                return False

            return True

        except Exception as e:
            logger.error("Data quality validation error", error=str(e))
            return False

    def _retrain_model(self):
        """Perform model retraining with accumulated data"""
        try:
            logger.info("Starting model retraining")

            # Load training samples
            samples = self._load_training_samples()
            if not samples:
                logger.warning("No training samples available")
                return

            # Prepare dataset
            train_data, val_data = self._prepare_dataset(samples)

            # Load current model
            current_model_path = os.path.join(self.model_dir, 'crop_disease_model.h5')
            if not os.path.exists(current_model_path):
                logger.error("Current model not found", path=current_model_path)
                return

            model = tf.keras.models.load_model(current_model_path)

            # Fine-tune model
            new_model = self._fine_tune_model(model, train_data, val_data)

            # Evaluate new model
            performance = self._evaluate_model(new_model, val_data)

            # Check if performance improved
            if performance["accuracy"] > self.performance_threshold:
                # Save new model version
                version = self._save_model_version(new_model, performance)

                # Update learning state
                self.learning_state["last_retraining"] = datetime.now().isoformat()
                self.learning_state["current_model_version"] = version
                self.learning_state["performance_history"].append({
                    "version": version,
                    "accuracy": performance["accuracy"],
                    "timestamp": datetime.now().isoformat()
                })

                # Keep only recent history
                self.learning_state["performance_history"] = \
                    self.learning_state["performance_history"][-self.max_retraining_history:]

                # Convert and save TFLite version for offline use
                self._convert_to_tflite(new_model, version)

                logger.info("Model retraining completed successfully",
                           version=version,
                           accuracy=performance["accuracy"])

            else:
                logger.warning("New model performance below threshold, keeping old model",
                             accuracy=performance["accuracy"],
                             threshold=self.performance_threshold)

            self._save_learning_state()

        except Exception as e:
            logger.error("Model retraining failed", error=str(e))

    def _load_training_samples(self) -> List[Dict[str, Any]]:
        """Load training samples from storage"""
        samples_file = os.path.join(self.data_dir, 'training_samples.jsonl')

        if not os.path.exists(samples_file):
            return []

        samples = []
        try:
            with open(samples_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line.strip()))
        except Exception as e:
            logger.error("Failed to load training samples", error=str(e))

        return samples

    def _prepare_dataset(self, samples: List[Dict[str, Any]]) -> Tuple[Any, Any]:
        """Prepare dataset for training"""
        # This is a simplified implementation
        # In production, you'd implement proper data loading and preprocessing

        # For now, return mock data generators
        # Real implementation would load images and create proper generators

        # Create mock data for demonstration
        num_samples = len(samples)
        img_height, img_width = 224, 224

        # Mock training data
        X_train = np.random.rand(num_samples, img_height, img_width, 3)
        y_train = np.random.randint(0, 6, num_samples)  # 6 classes

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        # Create data generators (simplified)
        train_generator = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
        val_generator = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)

        return train_generator, val_generator

    def _fine_tune_model(self, model: tf.keras.Model, train_data: Any, val_data: Any) -> tf.keras.Model:
        """Fine-tune the model with new data"""
        try:
            # Unfreeze some layers for fine-tuning
            for layer in model.layers[-10:]:  # Fine-tune last 10 layers
                layer.trainable = True

            # Compile with lower learning rate
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            # Callbacks
            callbacks = [
                EarlyStopping(patience=5, restore_best_weights=True),
                ModelCheckpoint(
                    os.path.join(self.model_dir, 'temp_model.h5'),
                    save_best_only=True,
                    monitor='val_accuracy'
                )
            ]

            # Fine-tune
            model.fit(
                train_data,
                validation_data=val_data,
                epochs=10,
                callbacks=callbacks,
                verbose=1
            )

            return model

        except Exception as e:
            logger.error("Model fine-tuning failed", error=str(e))
            return model  # Return original model

    def _evaluate_model(self, model: tf.keras.Model, val_data: Any) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            # Evaluate on validation data
            results = model.evaluate(val_data, verbose=0)

            return {
                "loss": results[0],
                "accuracy": results[1]
            }

        except Exception as e:
            logger.error("Model evaluation failed", error=str(e))
            return {"loss": 1.0, "accuracy": 0.0}

    def _save_model_version(self, model: tf.keras.Model, performance: Dict[str, float]) -> str:
        """Save a new version of the model"""
        try:
            # Generate version number
            current_version = self.learning_state.get("current_model_version", "v1.0.0")
            version_parts = current_version.split('.')
            new_version = f"v{version_parts[0][1:]}.{int(version_parts[1]) + 1}.0"

            # Save model
            version_dir = os.path.join(self.model_dir, 'versions', new_version)
            os.makedirs(version_dir, exist_ok=True)

            model_path = os.path.join(version_dir, 'model.h5')
            model.save(model_path)

            # Save metadata
            metadata = {
                "version": new_version,
                "created_at": datetime.now().isoformat(),
                "performance": performance,
                "training_samples": self.learning_state["total_samples"],
                "parent_version": current_version
            }

            metadata_path = os.path.join(version_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            # Update model versions list
            self.learning_state["model_versions"].append({
                "version": new_version,
                "path": version_dir,
                "performance": performance,
                "created_at": datetime.now().isoformat()
            })

            logger.info("Model version saved", version=new_version, accuracy=performance.get("accuracy"))
            return new_version

        except Exception as e:
            logger.error("Failed to save model version", error=str(e))
            return current_version

    def _convert_to_tflite(self, model: tf.keras.Model, version: str):
        """Convert model to TensorFlow Lite format"""
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()

            # Save TFLite model
            tflite_path = os.path.join(self.model_dir, f'model_{version}.tflite')
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)

            # Update main TFLite model
            main_tflite_path = os.path.join(self.model_dir, 'model.tflite')
            with open(main_tflite_path, 'wb') as f:
                f.write(tflite_model)

            logger.info("TFLite model converted", version=version)

        except Exception as e:
            logger.error("TFLite conversion failed", error=str(e))

    def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning status and statistics"""
        return {
            "total_samples": self.learning_state["total_samples"],
            "last_retraining": self.learning_state["last_retraining"],
            "current_version": self.learning_state["current_model_version"],
            "performance_history": self.learning_state["performance_history"],
            "data_quality": self.learning_state["data_quality_metrics"],
            "next_retraining_days": max(0, self.retraining_interval_days -
                                      (datetime.now() - datetime.fromisoformat(
                                          self.learning_state["last_retraining"] or datetime.now().isoformat()
                                      )).days)
        }

    def rollback_model(self, version: str) -> bool:
        """
        Rollback to a previous model version

        Args:
            version: Version to rollback to

        Returns:
            bool: True if rollback successful
        """
        try:
            # Find version in history
            version_info = None
            for v in self.learning_state["model_versions"]:
                if v["version"] == version:
                    version_info = v
                    break

            if not version_info:
                logger.error("Version not found", version=version)
                return False

            # Copy model to main location
            version_model = os.path.join(version_info["path"], 'model.h5')
            main_model = os.path.join(self.model_dir, 'crop_disease_model.h5')

            if os.path.exists(version_model):
                import shutil
                shutil.copy2(version_model, main_model)

                # Update current version
                self.learning_state["current_model_version"] = version
                self._save_learning_state()

                logger.info("Model rollback completed", version=version)
                return True
            else:
                logger.error("Model file not found", path=version_model)
                return False

        except Exception as e:
            logger.error("Model rollback failed", error=str(e))
            return False