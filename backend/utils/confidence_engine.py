# utils/confidence_engine.py - Progressive confidence calculation engine
"""
Confidence Engine - Implements progressive confidence refinement system.

Features:
- Weighted confidence calculation from multiple sources
- Progressive refinement through Q&A
- Confidence thresholds for decision making
- Explainable confidence breakdown
- Data-driven weight learning
- Uncertainty quantification

Confidence Sources:
- Image prediction (dynamic weight)
- Crop validation (dynamic weight)
- Q&A reasoning (dynamic weight)
- Historical accuracy (learned weight)
"""

from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class ConfidenceEngine:
    """
    Engine for calculating and refining prediction confidence.
    Implements weighted confidence system with progressive refinement and learning.
    """

    # Default weights for confidence sources
    DEFAULT_WEIGHTS = {
        'image_prediction': 0.5,
        'crop_validation': 0.2,
        'qa_reasoning': 0.3
    }

    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.9
    MEDIUM_CONFIDENCE_THRESHOLD = 0.7
    LOW_CONFIDENCE_THRESHOLD = 0.5

    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 learning_enabled: bool = True):
        """
        Initialize confidence engine.

        Args:
            weights: Custom weights for confidence sources
            learning_enabled: Whether to enable weight learning from historical data
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.learning_enabled = learning_enabled
        self.historical_data = []
        self._validate_weights()
        self._load_historical_data()

    def _validate_weights(self):
        """Validate that weights sum to 1.0"""
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.001:  # Allow small floating point errors
            logger.warning(f"Confidence weights don't sum to 1.0: {total}. Normalizing.")
            # Normalize weights
            for key in self.weights:
                self.weights[key] /= total

    def _load_historical_data(self):
        """Load historical confidence data for learning"""
        if not self.learning_enabled:
            return

        try:
            data_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'confidence_history.json')
            if os.path.exists(data_file):
                with open(data_file, 'r') as f:
                    self.historical_data = json.load(f)
                logger.info(f"Loaded {len(self.historical_data)} historical confidence records")
        except Exception as e:
            logger.warning(f"Failed to load historical data: {str(e)}")

    def calculate_initial_confidence(self, predictions: List[Dict],
                                   crop_type: str, model_uncertainty: float = 0.0) -> float:
        """
        Calculate initial confidence from image predictions with uncertainty.

        Args:
            predictions: ML model predictions
            crop_type: Identified crop type
            model_uncertainty: Model uncertainty score (0.0 to 1.0)

        Returns:
            Initial confidence score (0.0 to 1.0)
        """
        if not predictions:
            return 0.0

        # Get top prediction confidence
        top_confidence = predictions[0]['confidence']

        # Adjust for model uncertainty
        uncertainty_penalty = model_uncertainty * 0.2  # Reduce confidence by up to 20%
        adjusted_confidence = max(0.1, top_confidence - uncertainty_penalty)

        # Crop validation bonus with historical learning
        crop_bonus = self._calculate_crop_validation_bonus(predictions, crop_type)

        # Apply learned weights if available
        weights = self._get_learned_weights(crop_type)

        # Calculate weighted confidence
        confidence = (
            adjusted_confidence * weights['image_prediction'] +
            crop_bonus * weights['crop_validation']
        )

        # Add entropy-based uncertainty reduction
        entropy_penalty = self._calculate_prediction_entropy(predictions)
        confidence *= (1.0 - entropy_penalty * 0.1)  # Reduce by up to 10% based on entropy

        return min(1.0, max(0.0, confidence))

    def refine_confidence(self, qa_analysis: Dict, source: str = 'qa_reasoning',
                         session_context: Dict = None) -> Tuple[float, Dict]:
        """
        Refine confidence based on Q&A analysis with context awareness.

        Args:
            qa_analysis: Analysis from LLM service
            source: Confidence source being updated
            session_context: Additional session context

        Returns:
            Tuple of (refined confidence, detailed breakdown)
        """
        # Get confidence boost from analysis
        boost = qa_analysis.get('confidence_boost', 0.0)

        # Current confidence breakdown
        current_breakdown = qa_analysis.get('confidence_breakdown', self.weights.copy())

        # Apply context-aware adjustments
        if session_context:
            boost = self._adjust_boost_for_context(boost, session_context)

        # Update the specified source with momentum
        momentum_factor = 0.7  # How much previous confidence to retain
        current_breakdown[source] = (
            current_breakdown[source] * momentum_factor +
            (current_breakdown[source] + boost) * (1 - momentum_factor)
        )
        current_breakdown[source] = max(0.0, min(1.0, current_breakdown[source]))

        # Calculate new total confidence with learned weights
        weights = self._get_learned_weights(session_context.get('crop_type') if session_context else None)
        total_confidence = sum(
            current_breakdown[source_name] * weights.get(source_name, 0)
            for source_name in current_breakdown.keys()
        )

        # Apply uncertainty quantification
        uncertainty = self._quantify_uncertainty(current_breakdown)
        total_confidence *= (1.0 - uncertainty * 0.15)  # Reduce by up to 15% for uncertainty

        total_confidence = min(1.0, max(0.0, total_confidence))

        # Create detailed breakdown
        breakdown = {
            'total_confidence': total_confidence,
            'level': self.get_confidence_level(total_confidence),
            'sources': current_breakdown,
            'weights': weights,
            'uncertainty': uncertainty,
            'recommendation': self._get_recommendation(total_confidence, uncertainty),
            'confidence_range': self._calculate_confidence_range(total_confidence, uncertainty)
        }

        return total_confidence, breakdown

    def _calculate_crop_validation_bonus(self, predictions: List[Dict],
                                       crop_type: str) -> float:
        """
        Calculate bonus confidence from crop type validation with historical learning.

        Args:
            predictions: ML predictions
            crop_type: Crop type

        Returns:
            Crop validation bonus (0.0 to 1.0)
        """
        if not crop_type or crop_type == 'unknown':
            return 0.0

        # Check if top predictions are consistent with crop type
        top_disease = predictions[0]['disease'] if predictions else ''

        # Enhanced crop-disease mapping validation
        crop_disease_map = {
            'tomato': ['Tomato_', 'tomato'],
            'potato': ['Potato_', 'potato'],
            'corn': ['Corn_', 'corn'],
            'apple': ['Apple_', 'apple'],
            'grape': ['Grape_', 'grape'],
            'pepper': ['Pepper,', 'pepper'],
            'strawberry': ['Strawberry_', 'strawberry'],
            'orange': ['Orange_', 'orange']
        }

        expected_prefixes = crop_disease_map.get(crop_type.lower(), [])
        if any(prefix.lower() in top_disease.lower() for prefix in expected_prefixes):
            base_bonus = 0.8  # High bonus for matching crop-disease
        else:
            base_bonus = 0.3  # Moderate bonus for any prediction

        # Adjust based on historical accuracy for this crop
        historical_accuracy = self._get_historical_crop_accuracy(crop_type)
        adjusted_bonus = base_bonus * (0.5 + historical_accuracy * 0.5)  # Blend with history

        return min(1.0, adjusted_bonus)

    def _calculate_prediction_entropy(self, predictions: List[Dict]) -> float:
        """
        Calculate prediction entropy as a measure of uncertainty.

        Args:
            predictions: List of predictions with confidence scores

        Returns:
            Entropy value (0.0 to 1.0, higher means more uncertain)
        """
        if not predictions or len(predictions) < 2:
            return 0.0

        # Extract confidence scores
        confidences = np.array([p['confidence'] for p in predictions[:5]])  # Top 5
        confidences = confidences / np.sum(confidences)  # Normalize to probabilities

        # Calculate entropy
        entropy = -np.sum(confidences * np.log2(confidences + 1e-10))
        max_entropy = np.log2(len(confidences))

        # Normalize to 0-1 scale
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return min(1.0, normalized_entropy)

    def _quantify_uncertainty(self, breakdown: Dict) -> float:
        """
        Quantify overall uncertainty from confidence breakdown.

        Args:
            breakdown: Confidence breakdown dictionary

        Returns:
            Uncertainty score (0.0 to 1.0)
        """
        # Calculate variance in source confidences as uncertainty measure
        values = list(breakdown.values())
        if len(values) < 2:
            return 0.0

        variance = np.var(values)
        # Normalize variance to 0-1 scale (assuming max reasonable variance is 0.25)
        uncertainty = min(1.0, variance / 0.25)

        return uncertainty

    def _get_learned_weights(self, crop_type: Optional[str] = None) -> Dict[str, float]:
        """
        Get learned weights based on historical performance.

        Args:
            crop_type: Optional crop type for crop-specific weights

        Returns:
            Dictionary of learned weights
        """
        if not self.learning_enabled or not self.historical_data:
            return self.weights.copy()

        try:
            # Calculate average performance per source
            source_performance = {}
            crop_performance = {}

            for record in self.historical_data[-1000:]:  # Use last 1000 records
                final_confidence = record.get('final_confidence', 0.5)
                sources = record.get('sources', {})

                # Update source performance
                for source, confidence in sources.items():
                    if source not in source_performance:
                        source_performance[source] = []
                    source_performance[source].append((confidence, final_confidence))

                # Update crop-specific performance
                if crop_type and record.get('crop_type') == crop_type:
                    for source, confidence in sources.items():
                        if source not in crop_performance:
                            crop_performance[source] = []
                        crop_performance[source].append((confidence, final_confidence))

            # Calculate weights based on correlation with final confidence
            learned_weights = {}
            performance_data = crop_performance if crop_type and crop_performance else source_performance

            for source, data in performance_data.items():
                if len(data) < 10:  # Need minimum samples
                    learned_weights[source] = self.weights.get(source, 0.0)
                    continue

                # Calculate correlation between source confidence and final confidence
                source_confidences, final_confidences = zip(*data)
                correlation = np.corrcoef(source_confidences, final_confidences)[0, 1]

                # Convert correlation to weight (0.3 to 0.7 range)
                weight = 0.5 + correlation * 0.2
                weight = max(0.3, min(0.7, weight))
                learned_weights[source] = weight

            # Normalize weights
            total = sum(learned_weights.values())
            if total > 0:
                learned_weights = {k: v/total for k, v in learned_weights.items()}

            return learned_weights

        except Exception as e:
            logger.warning(f"Weight learning failed: {str(e)}")
            return self.weights.copy()

    def _get_historical_crop_accuracy(self, crop_type: str) -> float:
        """Get historical accuracy for a specific crop type"""
        if not self.historical_data:
            return 0.5  # Default

        crop_records = [r for r in self.historical_data if r.get('crop_type') == crop_type]
        if not crop_records:
            return 0.5

        accuracies = [r.get('final_confidence', 0.5) for r in crop_records[-100:]]  # Last 100
        return np.mean(accuracies) if accuracies else 0.5

    def _adjust_boost_for_context(self, boost: float, context: Dict) -> float:
        """Adjust confidence boost based on session context"""
        # Example: Reduce boost if user has provided inconsistent information
        if context.get('inconsistent_answers', False):
            boost *= 0.8

        # Increase boost for expert users
        if context.get('user_expertise') == 'expert':
            boost *= 1.1

        return boost

    def _calculate_confidence_range(self, confidence: float, uncertainty: float) -> Tuple[float, float]:
        """Calculate confidence interval"""
        margin = uncertainty * 0.2  # Uncertainty contributes to margin
        lower = max(0.0, confidence - margin)
        upper = min(1.0, confidence + margin)
        return (lower, upper)

    def get_confidence_level(self, confidence: float) -> str:
        """
        Get confidence level description.

        Args:
            confidence: Confidence score

        Returns:
            Confidence level string
        """
        if confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
            return 'high'
        elif confidence >= self.MEDIUM_CONFIDENCE_THRESHOLD:
            return 'medium'
        elif confidence >= self.LOW_CONFIDENCE_THRESHOLD:
            return 'low'
        else:
            return 'very_low'

    def should_continue_questioning(self, confidence: float,
                                  question_count: int, uncertainty: float = 0.0) -> bool:
        """
        Determine if more questions should be asked.

        Args:
            confidence: Current confidence score
            question_count: Number of questions asked so far
            uncertainty: Current uncertainty score

        Returns:
            True if should continue questioning
        """
        # Stop if confidence is high enough and uncertainty is low
        if confidence >= self.HIGH_CONFIDENCE_THRESHOLD and uncertainty < 0.3:
            return False

        # Stop if too many questions asked (avoid infinite loop)
        if question_count >= 5:
            return False

        # Continue if confidence is still low or uncertainty is high
        return confidence < self.MEDIUM_CONFIDENCE_THRESHOLD or uncertainty > 0.5

    def get_confidence_breakdown(self, confidence: float,
                               breakdown: Optional[Dict] = None,
                               uncertainty: float = 0.0) -> Dict:
        """
        Get detailed confidence breakdown.

        Args:
            confidence: Total confidence score
            breakdown: Source confidence values
            uncertainty: Uncertainty score

        Returns:
            Detailed breakdown dictionary
        """
        if breakdown is None:
            breakdown = self.weights.copy()

        return {
            'total_confidence': confidence,
            'level': self.get_confidence_level(confidence),
            'sources': breakdown,
            'weights': self.weights,
            'uncertainty': uncertainty,
            'confidence_range': self._calculate_confidence_range(confidence, uncertainty),
            'recommendation': self._get_recommendation(confidence, uncertainty)
        }

    def _get_recommendation(self, confidence: float, uncertainty: float = 0.0) -> str:
        """Get recommendation based on confidence level and uncertainty"""
        level = self.get_confidence_level(confidence)

        # Adjust recommendation based on uncertainty
        uncertainty_modifier = ""
        if uncertainty > 0.5:
            uncertainty_modifier = " High uncertainty detected - consider additional verification."

        recommendations = {
            'high': f'Diagnosis is reliable. Proceed with treatment recommendations.{uncertainty_modifier}',
            'medium': f'Diagnosis is moderately confident. Consider additional verification.{uncertainty_modifier}',
            'low': f'Diagnosis has low confidence. More information needed.{uncertainty_modifier}',
            'very_low': f'Diagnosis is uncertain. Recommend expert consultation or better image.{uncertainty_modifier}'
        }

        return recommendations.get(level, 'Further assessment required.')

    def update_historical_data(self, session_data: Dict):
        """
        Update historical data for learning.

        Args:
            session_data: Session data including final confidence and sources
        """
        if not self.learning_enabled:
            return

        try:
            record = {
                'timestamp': datetime.utcnow().isoformat(),
                'crop_type': session_data.get('crop_type'),
                'final_confidence': session_data.get('final_confidence'),
                'sources': session_data.get('sources', {}),
                'question_count': session_data.get('question_count', 0)
            }

            self.historical_data.append(record)

            # Keep only last 5000 records to prevent memory issues
            if len(self.historical_data) > 5000:
                self.historical_data = self.historical_data[-5000:]

            # Save to file periodically (every 100 records)
            if len(self.historical_data) % 100 == 0:
                self._save_historical_data()

        except Exception as e:
            logger.error(f"Failed to update historical data: {str(e)}")

    def _save_historical_data(self):
        """Save historical data to file"""
        try:
            data_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'confidence_history.json')
            os.makedirs(os.path.dirname(data_file), exist_ok=True)

            with open(data_file, 'w') as f:
                json.dump(self.historical_data[-2000:], f, indent=2)  # Save last 2000 records

        except Exception as e:
            logger.error(f"Failed to save historical data: {str(e)}")

    def _validate_weights(self):
        """Validate that weights sum to 1.0"""
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.001:  # Allow small floating point errors
            logger.warning(f"Confidence weights don't sum to 1.0: {total}. Normalizing.")
            # Normalize weights
            for key in self.weights:
                self.weights[key] /= total

    def calculate_initial_confidence(self, predictions: List[Dict],
                                   crop_type: str) -> float:
        """
        Calculate initial confidence from image predictions.

        Args:
            predictions: ML model predictions
            crop_type: Identified crop type

        Returns:
            Initial confidence score (0.0 to 1.0)
        """
        if not predictions:
            return 0.0

        # Get top prediction confidence
        top_confidence = predictions[0]['confidence']

        # Apply initial confidence reduction for UX
        # This encourages the Q&A process
        initial_reduction = 0.2  # Reduce by 20% initially
        adjusted_confidence = max(0.1, top_confidence - initial_reduction)

        # Crop validation bonus
        crop_bonus = self._calculate_crop_validation_bonus(predictions, crop_type)

        # Calculate weighted confidence
        confidence = (
            adjusted_confidence * self.weights['image_prediction'] +
            crop_bonus * self.weights['crop_validation']
        )

        return min(1.0, confidence)

    def refine_confidence(self, qa_analysis: Dict, source: str = 'qa_reasoning') -> float:
        """
        Refine confidence based on Q&A analysis.

        Args:
            qa_analysis: Analysis from LLM service
            source: Confidence source being updated

        Returns:
            Refined confidence score
        """
        # Get confidence boost from analysis
        boost = qa_analysis.get('confidence_boost', 0.0)

        # Current confidence breakdown
        current_breakdown = qa_analysis.get('confidence_breakdown', self.DEFAULT_WEIGHTS.copy())

        # Update the specified source
        current_breakdown[source] += boost
        current_breakdown[source] = max(0.0, min(1.0, current_breakdown[source]))

        # Calculate new total confidence
        total_confidence = sum(
            current_breakdown[source] * weight
            for source, weight in self.weights.items()
        )

        return min(1.0, total_confidence)

    def _calculate_crop_validation_bonus(self, predictions: List[Dict],
                                       crop_type: str) -> float:
        """
        Calculate bonus confidence from crop type validation.

        Args:
            predictions: ML predictions
            crop_type: Crop type

        Returns:
            Crop validation bonus (0.0 to 1.0)
        """
        if not crop_type or crop_type == 'unknown':
            return 0.0

        # Check if top predictions are consistent with crop type
        top_disease = predictions[0]['disease'] if predictions else ''

        # Simple crop-disease mapping validation
        crop_disease_map = {
            'tomato': ['Tomato_', 'tomato'],
            'potato': ['Potato_', 'potato'],
            'corn': ['Corn_', 'corn'],
            'apple': ['Apple_', 'apple'],
            'grape': ['Grape_', 'grape']
        }

        expected_prefixes = crop_disease_map.get(crop_type.lower(), [])
        if any(prefix.lower() in top_disease.lower() for prefix in expected_prefixes):
            return 0.8  # High bonus for matching crop-disease
        else:
            return 0.3  # Moderate bonus for any prediction

    def get_confidence_level(self, confidence: float) -> str:
        """
        Get confidence level description.

        Args:
            confidence: Confidence score

        Returns:
            Confidence level string
        """
        if confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
            return 'high'
        elif confidence >= self.MEDIUM_CONFIDENCE_THRESHOLD:
            return 'medium'
        elif confidence >= self.LOW_CONFIDENCE_THRESHOLD:
            return 'low'
        else:
            return 'very_low'

    def should_continue_questioning(self, confidence: float,
                                  question_count: int) -> bool:
        """
        Determine if more questions should be asked.

        Args:
            confidence: Current confidence score
            question_count: Number of questions asked so far

        Returns:
            True if should continue questioning
        """
        # Stop if confidence is high enough
        if confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
            return False

        # Stop if too many questions asked (avoid infinite loop)
        if question_count >= 5:
            return False

        # Continue if confidence is still low
        return confidence < self.MEDIUM_CONFIDENCE_THRESHOLD

    def get_confidence_breakdown(self, confidence: float,
                               breakdown: Optional[Dict] = None) -> Dict:
        """
        Get detailed confidence breakdown.

        Args:
            confidence: Total confidence score
            breakdown: Source confidence values

        Returns:
            Detailed breakdown dictionary
        """
        if breakdown is None:
            breakdown = self.DEFAULT_WEIGHTS.copy()

        return {
            'total_confidence': confidence,
            'level': self.get_confidence_level(confidence),
            'sources': breakdown,
            'weights': self.weights,
            'recommendation': self._get_recommendation(confidence)
        }

    def _get_recommendation(self, confidence: float) -> str:
        """Get recommendation based on confidence level"""
        level = self.get_confidence_level(confidence)

        recommendations = {
            'high': 'Diagnosis is reliable. Proceed with treatment recommendations.',
            'medium': 'Diagnosis is moderately confident. Consider additional verification.',
            'low': 'Diagnosis has low confidence. More information needed.',
            'very_low': 'Diagnosis is uncertain. Recommend expert consultation or better image.'
        }

        return recommendations.get(level, 'Further assessment required.')