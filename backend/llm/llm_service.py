# llm/llm_service.py - LLM service for intelligent questioning
"""
LLM Service - Handles intelligent follow-up question generation and reasoning.

Responsibilities:
- Generate crop-specific diagnostic questions
- Analyze user answers to refine predictions
- Provide fallback rule-based questioning
- Integrate with Google Gemini API or local LLM

Features:
- Progressive questioning based on confidence
- Visually verifiable symptom questions
- Bias-free question formulation
- Multi-turn conversation management
"""

import os
import json
import logging
from typing import Dict, List, Optional
import requests
from flask import current_app
import google.generativeai as genai

logger = logging.getLogger(__name__)

class LLMService:
    """
    Service for LLM-powered diagnostic reasoning and questioning.
    Provides intelligent follow-up questions to improve prediction confidence.
    """

    def __init__(self):
        """Initialize LLM service"""
        self.gemini_api_key = current_app.config.get('GEMINI_API_KEY')
        self.ollama_base_url = current_app.config.get('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.use_gemini = bool(self.gemini_api_key)
        self.use_ollama = self._check_ollama_availability()

        # Initialize Gemini if API key is available
        if self.use_gemini:
            try:
                genai.configure(api_key=self.gemini_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
                logger.info("Gemini API initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {str(e)}")
                self.use_gemini = False

        # Fallback question templates
        self.fallback_questions = self._load_fallback_questions()

        logger.info(f"LLM Service initialized - Gemini: {self.use_gemini}, Ollama: {self.use_ollama}")

    def _check_ollama_availability(self) -> bool:
        """Check if Ollama service is available"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def _load_fallback_questions(self) -> Dict:
        """Load rule-based fallback questions"""
        return {
            'tomato': [
                "Are there dark spots with yellow halos on the leaves?",
                "Do you see white mold on the underside of leaves?",
                "Are the leaves curling upward with yellow mottling?",
                "Is there a fuzzy gray mold on green or ripening fruit?"
            ],
            'potato': [
                "Are there dark lesions with concentric rings on leaves?",
                "Do you see water-soaked spots that turn brown?",
                "Are the stems showing dark cankers?",
                "Is there wilting of the entire plant?"
            ],
            'corn': [
                "Are there rust-colored pustules on both leaf surfaces?",
                "Do you see gray-green or tan lesions with dark borders?",
                "Are the leaves showing yellow streaks?",
                "Is there any ear rot or kernel damage?"
            ],
            'apple': [
                "Are there olive-green to black spots on leaves?",
                "Do you see scabby lesions on fruit?",
                "Are there velvety olive-green spots on leaves?",
                "Is there premature leaf drop?"
            ],
            'grape': [
                "Are there dark brown spots with yellow borders on leaves?",
                "Do you see black measles-like spots on leaves?",
                "Are there reddish-brown spots on leaves?",
                "Is there discoloration on fruit?"
            ],
            'general': [
                "Are the symptoms visible on both sides of the leaf?",
                "When did you first notice these symptoms?",
                "How widespread are the symptoms across the plant?",
                "Have you applied any pesticides recently?"
            ]
        }

    def is_available(self) -> bool:
        """Check if any LLM service is available"""
        return self.use_gemini or self.use_ollama

    def generate_questions(self, predictions: List[Dict], crop_type: str,
                          session_id: str) -> List[Dict]:
        """
        Generate follow-up questions based on predictions.

        Args:
            predictions: ML model predictions
            crop_type: Identified crop type
            session_id: Session identifier

        Returns:
            List of question dictionaries
        """
        try:
            if self.use_gemini:
                return self._generate_gemini_questions(predictions, crop_type)
            elif self.use_ollama:
                return self._generate_ollama_questions(predictions, crop_type)
            else:
                return self._generate_fallback_questions(predictions, crop_type)

        except Exception as e:
            logger.error(f"Question generation failed: {str(e)}")
            return self._generate_fallback_questions(predictions, crop_type)

    def _generate_gemini_questions(self, predictions: List[Dict], crop_type: str) -> List[Dict]:
        """Generate questions using Google Gemini API"""
        try:
            # Prepare prompt
            top_disease = predictions[0]['disease'] if predictions else 'unknown'
            confidence = predictions[0]['confidence'] if predictions else 0.0

            prompt = f"""
            You are an agricultural disease diagnostic assistant. Based on an image analysis,
            the most likely disease for a {crop_type} plant is {top_disease} with {confidence:.1%} confidence.

            Generate 2-3 follow-up questions to help confirm or refine this diagnosis.
            Questions should:
            - Be visually verifiable (what the farmer can see)
            - Avoid yes/no bias (use open-ended questions)
            - Focus on specific symptoms or patterns
            - Be relevant to {crop_type} plants and {top_disease}

            Return as JSON array of question objects with format:
            [{{"id": "q1", "question": "Describe the color and shape of the spots...", "type": "descriptive"}}]
            """

            # Call Gemini API
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text.strip()

            # Try to parse JSON response
            try:
                questions = json.loads(response_text)
                return questions
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                logger.warning("Failed to parse Gemini JSON response, using fallback")
                return self._generate_fallback_questions(predictions, crop_type)

        except Exception as e:
            logger.error(f"Gemini question generation failed: {str(e)}")
            return self._generate_fallback_questions(predictions, crop_type)

    def _generate_ollama_questions(self, predictions: List[Dict], crop_type: str) -> List[Dict]:
        """Generate questions using local Ollama LLM"""
        try:
            top_disease = predictions[0]['disease'] if predictions else 'unknown'
            confidence = predictions[0]['confidence'] if predictions else 0.0

            prompt = f"""
            You are an agricultural disease diagnostic assistant. Based on an image analysis,
            the most likely disease for a {crop_type} plant is {top_disease} with {confidence:.1%} confidence.

            Generate 2-3 follow-up questions to help confirm or refine this diagnosis.
            Questions should be visually verifiable and relevant to {crop_type} plants.

            Return as JSON array: [{{"id": "q1", "question": "Question text", "type": "descriptive"}}]
            """

            # Call Ollama API
            payload = {
                "model": "llama2",  # or whatever model is available
                "prompt": prompt,
                "stream": False
            }

            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')

                # Try to parse JSON
                try:
                    questions = json.loads(response_text)
                    return questions
                except json.JSONDecodeError:
                    pass

            return self._generate_fallback_questions(predictions, crop_type)

        except Exception as e:
            logger.error(f"Ollama question generation failed: {str(e)}")
            return self._generate_fallback_questions(predictions, crop_type)

    def _generate_fallback_questions(self, predictions: List[Dict], crop_type: str) -> List[Dict]:
        """Generate rule-based fallback questions"""
        questions = []

        # Get crop-specific questions
        crop_questions = self.fallback_questions.get(crop_type, self.fallback_questions['general'])

        for i, question_text in enumerate(crop_questions[:3]):  # Limit to 3 questions
            questions.append({
                "id": f"fallback_{crop_type}_{i+1}",
                "question": question_text,
                "type": "descriptive"
            })

        return questions

    def analyze_answer(self, session_id: str, question_id: str, answer: str) -> Dict:
        """
        Analyze user answer and provide reasoning for confidence refinement.

        Args:
            session_id: Session identifier
            question_id: Question identifier
            answer: User's answer text

        Returns:
            Dictionary with analysis and confidence refinement
        """
        try:
            if self.use_gemini:
                return self._analyze_gemini_answer(question_id, answer)
            else:
                return self._analyze_fallback_answer(question_id, answer)

        except Exception as e:
            logger.error(f"Answer analysis failed: {str(e)}")
            return self._analyze_fallback_answer(question_id, answer)

    def _analyze_gemini_answer(self, question_id: str, answer: str) -> Dict:
        """Analyze answer using Gemini"""
        try:
            prompt = f"""
            Analyze this farmer's answer to a crop disease diagnostic question.

            Question ID: {question_id}
            Answer: "{answer}"

            Determine how this answer affects diagnostic confidence. Consider:
            - Does the answer support or contradict the suspected diagnosis?
            - How specific and detailed is the answer?
            - What confidence boost/deduction should be applied?

            Return as JSON with format:
            {{
                "confidence_boost": 0.1,
                "reasoning": "Explanation of analysis",
                "next_action": "continue_questioning" or "sufficient_info",
                "confidence_breakdown": {{
                    "qa_reasoning": 0.3
                }}
            }}
            """

            response = self.gemini_model.generate_content(prompt)
            response_text = response.text.strip()

            # Try to parse JSON
            try:
                analysis = json.loads(response_text)
                return analysis
            except json.JSONDecodeError:
                logger.warning("Failed to parse Gemini analysis response")
                return self._analyze_fallback_answer(question_id, answer)

        except Exception as e:
            logger.error(f"Gemini answer analysis failed: {str(e)}")
            return self._analyze_fallback_answer(question_id, answer)

    def _analyze_fallback_answer(self, question_id: str, answer: str) -> Dict:
        """Fallback rule-based answer analysis"""
        # Simple keyword-based analysis
        positive_keywords = ['yes', 'true', 'present', 'visible', 'showing', 'dark', 'spots', 'lesions']
        negative_keywords = ['no', 'false', 'absent', 'not visible', 'no signs', 'healthy', 'normal']

        answer_lower = answer.lower()

        confidence_boost = 0.0
        if any(keyword in answer_lower for keyword in positive_keywords):
            confidence_boost = 0.15
        elif any(keyword in answer_lower for keyword in negative_keywords):
            confidence_boost = -0.1

        return {
            'confidence_boost': confidence_boost,
            'reasoning': 'Rule-based analysis of answer keywords',
            'next_action': 'continue_questioning',
            'confidence_breakdown': {
                'qa_reasoning': 0.3 + confidence_boost
            }
        }

    def generate_next_question(self, session_id: str, previous_analysis: Dict) -> List[Dict]:
        """
        Generate next question based on previous analysis.

        Args:
            session_id: Session identifier
            previous_analysis: Results from previous answer analysis

        Returns:
            List of next questions
        """
        try:
            if self.use_gemini:
                return self._generate_gemini_next_question(previous_analysis)
            else:
                return []  # Stop questioning for fallback
        except Exception as e:
            logger.error(f"Next question generation failed: {str(e)}")
            return []

    def _generate_gemini_next_question(self, previous_analysis: Dict) -> List[Dict]:
        """Generate next question using Gemini based on previous analysis"""
        try:
            prompt = f"""
            Based on the previous analysis: {json.dumps(previous_analysis)}

            Generate 1 follow-up question to further clarify the diagnosis.
            Return as JSON array: [{{"id": "q_next", "question": "Question text", "type": "followup"}}]
            """

            response = self.gemini_model.generate_content(prompt)
            response_text = response.text.strip()

            try:
                questions = json.loads(response_text)
                return questions
            except json.JSONDecodeError:
                return []

        except Exception as e:
            return []