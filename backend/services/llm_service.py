# services/llm_service.py - LLM integration for disease analysis and treatment recommendations
"""
LLM Service - Integrates with Google Gemini and Ollama for:
- Disease analysis and reasoning
- Treatment recommendations
- Agricultural knowledge base integration
- Multi-language support for responses
"""

import os
import json
import logging
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
import google.generativeai as genai
from functools import lru_cache
import structlog

logger = structlog.get_logger(__name__)

class LLMService:
    """Service for LLM-powered disease analysis and treatment recommendations"""

    def __init__(self):
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')

        # Initialize Gemini if API key is available
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        else:
            self.gemini_model = None
            logger.warning("Gemini API key not found, falling back to Ollama")

        # Treatment knowledge base
        self.treatment_db = self._load_treatment_database()

    def _load_treatment_database(self) -> Dict[str, Any]:
        """Load treatment database with agricultural disease treatments"""
        return {
            "bacterial_blight": {
                "name": "Bacterial Blight",
                "causes": ["Xanthomonas campestris", "Wet conditions", "Poor air circulation"],
                "treatments": {
                    "chemical": [
                        "Copper-based fungicides (e.g., Copper hydroxide)",
                        "Streptomycin sulfate (with caution for resistance)",
                        "Kasugamycin"
                    ],
                    "biological": [
                        "Bacillus subtilis strains",
                        "Pseudomonas fluorescens",
                        "Plant growth-promoting rhizobacteria"
                    ],
                    "cultural": [
                        "Improve air circulation between plants",
                        "Avoid overhead irrigation",
                        "Remove and destroy infected plant material",
                        "Crop rotation with non-host plants",
                        "Use disease-resistant varieties"
                    ]
                },
                "prevention": [
                    "Use certified disease-free seeds",
                    "Practice proper field sanitation",
                    "Implement 2-3 year crop rotation",
                    "Avoid working in fields when plants are wet",
                    "Apply preventive copper sprays during wet seasons"
                ]
            },
            "fungal_infection": {
                "name": "Fungal Infection",
                "causes": ["High humidity", "Poor drainage", "Fungal spores", "Overcrowding"],
                "treatments": {
                    "chemical": [
                        "Azoxystrobin (systemic fungicide)",
                        "Tebuconazole",
                        "Propiconazole",
                        "Chlorothalonil (contact fungicide)"
                    ],
                    "biological": [
                        "Trichoderma harzianum",
                        "Bacillus amyloliquefaciens",
                        "Mycorrhizal fungi for plant health improvement"
                    ],
                    "cultural": [
                        "Improve soil drainage",
                        "Reduce plant density for better air circulation",
                        "Remove infected leaves immediately",
                        "Apply mulch to prevent soil splash",
                        "Water at soil level, avoid wetting leaves"
                    ]
                },
                "prevention": [
                    "Ensure proper plant spacing",
                    "Improve field drainage",
                    "Use drip irrigation instead of overhead",
                    "Apply preventive fungicide sprays",
                    "Monitor weather conditions for disease-favorable periods"
                ]
            },
            "viral_disease": {
                "name": "Viral Disease",
                "causes": ["Insect vectors (aphids, whiteflies)", "Contaminated tools", "Infected seeds"],
                "treatments": {
                    "chemical": [
                        "Insecticides for vector control",
                        "Imidacloprid (systemic insecticide)",
                        "Acetamiprid"
                    ],
                    "biological": [
                        "Natural predators (ladybugs, lacewings)",
                        "Neem oil sprays",
                        "Insecticidal soaps"
                    ],
                    "cultural": [
                        "Remove and destroy infected plants immediately",
                        "Control insect vectors with row covers",
                        "Use virus-free certified seeds",
                        "Disinfect tools between plants",
                        "Practice crop rotation"
                    ]
                },
                "prevention": [
                    "Use virus-tested planting material",
                    "Implement strict sanitation practices",
                    "Control insect populations early",
                    "Avoid planting near infected fields",
                    "Use reflective mulches to deter aphids"
                ]
            },
            "nutrient_deficiency": {
                "name": "Nutrient Deficiency",
                "causes": ["Poor soil fertility", "pH imbalance", "Waterlogging", "Over-fertilization"],
                "treatments": {
                    "chemical": [
                        "Balanced NPK fertilizers",
                        "Micronutrient supplements (iron, zinc, magnesium)",
                        "pH adjusters (lime for acidic soil, sulfur for alkaline)"
                    ],
                    "biological": [
                        "Compost and organic matter addition",
                        "Mycorrhizal inoculants",
                        "Biofertilizers (Azospirillum, Rhizobium)"
                    ],
                    "cultural": [
                        "Soil testing and amendment",
                        "Proper irrigation management",
                        "Balanced fertilization program",
                        "Crop rotation with legumes",
                        "Green manuring"
                    ]
                },
                "prevention": [
                    "Regular soil testing",
                    "Maintain proper soil pH (6.0-7.0)",
                    "Use balanced fertilization",
                    "Incorporate organic matter",
                    "Monitor plant health regularly"
                ]
            },
            "pest_damage": {
                "name": "Pest Damage",
                "causes": ["Insect pests", "Mites", "Nematodes", "Rodents"],
                "treatments": {
                    "chemical": [
                        "Pyrethroid insecticides",
                        "Neonicotinoids",
                        "Organophosphate insecticides (with caution)",
                        "Acaricides for mites"
                    ],
                    "biological": [
                        "Beneficial insects (predators, parasitoids)",
                        "Bacillus thuringiensis (Bt)",
                        "Entomopathogenic fungi",
                        "Entomopathogenic nematodes"
                    ],
                    "cultural": [
                        "Companion planting",
                        "Crop rotation",
                        "Physical barriers (row covers)",
                        "Trap crops",
                        "Proper weed management"
                    ]
                },
                "prevention": [
                    "Monitor pest populations regularly",
                    "Use resistant varieties",
                    "Maintain field sanitation",
                    "Encourage beneficial insects",
                    "Implement integrated pest management"
                ]
            }
        }

    @lru_cache(maxsize=100)
    def analyze_disease(self, disease_name: str, confidence: float, image_description: str = "",
                       language: str = "en") -> Dict[str, Any]:
        """
        Analyze disease and provide detailed reasoning using LLM

        Args:
            disease_name: Predicted disease name
            confidence: Model confidence score
            image_description: Description of the affected plant/area
            language: Response language ('en' or 'hi')

        Returns:
            Dict with analysis results
        """
        try:
            # Get treatment information
            treatment_info = self.treatment_db.get(disease_name.lower().replace(" ", "_"), {})

            # Create analysis prompt
            prompt = self._create_analysis_prompt(disease_name, confidence, image_description, treatment_info, language)

            # Get LLM response
            analysis = self._call_llm(prompt, language)

            # Structure the response
            result = {
                "disease": disease_name,
                "confidence": confidence,
                "analysis": analysis,
                "treatment_info": treatment_info,
                "recommendations": self._generate_recommendations(treatment_info, language),
                "language": language,
                "timestamp": datetime.now().isoformat()
            }

            logger.info("Disease analysis completed", disease=disease_name, confidence=confidence)
            return result

        except Exception as e:
            logger.error("Disease analysis failed", error=str(e), disease=disease_name)
            return self._fallback_analysis(disease_name, confidence, language)

    def _create_analysis_prompt(self, disease_name: str, confidence: float,
                              image_description: str, treatment_info: Dict, language: str) -> str:
        """Create analysis prompt for LLM"""
        confidence_percent = confidence * 100
        base_prompt = f"""
        Analyze the following crop disease case:

        Disease: {disease_name}
        Confidence: {confidence_percent:.2f}%
        Image Description: {image_description}

        Treatment Information:
        {json.dumps(treatment_info, indent=2)}

        Please provide:
        1. Detailed explanation of the disease
        2. Likely causes based on symptoms
        3. Recommended treatment approaches
        4. Prevention strategies
        5. Expected outcomes and timeline

        Respond in {'Hindi' if language == 'hi' else 'English'}.
        """

        if language == "hi":
            base_prompt += "\n\nकृपया हिंदी में उत्तर दें।"

        return base_prompt

    def _call_llm(self, prompt: str, language: str) -> str:
        """Call appropriate LLM service"""
        try:
            # Try Gemini first
            if self.gemini_model:
                response = self.gemini_model.generate_content(prompt)
                return response.text

            # Fallback to Ollama
            return self._call_ollama(prompt, language)

        except Exception as e:
            logger.warning("LLM call failed, using fallback", error=str(e))
            return self._call_ollama(prompt, language)

    def _call_ollama(self, prompt: str, language: str) -> str:
        """Call Ollama API"""
        try:
            model_name = "llama2" if language == "en" else "llama2:13b"  # Use larger model for Hindi

            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 500
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                return response.json().get("response", "Analysis not available")
            else:
                raise Exception(f"Ollama API error: {response.status_code}")

        except Exception as e:
            logger.error("Ollama call failed", error=str(e))
            return self._fallback_response(language)

    def _generate_recommendations(self, treatment_info: Dict, language: str) -> Dict[str, List[str]]:
        """Generate treatment recommendations"""
        if not treatment_info:
            return {"treatments": [], "prevention": []}

        recommendations = {
            "treatments": [],
            "prevention": treatment_info.get("prevention", [])
        }

        # Combine different treatment approaches
        treatments = treatment_info.get("treatments", {})
        for category, methods in treatments.items():
            recommendations["treatments"].extend(methods[:2])  # Limit to 2 per category

        # Translate if needed
        if language == "hi":
            recommendations = self._translate_recommendations(recommendations)

        return recommendations

    def _translate_recommendations(self, recommendations: Dict) -> Dict:
        """Translate recommendations to Hindi"""
        # Simple translation mapping - in production, use proper translation service
        translations = {
            "treatments": "उपचार",
            "prevention": "रोकथाम",
            "Use certified disease-free seeds": "प्रमाणित बीज का उपयोग करें",
            "Practice proper field sanitation": "क्षेत्र स्वच्छता का अभ्यास करें",
            "Improve air circulation between plants": "पौधों के बीच हवा का संचार सुधारें",
            "Remove and destroy infected plant material": "संक्रमित पौध सामग्री को हटा दें और नष्ट करें"
        }

        translated = {}
        for key, items in recommendations.items():
            translated_key = translations.get(key, key)
            translated[translated_key] = [
                translations.get(item, item) for item in items
            ]

        return translated

    def _fallback_analysis(self, disease_name: str, confidence: float, language: str) -> Dict[str, Any]:
        """Fallback analysis when LLM is unavailable"""
        treatment_info = self.treatment_db.get(disease_name.lower().replace(" ", "_"), {})

        fallback_response = {
            "disease": disease_name,
            "confidence": confidence,
            "analysis": "Automated analysis based on disease database" if language == "en"
                      else "रोग डेटाबेस के आधार पर स्वचालित विश्लेषण",
            "treatment_info": treatment_info,
            "recommendations": self._generate_recommendations(treatment_info, language),
            "language": language,
            "timestamp": datetime.now().isoformat()
        }

        return fallback_response

    def _fallback_response(self, language: str) -> str:
        """Fallback response when all LLM services fail"""
        if language == "hi":
            return "विश्लेषण अस्थायी रूप से उपलब्ध नहीं है। कृपया बाद में पुनः प्रयास करें।"
        else:
            return "Analysis temporarily unavailable. Please try again later."

    def get_treatment_for_disease(self, disease_name: str, language: str = "en") -> Dict[str, Any]:
        """
        Get comprehensive treatment information for a specific disease

        Args:
            disease_name: Name of the disease
            language: Response language

        Returns:
            Dict with treatment information
        """
        disease_key = disease_name.lower().replace(" ", "_")
        treatment_info = self.treatment_db.get(disease_key, {})

        if language == "hi":
            treatment_info = self._translate_treatment_info(treatment_info)

        return {
            "disease": disease_name,
            "treatment_info": treatment_info,
            "language": language,
            "timestamp": datetime.now().isoformat()
        }

    def _translate_treatment_info(self, treatment_info: Dict) -> Dict:
        """Translate treatment information to Hindi"""
        if not treatment_info:
            return treatment_info

        # Deep copy and translate
        translated = json.loads(json.dumps(treatment_info))

        # Translate key fields
        translations = {
            "name": treatment_info.get("name", ""),
            "causes": "कारण",
            "treatments": "उपचार",
            "prevention": "रोकथाम",
            "chemical": "रासायनिक",
            "biological": "जैविक",
            "cultural": "सांस्कृतिक"
        }

        # Apply translations (simplified - in production use proper translation)
        translated["causes_translated"] = "कारण"  # Placeholder
        translated["treatments_translated"] = "उपचार"  # Placeholder

        return translated