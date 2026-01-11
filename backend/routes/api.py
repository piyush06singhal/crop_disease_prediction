# routes/api.py - Advanced REST API endpoints for Crop Disease Prediction System
"""
API Layer - RESTful endpoints for disease prediction and system interaction.

Endpoints:
- GET /health - System health check with detailed status
- GET /crops - Available crop types with metadata
- POST /predict - Disease prediction from image with advanced options
- POST /answer - Submit answer to follow-up question with validation
- POST /refine - Refine prediction with additional data
- GET /explain - Get explanation for prediction with Grad-CAM
- GET /history - Get prediction history with filtering
- POST /batch-predict - Batch prediction for multiple images
- GET /analytics - System analytics and metrics
- POST /feedback - Submit user feedback on predictions

Features:
- JWT-based authentication
- Rate limiting and request throttling
- Comprehensive error handling and logging
- API versioning support
- Request/response validation
- Background task processing for heavy operations
"""

import os
import uuid
import time
from functools import wraps
from flask import Blueprint, request, jsonify, current_app, g
from werkzeug.utils import secure_filename
from services.prediction_service import PredictionService
from services.session_service import SessionService
from services.llm_service import LLMService
from services.offline_inference import OfflineInferenceService
from services.continual_learning import ContinualLearningService
from utils.validators import validate_image_file, validate_crop_type, validate_session_id
from utils.response_formatter import format_success_response, format_error_response
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import jwt
from datetime import datetime, timedelta
import logging

api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

# Initialize services
prediction_service = PredictionService()
session_service = SessionService()
llm_service = LLMService()
offline_service = OfflineInferenceService()
learning_service = ContinualLearningService()

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Logger
logger = logging.getLogger(__name__)

# JWT Configuration
JWT_SECRET = current_app.config.get('JWT_SECRET', 'your-secret-key')
JWT_ALGORITHM = 'HS256'

def token_required(f):
    """Decorator for JWT token authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return format_error_response("Token is missing", 401)

        try:
            # Remove 'Bearer ' prefix if present
            if token.startswith('Bearer '):
                token = token[7:]

            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])

            # Check expiration
            if datetime.utcnow().timestamp() > payload['exp']:
                return format_error_response("Token has expired", 401)

            g.user_id = payload['user_id']
            g.user_role = payload.get('role', 'user')

        except jwt.ExpiredSignatureError:
            return format_error_response("Token has expired", 401)
        except jwt.InvalidTokenError:
            return format_error_response("Invalid token", 401)

        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    """Decorator for admin-only endpoints"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if g.user_role != 'admin':
            return format_error_response("Admin access required", 403)
        return f(*args, **kwargs)
    return decorated_function

@api_bp.route('/health', methods=['GET'])
@limiter.limit("30 per minute")
def health():
    """
    Comprehensive health check endpoint.

    Returns:
        JSON: Detailed system status, version, and component health
    """
    try:
        # Check database connectivity
        db_healthy = session_service._check_db_health() if hasattr(session_service, '_check_db_health') else True

        # Check Redis connectivity
        redis_healthy = session_service.redis_client is not None if hasattr(session_service, 'redis_client') else False

        # Check ML model status
        ml_status = {
            'loaded': prediction_service.is_model_loaded(),
            'type': 'tflite' if hasattr(prediction_service.model_service, 'interpreter') and prediction_service.model_service.interpreter else 'keras'
        }

        # Check LLM availability
        llm_status = {
            'available': prediction_service.is_llm_available(),
            'services': ['gemini', 'ollama']
        }

        # System metrics
        import psutil
        system_metrics = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent
        }

        health_data = {
            'status': 'healthy' if all([db_healthy, ml_status['loaded']]) else 'degraded',
            'version': current_app.config.get('VERSION', '1.0.0'),
            'timestamp': datetime.utcnow().isoformat(),
            'uptime': time.time() - current_app.config.get('START_TIME', time.time()),
            'components': {
                'database': 'healthy' if db_healthy else 'unhealthy',
                'redis': 'healthy' if redis_healthy else 'unhealthy',
                'ml_model': ml_status,
                'llm_service': llm_status
            },
            'system': system_metrics,
            'features': {
                'authentication': True,
                'rate_limiting': True,
                'batch_processing': True,
                'real_time_processing': True
            }
        }

        status_code = 200 if health_data['status'] == 'healthy' else 503
        return format_success_response(health_data), status_code

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return format_error_response("Health check failed", 500)

@api_bp.route('/auth/login', methods=['POST'])
@limiter.limit("5 per minute")
def login():
    """
    User authentication endpoint.

    Expects:
        JSON: {
            "username": "string",
            "password": "string"
        }

    Returns:
        JSON: JWT token and user info
    """
    try:
        data = request.get_json()
        if not data:
            return format_error_response("No JSON data provided", 400)

        username = data.get('username')
        password = data.get('password')

        if not username or not password:
            return format_error_response("Username and password required", 400)

        # Mock authentication (replace with real auth system)
        if username == 'admin' and password == 'admin':
            token = jwt.encode({
                'user_id': 'admin',
                'role': 'admin',
                'exp': datetime.utcnow() + timedelta(hours=24)
            }, JWT_SECRET, algorithm=JWT_ALGORITHM)

            return format_success_response({
                'token': token,
                'user': {
                    'id': 'admin',
                    'username': 'admin',
                    'role': 'admin'
                },
                'expires_in': 86400  # 24 hours
            })

        return format_error_response("Invalid credentials", 401)

    except Exception as e:
        logger.error(f"Login failed: {str(e)}")
        return format_error_response("Authentication failed", 500)

@api_bp.route('/crops', methods=['GET'])
@limiter.limit("60 per minute")
def get_crops():
    """
    Get available crop types with detailed metadata.

    Returns:
        JSON: List of supported crop types with descriptions and disease counts
    """
    try:
        crops = prediction_service.get_supported_crops()

        # Enhanced crop information
        crop_metadata = {
            'tomato': {
                'description': 'Solanum lycopersicum - Common vegetable crop',
                'diseases': 10,
                'regions': ['Global'],
                'season': 'Year-round in controlled environments'
            },
            'potato': {
                'description': 'Solanum tuberosum - Important staple crop',
                'diseases': 8,
                'regions': ['Global'],
                'season': 'Spring to Fall'
            },
            'corn': {
                'description': 'Zea mays - Major cereal crop',
                'diseases': 6,
                'regions': ['Americas', 'Europe', 'Asia'],
                'season': 'Summer'
            },
            'apple': {
                'description': 'Malus domestica - Popular fruit crop',
                'diseases': 4,
                'regions': ['Temperate regions'],
                'season': 'Fall'
            },
            'grape': {
                'description': 'Vitis vinifera - Wine and fruit grape',
                'diseases': 5,
                'regions': ['Mediterranean', 'California', 'Australia'],
                'season': 'Summer to Fall'
            }
        }

        crops_data = []
        for crop in crops:
            metadata = crop_metadata.get(crop, {
                'description': f'{crop.title()} crop',
                'diseases': 'Unknown',
                'regions': ['Various'],
                'season': 'Varies'
            })
            crops_data.append({
                'name': crop,
                'display_name': crop.title(),
                **metadata
            })

        return format_success_response({
            'crops': crops_data,
            'total': len(crops_data)
        })

    except Exception as e:
        logger.error(f"Error fetching crops: {str(e)}")
        return format_error_response("Failed to fetch crop types", 500)

@api_bp.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
@token_required
def predict():
    """
    Advanced disease prediction from uploaded image.

    Expects:
        - image: File upload (image file)
        - crop_type: Optional crop type hint
        - options: JSON string with processing options

    Returns:
        JSON: Prediction results with confidence, questions, and metadata
    """
    start_time = time.time()

    try:
        # Validate file upload
        if 'image' not in request.files:
            return format_error_response("No image file provided", 400)

        file = request.files['image']
        if file.filename == '':
            return format_error_response("No image file selected", 400)

        # Validate image file with size limits
        if not validate_image_file(file, max_size=10*1024*1024):  # 10MB limit
            return format_error_response("Invalid or too large image file", 400)

        # Get optional parameters
        crop_type = request.form.get('crop_type')
        if crop_type and not validate_crop_type(crop_type):
            return format_error_response("Invalid crop type", 400)

        # Parse options
        options = {}
        options_str = request.form.get('options')
        if options_str:
            try:
                options = json.loads(options_str)
            except json.JSONDecodeError:
                return format_error_response("Invalid options JSON", 400)

        # Generate session ID
        session_id = str(uuid.uuid4())

        # Save uploaded file temporarily with user isolation
        user_id = g.user_id
        filename = secure_filename(file.filename)
        temp_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], user_id)
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"{session_id}_{filename}")
        file.save(temp_path)

        # Perform prediction with options
        enhance = options.get('enhance', True)
        segment = options.get('segment_leaf', False)

        result = prediction_service.predict_disease(temp_path, crop_type, session_id)

        # Add processing metadata
        processing_time = time.time() - start_time
        result.update({
            'processing_time': round(processing_time, 2),
            'user_id': user_id,
            'options_used': options,
            'image_info': {
                'original_size': os.path.getsize(temp_path),
                'filename': filename
            }
        })

        # Store session data
        session_service.store_prediction(session_id, result)

        # Clean up temp file
        os.remove(temp_path)

        # Log analytics
        logger.info(f"Prediction completed for user {user_id}, session {session_id}, time: {processing_time:.2f}s")

        return format_success_response(result)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return format_error_response("Prediction failed", 500)

@api_bp.route('/batch-predict', methods=['POST'])
@limiter.limit("5 per minute")
@token_required
def batch_predict():
    """
    Batch prediction for multiple images.

    Expects:
        - images: Multiple file uploads
        - crop_type: Optional crop type hint
        - options: Processing options

    Returns:
        JSON: Batch prediction results
    """
    try:
        if 'images' not in request.files:
            return format_error_response("No images provided", 400)

        files = request.files.getlist('images')
        if len(files) > 10:  # Limit batch size
            return format_error_response("Too many images (max 10)", 400)

        crop_type = request.form.get('crop_type')
        options_str = request.form.get('options', '{}')

        try:
            options = json.loads(options_str)
        except json.JSONDecodeError:
            options = {}

        results = []
        user_id = g.user_id

        for file in files:
            if file.filename == '':
                continue

            if not validate_image_file(file):
                continue

            session_id = str(uuid.uuid4())
            filename = secure_filename(file.filename)

            # Save temporarily
            temp_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], user_id)
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, f"{session_id}_{filename}")
            file.save(temp_path)

            try:
                # Perform prediction
                result = prediction_service.predict_disease(temp_path, crop_type, session_id)
                result['filename'] = filename
                results.append(result)

                # Store session
                session_service.store_prediction(session_id, result)

            finally:
                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        return format_success_response({
            'batch_results': results,
            'total_processed': len(results),
            'total_submitted': len(files)
        })

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return format_error_response("Batch prediction failed", 500)

@api_bp.route('/answer', methods=['POST'])
@limiter.limit("30 per minute")
@token_required
def answer_question():
    """
    Submit answer to follow-up question with enhanced validation.

    Expects:
        JSON: {
            "session_id": "string",
            "question_id": "string",
            "answer": "string",
            "confidence_in_answer": 0.0-1.0
        }

    Returns:
        JSON: Updated prediction with refined confidence
    """
    try:
        data = request.get_json()
        if not data:
            return format_error_response("No JSON data provided", 400)

        session_id = data.get('session_id')
        question_id = data.get('question_id')
        answer = data.get('answer')
        confidence = data.get('confidence_in_answer', 1.0)

        if not all([session_id, question_id, answer]):
            return format_error_response("Missing required fields", 400)

        if not validate_session_id(session_id):
            return format_error_response("Invalid session ID", 400)

        # Verify session ownership
        session_data = session_service.get_prediction(session_id)
        if not session_data or session_data.get('user_id') != g.user_id:
            return format_error_response("Session not found or access denied", 404)

        # Process answer with confidence
        result = prediction_service.process_answer(session_id, question_id, answer)

        # Adjust based on user's confidence in answer
        if confidence < 0.7:
            result['confidence'] *= 0.9  # Reduce confidence if user is uncertain

        # Update session
        session_service.update_prediction(session_id, result)

        # Log user interaction
        logger.info(f"Answer submitted for session {session_id} by user {g.user_id}")

        return format_success_response(result)

    except Exception as e:
        logger.error(f"Answer processing error: {str(e)}")
        return format_error_response("Failed to process answer", 500)

@api_bp.route('/explain/<session_id>', methods=['GET'])
@limiter.limit("20 per minute")
@token_required
def get_explanation(session_id):
    """
    Get detailed explanation for a prediction with Grad-CAM.

    Args:
        session_id: Prediction session ID

    Returns:
        JSON: Explanation including Grad-CAM heatmap and feature importance
    """
    try:
        if not validate_session_id(session_id):
            return format_error_response("Invalid session ID", 400)

        # Verify session ownership
        session_data = session_service.get_prediction(session_id)
        if not session_data or session_data.get('user_id') != g.user_id:
            return format_error_response("Session not found or access denied", 404)

        explanation = prediction_service.get_explanation(session_id)

        # Add user context
        explanation['user_id'] = g.user_id
        explanation['generated_at'] = datetime.utcnow().isoformat()

        return format_success_response(explanation)

    except Exception as e:
        logger.error(f"Explanation error: {str(e)}")
        return format_error_response("Failed to generate explanation", 500)

@api_bp.route('/history', methods=['GET'])
@limiter.limit("30 per minute")
@token_required
def get_history():
    """
    Get prediction history with advanced filtering.

    Query params:
        - limit: Number of records (default: 10, max: 100)
        - offset: Pagination offset (default: 0)
        - crop_type: Filter by crop type
        - date_from: Filter from date (ISO format)
        - date_to: Filter to date (ISO format)
        - min_confidence: Minimum confidence filter

    Returns:
        JSON: Filtered prediction history
    """
    try:
        # Parse query parameters
        limit = min(int(request.args.get('limit', 10)), 100)
        offset = int(request.args.get('offset', 0))
        crop_type = request.args.get('crop_type')
        date_from = request.args.get('date_from')
        date_to = request.args.get('date_to')
        min_confidence = request.args.get('min_confidence')

        # Build filter criteria
        filters = {'user_id': g.user_id}
        if crop_type:
            filters['crop_type'] = crop_type
        if date_from:
            filters['date_from'] = date_from
        if date_to:
            filters['date_to'] = date_to
        if min_confidence:
            filters['min_confidence'] = float(min_confidence)

        # Get filtered history
        history = session_service.get_prediction_history(limit, offset, filters)

        return format_success_response({
            'history': history,
            'pagination': {
                'limit': limit,
                'offset': offset,
                'has_more': len(history) == limit
            },
            'filters_applied': filters
        })

    except Exception as e:
        logger.error(f"History retrieval error: {str(e)}")
        return format_error_response("Failed to retrieve history", 500)

@api_bp.route('/analytics', methods=['GET'])
@limiter.limit("10 per minute")
@token_required
@admin_required
def get_analytics():
    """
    Get system analytics and metrics (admin only).

    Returns:
        JSON: Comprehensive analytics data
    """
    try:
        analytics = session_service.get_session_stats()

        # Add additional metrics
        analytics.update({
            'api_usage': {
                'total_requests': 0,  # Would be populated from logs
                'avg_response_time': 0.0,
                'error_rate': 0.0
            },
            'model_performance': {
                'accuracy_trend': [],
                'popular_diseases': analytics.get('popular_diseases', [])
            }
        })

        return format_success_response(analytics)

    except Exception as e:
        logger.error(f"Analytics retrieval error: {str(e)}")
        return format_error_response("Failed to retrieve analytics", 500)

@api_bp.route('/feedback', methods=['POST'])
@limiter.limit("20 per minute")
@token_required
def submit_feedback():
    """
    Submit user feedback on prediction accuracy.

    Expects:
        JSON: {
            "session_id": "string",
            "rating": 1-5,
            "comments": "string",
            "correct_disease": "string"
        }

    Returns:
        JSON: Feedback submission confirmation
    """
    try:
        data = request.get_json()
        if not data:
            return format_error_response("No JSON data provided", 400)

        session_id = data.get('session_id')
        rating = data.get('rating')
        comments = data.get('comments')
        correct_disease = data.get('correct_disease')

        if not session_id or rating is None:
            return format_error_response("Session ID and rating required", 400)

        if not (1 <= rating <= 5):
            return format_error_response("Rating must be between 1 and 5", 400)

        # Store feedback (would be saved to database)
        feedback_data = {
            'session_id': session_id,
            'user_id': g.user_id,
            'rating': rating,
            'comments': comments,
            'correct_disease': correct_disease,
            'submitted_at': datetime.utcnow().isoformat()
        }

        # Log feedback for model improvement
        logger.info(f"Feedback submitted: {feedback_data}")

        return format_success_response({
            'message': 'Feedback submitted successfully',
            'feedback_id': str(uuid.uuid4())
        })

    except Exception as e:
        logger.error(f"Feedback submission error: {str(e)}")
        return format_error_response("Failed to submit feedback", 500)

# ===== TREATMENT RECOMMENDATIONS ENDPOINTS =====

@api_bp.route('/treatment/<disease_name>', methods=['GET'])
@limiter.limit("50 per minute")
def get_treatment(disease_name):
    """
    Get treatment recommendations for a specific disease

    Args:
        disease_name: Name of the disease

    Query Parameters:
        language: Response language ('en' or 'hi')

    Returns:
        Treatment information and recommendations
    """
    try:
        language = request.args.get('language', 'en')

        if language not in ['en', 'hi']:
            return format_error_response("Invalid language. Use 'en' or 'hi'", 400)

        treatment_info = llm_service.get_treatment_for_disease(disease_name, language)

        return format_success_response(treatment_info)

    except Exception as e:
        logger.error(f"Treatment retrieval error: {str(e)}")
        return format_error_response("Failed to get treatment information", 500)

@api_bp.route('/analyze-disease', methods=['POST'])
@limiter.limit("20 per minute")
def analyze_disease():
    """
    Get detailed disease analysis with LLM reasoning

    Request Body:
        disease_name: Name of the detected disease
        confidence: Model confidence score
        image_description: Description of affected plant/area
        language: Response language ('en' or 'hi')

    Returns:
        Detailed analysis with treatment recommendations
    """
    try:
        data = request.get_json()

        if not data:
            return format_error_response("JSON data required", 400)

        disease_name = data.get('disease_name')
        confidence = data.get('confidence', 0.0)
        image_description = data.get('image_description', '')
        language = data.get('language', 'en')

        if not disease_name:
            return format_error_response("Disease name required", 400)

        if language not in ['en', 'hi']:
            return format_error_response("Invalid language. Use 'en' or 'hi'", 400)

        analysis = llm_service.analyze_disease(disease_name, confidence, image_description, language)

        return format_success_response(analysis)

    except Exception as e:
        logger.error(f"Disease analysis error: {str(e)}")
        return format_error_response("Failed to analyze disease", 500)

# ===== OFFLINE INFERENCE ENDPOINTS =====

@api_bp.route('/offline/status', methods=['GET'])
@limiter.limit("30 per minute")
def offline_status():
    """
    Check offline inference availability and model info

    Returns:
        Offline inference status and model information
    """
    try:
        is_available = offline_service.is_available()
        model_info = offline_service.get_model_info() if is_available else {"loaded": False}

        return format_success_response({
            "offline_available": is_available,
            "model_info": model_info,
            "tflite_supported": offline_service.TFLITE_AVAILABLE
        })

    except Exception as e:
        logger.error(f"Offline status check error: {str(e)}")
        return format_error_response("Failed to check offline status", 500)

@api_bp.route('/offline/predict', methods=['POST'])
@limiter.limit("10 per minute")
def offline_predict():
    """
    Perform offline inference on uploaded image

    Request Body:
        image: Base64 encoded image data
        image_data: Raw image bytes (alternative)

    Returns:
        Prediction results from offline model
    """
    try:
        if not offline_service.is_available():
            return format_error_response("Offline inference not available", 503)

        # Check for base64 image
        data = request.get_json()
        if data and 'image' in data:
            base64_image = data['image']
            results = offline_service.predict_from_base64(base64_image)
        else:
            # Check for uploaded file
            if 'image' not in request.files:
                return format_error_response("Image file required", 400)

            file = request.files['image']
            if not file or not validate_image_file(file.filename):
                return format_error_response("Invalid image file", 400)

            image_data = file.read()
            results = offline_service.predict(image_data)

        return format_success_response(results)

    except Exception as e:
        logger.error(f"Offline prediction error: {str(e)}")
        return format_error_response("Failed to perform offline prediction", 500)

@api_bp.route('/offline/model', methods=['GET'])
@limiter.limit("5 per minute")
def get_offline_model():
    """
    Get TFLite model data for offline caching

    Returns:
        Model binary data for PWA caching
    """
    try:
        model_data = offline_service.get_cached_model_data()

        if model_data is None:
            return format_error_response("Model not available for offline caching", 404)

        # Return model data with appropriate headers
        from flask import Response
        response = Response(model_data, mimetype='application/octet-stream')
        response.headers['Content-Disposition'] = 'attachment; filename=model.tflite'
        return response

    except Exception as e:
        logger.error(f"Model download error: {str(e)}")
        return format_error_response("Failed to download model", 500)

# ===== CONTINUAL LEARNING ENDPOINTS =====

@api_bp.route('/learning/status', methods=['GET'])
@limiter.limit("30 per minute")
def learning_status():
    """
    Get continual learning status and statistics

    Returns:
        Learning status, sample counts, and performance history
    """
    try:
        status = learning_service.get_learning_status()

        return format_success_response(status)

    except Exception as e:
        logger.error(f"Learning status error: {str(e)}")
        return format_error_response("Failed to get learning status", 500)

@api_bp.route('/learning/feedback', methods=['POST'])
@limiter.limit("20 per minute")
def submit_learning_feedback():
    """
    Submit user feedback for model improvement

    Request Body:
        session_id: Prediction session ID
        correct_label: Correct disease label
        predicted_label: Model's prediction
        confidence: Model confidence score
        feedback: Optional user feedback text

    Returns:
        Feedback submission confirmation
    """
    try:
        data = request.get_json()

        if not data:
            return format_error_response("JSON data required", 400)

        session_id = data.get('session_id')
        correct_label = data.get('correct_label')
        predicted_label = data.get('predicted_label')
        confidence = data.get('confidence', 0.0)
        feedback = data.get('feedback')

        if not all([session_id, correct_label, predicted_label]):
            return format_error_response("Session ID, correct label, and predicted label required", 400)

        # Get image path from session (simplified - would need proper session lookup)
        image_path = f"uploads/{session_id}.jpg"  # Placeholder

        success = learning_service.add_training_sample(
            image_path=image_path,
            true_label=correct_label,
            predicted_label=predicted_label,
            confidence=confidence,
            user_feedback=feedback
        )

        if success:
            return format_success_response({
                "message": "Feedback submitted for model improvement",
                "samples_added": 1
            })
        else:
            return format_error_response("Failed to add feedback sample", 400)

    except Exception as e:
        logger.error(f"Learning feedback error: {str(e)}")
        return format_error_response("Failed to submit learning feedback", 500)

@api_bp.route('/learning/retrain', methods=['POST'])
@token_required
@admin_required
@limiter.limit("2 per hour")
def trigger_retraining():
    """
    Manually trigger model retraining (admin only)

    Returns:
        Retraining initiation confirmation
    """
    try:
        # Trigger background retraining
        from concurrent.futures import ThreadPoolExecutor
        executor = ThreadPoolExecutor(max_workers=1)

        def retrain_async():
            try:
                learning_service._retrain_model()
                logger.info("Manual retraining completed")
            except Exception as e:
                logger.error(f"Manual retraining failed: {str(e)}")

        executor.submit(retrain_async)

        return format_success_response({
            "message": "Model retraining initiated",
            "status": "running"
        })

    except Exception as e:
        logger.error(f"Retraining trigger error: {str(e)}")
        return format_error_response("Failed to initiate retraining", 500)

@api_bp.route('/learning/rollback/<version>', methods=['POST'])
@token_required
@admin_required
@limiter.limit("1 per day")
def rollback_model(version):
    """
    Rollback to a previous model version (admin only)

    Args:
        version: Model version to rollback to

    Returns:
        Rollback confirmation
    """
    try:
        success = learning_service.rollback_model(version)

        if success:
            return format_success_response({
                "message": f"Successfully rolled back to version {version}",
                "current_version": version
            })
        else:
            return format_error_response(f"Failed to rollback to version {version}", 400)

    except Exception as e:
        logger.error(f"Model rollback error: {str(e)}")
        return format_error_response("Failed to rollback model", 500)

@api_bp.errorhandler(429)
def ratelimit_handler(e):
    """Handle rate limit exceeded"""
    return format_error_response("Rate limit exceeded. Please try again later.", 429)

@api_bp.errorhandler(500)
def internal_error_handler(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(e)}")
    return format_error_response("Internal server error", 500)