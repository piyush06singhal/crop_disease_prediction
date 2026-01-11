# utils/validators.py - Input validation utilities
"""
Validation utilities for API inputs and file uploads.

Provides validation functions for:
- Image file uploads
- Crop type validation
- API request parameters
- File size and type checking
"""

import os
from typing import Optional
from werkzeug.datastructures import FileStorage
from flask import current_app

def validate_image_file(file: FileStorage) -> bool:
    """
    Validate uploaded image file.

    Args:
        file: Flask file upload object

    Returns:
        True if file is valid image, False otherwise
    """
    if not file:
        return False

    # Check filename
    if not file.filename:
        return False

    # Check file extension
    allowed_extensions = current_app.config.get('ALLOWED_EXTENSIONS', {'png', 'jpg', 'jpeg', 'bmp'})
    file_ext = os.path.splitext(file.filename)[1].lower().lstrip('.')
    if file_ext not in allowed_extensions:
        return False

    # Check file size
    max_size = current_app.config.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024)  # 16MB default
    if hasattr(file, 'content_length') and file.content_length:
        if file.content_length > max_size:
            return False

    # Additional validation could include:
    # - MIME type checking
    # - Image dimension validation
    # - File corruption checks

    return True

def validate_crop_type(crop_type: str) -> bool:
    """
    Validate crop type parameter.

    Args:
        crop_type: Crop type string

    Returns:
        True if crop type is supported, False otherwise
    """
    supported_crops = [
        'tomato', 'potato', 'corn', 'pepper', 'apple', 'grape',
        'orange', 'peach', 'strawberry', 'cherry', 'wheat', 'rice'
    ]

    return crop_type.lower() in supported_crops

def validate_session_id(session_id: str) -> bool:
    """
    Validate session ID format.

    Args:
        session_id: Session identifier

    Returns:
        True if session ID is valid UUID format
    """
    import re
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    return bool(re.match(uuid_pattern, session_id))

def validate_question_id(question_id: str) -> bool:
    """
    Validate question ID format.

    Args:
        question_id: Question identifier

    Returns:
        True if question ID format is valid
    """
    # Question IDs should be alphanumeric with underscores
    import re
    return bool(re.match(r'^[a-zA-Z0-9_]+$', question_id))

def validate_prediction_request(data: dict) -> tuple[bool, Optional[str]]:
    """
    Validate prediction API request data.

    Args:
        data: Request JSON data

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(data, dict):
        return False, "Request data must be JSON object"

    # Check required fields for different endpoints
    # This would be customized per endpoint

    return True, None

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent security issues.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    from werkzeug.utils import secure_filename
    return secure_filename(filename)