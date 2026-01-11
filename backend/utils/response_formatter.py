# utils/response_formatter.py - API response formatting utilities
"""
Response formatting utilities for consistent API responses.

Provides standardized response formats for:
- Success responses
- Error responses
- Pagination responses
- Validation error responses
"""

from typing import Any, Dict, Optional
from flask import jsonify

def format_success_response(data: Any, message: Optional[str] = None,
                          status_code: int = 200) -> Dict:
    """
    Format successful API response.

    Args:
        data: Response data
        message: Optional success message
        status_code: HTTP status code

    Returns:
        Formatted response dictionary
    """
    response = {
        'success': True,
        'data': data,
        'timestamp': _get_timestamp()
    }

    if message:
        response['message'] = message

    # In Flask, we return the dict and let jsonify handle it
    # But for consistency, return the dict structure
    return response

def format_error_response(message: str, status_code: int = 400,
                         error_code: Optional[str] = None,
                         details: Optional[Dict] = None) -> Dict:
    """
    Format error API response.

    Args:
        message: Error message
        status_code: HTTP status code
        error_code: Optional error code for client handling
        details: Optional additional error details

    Returns:
        Formatted error response dictionary
    """
    response = {
        'success': False,
        'error': {
            'message': message,
            'code': error_code or _get_error_code(status_code),
            'status_code': status_code
        },
        'timestamp': _get_timestamp()
    }

    if details:
        response['error']['details'] = details

    return response

def format_paginated_response(data: list, total: int, page: int,
                            per_page: int, message: Optional[str] = None) -> Dict:
    """
    Format paginated API response.

    Args:
        data: List of items for current page
        total: Total number of items
        page: Current page number
        per_page: Items per page
        message: Optional message

    Returns:
        Formatted paginated response
    """
    response = {
        'success': True,
        'data': data,
        'pagination': {
            'total': total,
            'page': page,
            'per_page': per_page,
            'total_pages': (total + per_page - 1) // per_page
        },
        'timestamp': _get_timestamp()
    }

    if message:
        response['message'] = message

    return response

def format_validation_error(errors: Dict[str, list]) -> Dict:
    """
    Format validation error response.

    Args:
        errors: Dictionary of field errors

    Returns:
        Formatted validation error response
    """
    return format_error_response(
        message="Validation failed",
        status_code=422,
        error_code="VALIDATION_ERROR",
        details={'fields': errors}
    )

def _get_timestamp() -> str:
    """Get current timestamp in ISO format"""
    from datetime import datetime
    return datetime.utcnow().isoformat()

def _get_error_code(status_code: int) -> str:
    """Get error code based on HTTP status code"""
    error_codes = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        422: "VALIDATION_ERROR",
        500: "INTERNAL_SERVER_ERROR"
    }
    return error_codes.get(status_code, "UNKNOWN_ERROR")