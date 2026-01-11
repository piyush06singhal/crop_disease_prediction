# api/index.py - Vercel Serverless Function for Crop Disease Prediction System
"""
Vercel serverless function entry point for the Crop Disease Prediction System.
This wraps the Flask application to work with Vercel's serverless environment.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set environment variables for Vercel
os.environ.setdefault('FLASK_ENV', 'production')
os.environ.setdefault('SECRET_KEY', os.environ.get('SECRET_KEY', 'vercel-deployment-key'))

# Import the Flask app
from backend.app import create_app

# Create the Flask app
app = create_app()

# Vercel serverless function handler
def handler(event, context):
    """
    Vercel serverless function handler.
    Converts Vercel event format to WSGI environ and back.
    """
    from werkzeug.wrappers import Request
    from werkzeug.wsgi import get_input_stream
    from io import BytesIO
    import json

    # Extract request data from Vercel event
    request_data = event.get('body', '')
    if event.get('isBase64Encoded'):
        import base64
        request_data = base64.b64decode(request_data)

    # Create WSGI environ
    environ = {
        'REQUEST_METHOD': event.get('httpMethod', 'GET'),
        'SCRIPT_NAME': '',
        'PATH_INFO': event.get('path', '/'),
        'QUERY_STRING': event.get('queryStringParameters', {}).get('query', ''),
        'CONTENT_TYPE': event.get('headers', {}).get('content-type', ''),
        'CONTENT_LENGTH': str(len(request_data)) if request_data else '0',
        'SERVER_NAME': 'vercel',
        'SERVER_PORT': '443',
        'wsgi.version': (1, 0),
        'wsgi.url_scheme': 'https',
        'wsgi.input': BytesIO(request_data.encode() if isinstance(request_data, str) else request_data),
        'wsgi.errors': sys.stderr,
        'wsgi.multithread': False,
        'wsgi.multiprocess': False,
        'wsgi.run_once': False,
    }

    # Add headers
    for header_name, header_value in event.get('headers', {}).items():
        environ[f'HTTP_{header_name.upper().replace("-", "_")}'] = header_value

    # Response collector
    response_data = []
    response_headers = []
    response_status = None

    def start_response(status, headers, exc_info=None):
        nonlocal response_status, response_headers
        response_status = status
        response_headers = headers

    # Call the Flask app
    response_iterable = app(environ, start_response)

    # Collect response
    for data in response_iterable:
        response_data.append(data)

    # Prepare response for Vercel
    response_body = b''.join(response_data).decode('utf-8')

    # Check if response is JSON
    is_json = any('application/json' in header[1].lower() for header in response_headers)

    return {
        'statusCode': int(response_status.split()[0]),
        'headers': dict(response_headers),
        'body': response_body if not is_json else json.loads(response_body) if response_body else {},
        'isBase64Encoded': False
    }


# For local development and testing
if __name__ == '__main__':
    print("Starting Flask app for local development...")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)