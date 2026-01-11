# app.py - Production-ready Flask application for Crop Disease Prediction System
"""
Crop Disease Prediction System - Main Application
A production-ready AI-powered web application for detecting crop diseases
from leaf images using deep learning and LLM-driven diagnostic reasoning.

Architecture:
- Modular Flask blueprint architecture with API versioning
- Layered design: UI → API → Processing → ML → LLM → Storage
- Production features: Monitoring, health checks, scalability, security
- Scalable, testable, and maintainable codebase with comprehensive logging

Production Features:
- Prometheus metrics and monitoring
- Detailed health checks with component status
- Graceful shutdown handling
- Request ID tracking and correlation
- Rate limiting and security headers
- Background task processing
- Database connection pooling
- Redis session management
- Gunicorn-ready configuration
"""

import os
import sys
import time
import signal
import logging
import threading
from datetime import datetime
from flask import Flask, request, g, jsonify
from flask_cors import CORS
from flask_session import Session
from flask_sqlalchemy import SQLAlchemy
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_babel import Babel, gettext as _
from prometheus_client import Counter, Histogram, Gauge
import psutil
import structlog

from config import get_config

# Locale selector for Flask-Babel
def get_locale():
    """Get the best matching locale for the current request"""
    # Check if language is specified in session
    if 'lang' in session:
        return session['lang']

    # Check Accept-Language header
    return request.accept_languages.best_match(['en', 'hi']) or 'en'

# Initialize extensions
db = SQLAlchemy()
session = Session()
limiter = Limiter(key_func=get_remote_address)

# Global metrics
REQUEST_COUNT = Counter('flask_requests_total', 'Total number of requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('flask_request_duration_seconds', 'Request duration in seconds', ['method', 'endpoint'])
ACTIVE_REQUESTS = Gauge('flask_active_requests', 'Number of active requests')
SYSTEM_CPU = Gauge('system_cpu_percent', 'System CPU usage percentage')
SYSTEM_MEMORY = Gauge('system_memory_percent', 'System memory usage percentage')

# Application start time for uptime tracking
START_TIME = time.time()

class CropDiseaseApp:
    """Main application class with enhanced lifecycle management"""

    def __init__(self, config_name=None):
        self.config_name = config_name
        self.app = None
        self.metrics = None
        self.shutdown_event = threading.Event()

    def create_app(self):
        """
        Enhanced application factory with production features.

        Returns:
            Flask application instance with full production setup
        """
        self.app = Flask(__name__)

        # Load configuration
        config_class = get_config(self.config_name)
        self.app.config.from_object(config_class)

        # Store start time
        self.app.config['START_TIME'] = START_TIME

        # Initialize core extensions
        self._init_extensions()

        # Setup structured logging
        self._setup_structured_logging()

        # Register blueprints and routes
        self._register_blueprints()

        # Setup middleware and security
        self._setup_middleware()

        # Setup monitoring and metrics
        self._setup_monitoring()

        # Setup background tasks
        self._setup_background_tasks()

        # Register error handlers
        self._register_error_handlers()

        # Create database tables
        self._init_database()

        # Setup graceful shutdown
        self._setup_graceful_shutdown()

        return self.app

    def _init_extensions(self):
        """Initialize Flask extensions with production settings"""
        # Database with connection pooling
        db.init_app(self.app)

        # Session management with Redis
        session.init_app(self.app)

        # Rate limiting
        limiter.init_app(self.app)

        # Internationalization support
        babel = Babel(self.app)
        babel.init_app(self.app, locale_selector=get_locale)
        self.app.babel = babel

        # CORS with production settings
        CORS(self.app, resources={
            r"/api/*": {
                "origins": self.app.config.get('CORS_ORIGINS', ["http://localhost:3000", "http://localhost:5000"]),
                "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization", "X-Request-ID"],
                "expose_headers": ["X-Request-ID"],
                "supports_credentials": True
            }
        })

    def _setup_structured_logging(self):
        """Setup structured logging with JSON output for production"""
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        # Setup logging
        if not os.path.exists('logs'):
            os.makedirs('logs')

        # File handler for structured logs
        file_handler = logging.FileHandler(self.app.config['LOG_FILE'])
        file_handler.setFormatter(logging.Formatter('%(message)s'))

        # Console handler for development
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))

        # Configure root logger
        logging.getLogger().setLevel(getattr(logging, self.app.config['LOG_LEVEL']))
        logging.getLogger().addHandler(file_handler)
        if self.app.config['DEBUG']:
            logging.getLogger().addHandler(console_handler)

        # Reduce noise from third-party libraries
        logging.getLogger('werkzeug').setLevel(logging.WARNING)
        logging.getLogger('tensorflow').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)

        self.app.logger.info("Structured logging initialized")

    def _register_blueprints(self):
        """Register all Flask blueprints with proper prefixes"""
        from routes.api import api_bp
        from routes.web import web_bp

        # API blueprint with versioning
        self.app.register_blueprint(api_bp, url_prefix='/api/v1')

        # Web blueprint for frontend
        self.app.register_blueprint(web_bp)

        # Health check endpoints (outside blueprints for direct access)
        self._register_health_endpoints()

    def _register_health_endpoints(self):
        """Register comprehensive health check endpoints"""
        @self.app.route('/health')
        def basic_health():
            """Basic health check for load balancers"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat()
            }), 200

        @self.app.route('/health/detailed')
        def detailed_health():
            """Detailed health check with component status"""
            return self._get_detailed_health()

        @self.app.route('/health/ready')
        def readiness_health():
            """Kubernetes readiness probe"""
            # Check if app is ready to serve requests
            if self._check_readiness():
                return jsonify({'status': 'ready'}), 200
            return jsonify({'status': 'not ready'}), 503

        @self.app.route('/health/live')
        def liveness_health():
            """Kubernetes liveness probe"""
            # Check if app is alive (not deadlocked, etc.)
            return jsonify({'status': 'alive'}), 200

    def _setup_middleware(self):
        """Setup middleware for production features"""
        # Request ID middleware
        @self.app.before_request
        def set_request_id():
            request_id = request.headers.get('X-Request-ID') or str(uuid.uuid4())
            g.request_id = request_id
            # Add to response headers
            @self.app.after_request
            def add_request_id_header(response):
                response.headers['X-Request-ID'] = g.request_id
                return response

        # Request timing middleware
        @self.app.before_request
        def start_timer():
            g.start_time = time.time()

        @self.app.after_request
        def log_request(response):
            if hasattr(g, 'start_time'):
                duration = time.time() - g.start_time
                REQUEST_LATENCY.labels(
                    method=request.method,
                    endpoint=request.path
                ).observe(duration)

                REQUEST_COUNT.labels(
                    method=request.method,
                    endpoint=request.path,
                    status=response.status_code
                ).inc()

            # Structured logging for requests
            self.app.logger.info(
                "Request completed",
                method=request.method,
                path=request.path,
                status=response.status_code,
                duration=duration if 'duration' in locals() else None,
                user_agent=request.headers.get('User-Agent'),
                remote_addr=request.remote_addr
            )
            return response

        # Active requests tracking
        @self.app.before_request
        def track_active_requests():
            ACTIVE_REQUESTS.inc()

        @self.app.after_request
        def decrement_active_requests(response):
            ACTIVE_REQUESTS.dec()
            return response

        # Security headers
        @self.app.after_request
        def add_security_headers(response):
            response.headers['X-Content-Type-Options'] = 'nosniff'
            response.headers['X-Frame-Options'] = 'DENY'
            response.headers['X-XSS-Protection'] = '1; mode=block'
            response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
            return response

        # Proxy fix for production deployments
        self.app.wsgi_app = ProxyFix(self.app.wsgi_app, x_proto=1, x_host=1)

    def _setup_monitoring(self):
        """Setup Prometheus monitoring and metrics"""
        try:
            self.metrics = PrometheusMetrics(self.app, group_by_endpoint=True)

            # Custom metrics
            self.metrics.register_default(
                metrics=[
                    REQUEST_COUNT,
                    REQUEST_LATENCY,
                    ACTIVE_REQUESTS,
                    SYSTEM_CPU,
                    SYSTEM_MEMORY
                ]
            )

            # System metrics updater
            def update_system_metrics():
                while not self.shutdown_event.is_set():
                    try:
                        SYSTEM_CPU.set(psutil.cpu_percent(interval=1))
                        SYSTEM_MEMORY.set(psutil.virtual_memory().percent)
                    except Exception as e:
                        self.app.logger.error(f"Failed to update system metrics: {e}")
                    time.sleep(30)  # Update every 30 seconds

            metrics_thread = threading.Thread(target=update_system_metrics, daemon=True)
            metrics_thread.start()

            self.app.logger.info("Prometheus monitoring initialized")

        except ImportError:
            self.app.logger.warning("Prometheus metrics not available - install prometheus_flask_exporter")

    def _setup_background_tasks(self):
        """Setup background task processing"""
        # Import here to avoid circular imports
        from services.background_tasks import BackgroundTaskManager

        # Initialize background task manager
        task_manager = BackgroundTaskManager(self.app)
        task_manager.start()

        # Store reference for cleanup
        self.app.task_manager = task_manager

    def _register_error_handlers(self):
        """Register comprehensive error handlers"""
        @self.app.errorhandler(400)
        def bad_request(error):
            return jsonify({
                'error': 'Bad Request',
                'message': 'The request could not be understood',
                'status_code': 400
            }), 400

        @self.app.errorhandler(401)
        def unauthorized(error):
            return jsonify({
                'error': 'Unauthorized',
                'message': 'Authentication required',
                'status_code': 401
            }), 401

        @self.app.errorhandler(403)
        def forbidden(error):
            return jsonify({
                'error': 'Forbidden',
                'message': 'Access denied',
                'status_code': 403
            }), 403

        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({
                'error': 'Not Found',
                'message': 'The requested resource was not found',
                'status_code': 404
            }), 404

        @self.app.errorhandler(429)
        def rate_limit_exceeded(error):
            return jsonify({
                'error': 'Rate Limit Exceeded',
                'message': 'Too many requests. Please try again later.',
                'status_code': 429
            }), 429

        @self.app.errorhandler(500)
        def internal_error(error):
            self.app.logger.error(f"Internal server error: {error}")
            return jsonify({
                'error': 'Internal Server Error',
                'message': 'An unexpected error occurred',
                'status_code': 500
            }), 500

    def _init_database(self):
        """Initialize database with proper error handling"""
        try:
            with self.app.app_context():
                db.create_all()
                self.app.logger.info("Database tables created successfully")
        except Exception as e:
            self.app.logger.error(f"Database initialization failed: {e}")
            raise

    def _setup_graceful_shutdown(self):
        """Setup graceful shutdown handling"""
        def shutdown_handler(signum, frame):
            self.app.logger.info("Shutdown signal received, initiating graceful shutdown...")
            self.shutdown_event.set()

            # Stop background tasks
            if hasattr(self.app, 'task_manager'):
                self.app.task_manager.stop()

            # Close database connections
            db.session.remove()

            self.app.logger.info("Graceful shutdown completed")
            sys.exit(0)

        signal.signal(signal.SIGTERM, shutdown_handler)
        signal.signal(signal.SIGINT, shutdown_handler)

    def _get_detailed_health(self):
        """Get detailed health status of all components"""
        try:
            from services.prediction_service import PredictionService
            from services.session_service import SessionService

            prediction_service = PredictionService()
            session_service = SessionService()

            # Component health checks
            components = {
                'database': self._check_database_health(),
                'redis': self._check_redis_health(session_service),
                'ml_model': prediction_service.is_model_loaded(),
                'llm_service': prediction_service.is_llm_available(),
                'file_system': self._check_file_system_health()
            }

            # Overall status
            overall_status = 'healthy' if all(components.values()) else 'degraded'

            health_data = {
                'status': overall_status,
                'version': self.app.config.get('VERSION', '1.0.0'),
                'timestamp': datetime.utcnow().isoformat(),
                'uptime': time.time() - START_TIME,
                'components': components,
                'system': {
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_usage': psutil.disk_usage('/').percent
                },
                'active_requests': ACTIVE_REQUESTS._value if hasattr(ACTIVE_REQUESTS, '_value') else 0
            }

            status_code = 200 if overall_status == 'healthy' else 503
            return jsonify(health_data), status_code

        except Exception as e:
            self.app.logger.error(f"Detailed health check failed: {e}")
            return jsonify({
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }), 503

    def _check_database_health(self):
        """Check database connectivity"""
        try:
            db.session.execute(db.text('SELECT 1'))
            return True
        except Exception:
            return False

    def _check_redis_health(self, session_service):
        """Check Redis connectivity"""
        try:
            return session_service.redis_client.ping() if hasattr(session_service, 'redis_client') and session_service.redis_client else False
        except Exception:
            return False

    def _check_file_system_health(self):
        """Check file system health"""
        try:
            # Check upload directory
            upload_dir = self.app.config.get('UPLOAD_FOLDER', 'uploads')
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)
            # Try to write a test file
            test_file = os.path.join(upload_dir, '.health_check')
            with open(test_file, 'w') as f:
                f.write('health_check')
            os.remove(test_file)
            return True
        except Exception:
            return False

    def _check_readiness(self):
        """Check if application is ready to serve requests"""
        # Check critical components
        try:
            from services.prediction_service import PredictionService
            prediction_service = PredictionService()
            return prediction_service.is_model_loaded()
        except Exception:
            return False

def create_app(config_name=None):
    """
    Application factory function for backward compatibility.

    Args:
        config_name: Configuration environment name

    Returns:
        Flask application instance
    """
    crop_app = CropDiseaseApp(config_name)
    return crop_app.create_app()

if __name__ == '__main__':
    # Create application with production settings
    app = create_app()

    # Development server with production-like settings
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=app.config['DEBUG'],
        threaded=True,
        use_reloader=app.config['DEBUG']
    )
    app.register_blueprint(web_bp)

    # Import and register additional blueprints as they are created
    # from routes.ml import ml_bp
    # app.register_blueprint(ml_bp, url_prefix='/api/ml')

if __name__ == '__main__':
    app = create_app()
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=app.config['DEBUG']
    )