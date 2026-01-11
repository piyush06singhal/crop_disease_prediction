# config.py - Configuration management for Crop Disease Prediction System
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration class"""
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    TESTING = False

    # Database
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///crop_disease.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Redis for sessions and caching
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

    # ML Model paths
    MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
    MOBILENET_MODEL_PATH = os.path.join(MODEL_DIR, 'mobilenet_v2.h5')
    EFFICIENTNET_MODEL_PATH = os.path.join(MODEL_DIR, 'efficientnet_b0.h5')
    TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, 'model.tflite')

    # Image processing
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

    # LLM Configuration
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')

    # Confidence thresholds
    INITIAL_CONFIDENCE_WEIGHT = 0.5
    CROP_VALIDATION_WEIGHT = 0.2
    QA_REASONING_WEIGHT = 0.3
    HIGH_CONFIDENCE_THRESHOLD = 0.9
    LOW_CONFIDENCE_THRESHOLD = 0.6

    # Session management
    SESSION_TYPE = 'redis'
    SESSION_REDIS = REDIS_URL
    PERMANENT_SESSION_LIFETIME = 3600  # 1 hour

    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.path.join(os.path.dirname(__file__), 'logs', 'app.log')

    # Internationalization (i18n)
    BABEL_DEFAULT_LOCALE = 'en'
    BABEL_DEFAULT_TIMEZONE = 'UTC'
    BABEL_SUPPORTED_LOCALES = ['en', 'hi']
    BABEL_TRANSLATION_DIRECTORIES = os.path.join(os.path.dirname(__file__), 'translations')

class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///dev_crop_disease.db'

class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///test_crop_disease.db'
    WTF_CSRF_ENABLED = False

class ProductionConfig(Config):
    DEBUG = False
    # In production, ensure DATABASE_URL is set to PostgreSQL
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL')

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config(config_name=None):
    """Get configuration class based on environment"""
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'development')
    return config.get(config_name, config['default'])