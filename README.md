# Crop Disease Prediction System

AI-powered web application for detecting crop diseases from leaf images using deep learning and LLM-driven diagnostic reasoning.

## 🌟 Features

### Core Features
- **AI Disease Detection**: Transfer learning with MobileNetV2/EfficientNetB0 for 38+ disease classes
- **Progressive Confidence**: Intelligent confidence refinement through follow-up questioning
- **LLM Integration**: Google Gemini API with rule-based fallback for diagnostic questions
- **Explainable AI**: Grad-CAM heatmaps showing infected regions
- **Multi-Crop Support**: Tomato, potato, corn, pepper, apple, grape, and more
- **Mobile-First UI**: Responsive design with drag-and-drop upload

### Advanced Features ✨
- **🌐 Multi-Language**: Hindi + English support with Flask-Babel
- **📱 PWA**: Offline capability, installable app, background sync
- **🩺 Treatment Recommendations**: Comprehensive treatment database with LLM-enhanced advice
- **🔌 Offline Inference**: Browser-based TensorFlow Lite models
- **🧠 Continual Learning**: Background model retraining with user feedback
- **📊 Analytics Dashboard**: Real-time metrics and performance monitoring

## 🏗️ Tech Stack

- **Backend**: Python 3.12+, Flask, TensorFlow 2.20.0, Redis, Celery
- **Frontend**: HTML5, Tailwind CSS, Vanilla JavaScript, Service Workers
- **AI/ML**: Transfer Learning, TensorFlow Lite, Google Gemini API
- **Database**: SQLite (dev) → PostgreSQL (production)
- **Deployment**: Vercel, Docker, Streamlit

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Node.js 14+ (for Vercel deployment)
- Google Gemini API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/piyush06singhal/crop_disease_prediction.git
   cd crop-disease-prediction
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env  # Create from template
   # Edit .env with your API keys
   ```

4. **Run the application**
   ```bash
   # Development
   python backend/app.py

   # With Gunicorn (production)
   gunicorn -w 4 -b 0.0.0.0:5000 backend.app:create_app()
   ```

## 📋 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | System health check |
| GET | `/crops` | Supported crop types |
| POST | `/predict` | Disease prediction from image |
| POST | `/answer` | Submit answer to follow-up question |
| GET | `/explain/<id>` | Get prediction explanation |
| GET | `/treatment/<disease>` | Get treatment recommendations |
| GET | `/lang/<code>` | Switch language (en/hi) |
| GET | `/offline/status` | Check offline inference status |

## 🧪 Testing

```bash
# Run all tests
python tests/run_tests.py all

# Run unit tests only
python tests/run_tests.py unit

# Run API tests
python tests/run_tests.py api
```

## 🚢 Deployment

### Vercel (Recommended)
```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
npm run vercel-deploy
```

### Docker
```bash
# Build and run
docker-compose up --build
```

### Streamlit (Alternative)
```bash
streamlit run streamlit_app.py
```

## 🔧 Environment Variables

Create a `.env` file with:

```bash
# Flask
SECRET_KEY=your-secret-key
FLASK_ENV=development

# APIs
GEMINI_API_KEY=your-gemini-api-key
OLLAMA_BASE_URL=http://localhost:11434

# Database
DATABASE_URL=sqlite:///crop_disease.db

# Redis
REDIS_URL=redis://localhost:6379/0
```

## 📊 Project Structure

```
crop-disease-prediction/
├── backend/                 # Flask application
│   ├── app.py              # Main application
│   ├── config.py           # Configuration
│   ├── routes/             # API endpoints
│   ├── services/           # Business logic
│   ├── models/             # ML models
│   └── utils/              # Utilities
├── frontend/               # Web interface
│   ├── templates/          # HTML templates
│   └── static/             # CSS/JS assets
├── tests/                  # Test suite
├── analytics/              # Analytics dashboard
├── training/               # ML training notebooks
├── api/                    # Vercel deployment
└── requirements.txt        # Python dependencies
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PlantVillage dataset for training data
- TensorFlow team for ML framework
- Google for Gemini API
- Flask community for web framework

---

**Built with ❤️ for farmers worldwide** 🌾

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- pip
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd crop-disease-ai
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

5. **Run the application**
   ```bash
   python backend/app.py
   ```

6. **Open in browser**
   ```
   http://localhost:5000
   ```

## 📁 Project Structure

```
crop-disease-ai/
│
├── backend/
│   ├── app.py                 # Main Flask application
│   ├── config.py              # Configuration management
│   ├── models/                # Database models
│   ├── services/              # Business logic services
│   │   ├── prediction_service.py
│   │   └── session_service.py
│   ├── routes/                # API and web routes
│   │   ├── api.py
│   │   └── web.py
│   ├── ml/                    # ML model management
│   │   └── model_service.py
│   ├── llm/                   # LLM integration
│   │   └── llm_service.py
│   ├── utils/                 # Utility functions
│   │   ├── validators.py
│   │   ├── response_formatter.py
│   │   ├── confidence_engine.py
│   │   └── image_processor.py
│   ├── sessions/              # Session management
│   ├── tests/                 # Unit and integration tests
│   └── uploads/               # Temporary file storage
│
├── frontend/
│   ├── templates/             # HTML templates
│   │   └── index.html
│   ├── static/                # CSS, JS, images
│   │   └── js/
│   │       └── app.js
│   └── components/            # Reusable UI components
│
├── training/                  # ML training pipeline
│   ├── notebooks/             # Jupyter notebooks
│   ├── scripts/               # Training scripts
│   └── experiments/           # Experiment tracking
│
├── api/                       # Vercel serverless functions
│   ├── index.py               # Main serverless function
│   └── requirements.txt       # Vercel-specific dependencies
│
├── analytics/                 # Analytics dashboard
├── reports/                   # Documentation and reports
├── vercel.json                # Vercel deployment config
├── .vercelignore             # Vercel ignore patterns
└── README.md
```

## 🔧 Configuration

### Environment Variables (.env)

```bash
# Flask Configuration
FLASK_ENV=development
SECRET_KEY=your-secret-key-here

# Database
DATABASE_URL=sqlite:///crop_disease.db

# Redis
REDIS_URL=redis://localhost:6379/0

# LLM APIs
GEMINI_API_KEY=your-gemini-api-key
OLLAMA_BASE_URL=http://localhost:11434

# ML Models
MODEL_DIR=models/
UPLOAD_FOLDER=uploads/
```

## 🧪 API Endpoints

### Core Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | System health check |
| GET | `/crops` | Get supported crop types |
| POST | `/predict` | Disease prediction from image |
| POST | `/answer` | Submit answer to follow-up question |
| POST | `/refine` | Refine prediction with additional data |
| GET | `/explain/<session_id>` | Get prediction explanation |
| GET | `/history` | Get prediction history |
| POST | `/feedback` | Submit user feedback |

### Phase 3 Advanced Endpoints

#### 🌐 Multi-Language Support
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/lang/<code>` | Switch application language (en/hi) |

#### 🩺 Treatment & Analysis
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/treatment/<disease>` | Get treatment recommendations |
| POST | `/analyze-disease` | LLM-powered disease analysis |

#### 🔌 Offline Inference
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/offline/status` | Check offline inference availability |
| POST | `/offline/predict` | Perform offline prediction |
| GET | `/offline/model` | Download TFLite model for caching |

#### 🧠 Continual Learning
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/learning/status` | Get learning pipeline status |
| POST | `/learning/feedback` | Submit feedback for model improvement |
| POST | `/learning/retrain` | Trigger manual model retraining (admin) |
| POST | `/learning/rollback/<version>` | Rollback to model version (admin) |

#### 📊 Analytics
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/analytics/summary` | Get system analytics summary |
| GET | `/analytics/realtime` | Real-time metrics stream |
| GET | `/analytics/users` | User behavior analytics |
| GET | `/analytics/models` | Model performance tracking |

### Example API Usage

```bash
# Health check
curl http://localhost:5000/api/health

# Upload image for prediction
curl -X POST -F "image=@leaf.jpg" http://localhost:5000/api/predict

# Get treatment recommendations
curl "http://localhost:5000/api/treatment/bacterial_blight?language=hi"

# Check offline inference status
curl http://localhost:5000/api/offline/status

# Submit learning feedback
curl -X POST -H "Content-Type: application/json" \
  -d '{"session_id": "123", "correct_label": "Healthy", "predicted_label": "Bacterial Blight", "confidence": 0.85}' \
  http://localhost:5000/api/learning/feedback

# Switch language
curl http://localhost:5000/lang/hi
```

## 🤖 ML Pipeline

### Training
1. **Dataset**: PlantVillage (38 classes, 54,305 images)
2. **Preprocessing**: Resize, augmentation, normalization
3. **Architecture**: Transfer learning with MobileNetV2/EfficientNetB0
4. **Training**: Frozen base layers → fine-tuning → optimization

### Model Formats
- **Keras (.h5)**: Full model for training/research
- **TensorFlow Lite (.tflite)**: Optimized for inference
- **ONNX**: Cross-platform deployment

### Performance
- **Accuracy**: 95%+ on validation set
- **Inference Time**: <100ms on CPU
- **Model Size**: ~20MB (TFLite optimized)

## 🔍 Progressive Confidence System

The system uses a weighted confidence approach:

- **Image Prediction (50%)**: Initial ML model confidence
- **Crop Validation (20%)**: Consistency with crop type
- **Q&A Reasoning (30%)**: LLM analysis of user answers

Confidence thresholds:
- **High (≥90%)**: Reliable diagnosis
- **Medium (70-89%)**: Moderately confident
- **Low (<70%)**: Needs more information

## 🌐 LLM Integration

### Primary: Google Gemini API
- Intelligent question generation
- Answer analysis and reasoning
- Context-aware follow-up questions

### Fallback: Rule-based System
- Predefined question templates
- Keyword-based answer analysis
- Crop-specific diagnostic logic

### Local LLM: Ollama Integration
- Privacy-preserving local inference
- Offline capability
- Custom fine-tuned models

## 📊 Explainability (Grad-CAM)

The system provides visual explanations:
- Heatmaps showing infected regions
- Feature importance analysis
- Reasoning behind predictions

## 🧪 Testing

```bash
# Run unit tests
pytest backend/tests/

# Run with coverage
pytest --cov=backend --cov-report=html

# Run integration tests
pytest backend/tests/integration/
```

## 🚀 Deployment Options

Choose the deployment method that best fits your needs:

### 1. Vercel Deployment (Recommended) ⭐

**Best for**: Production web applications, automatic scaling, global CDN

```bash
# Quick deploy
./deploy.sh  # Linux/Mac
# or
deploy.bat   # Windows

# Manual deploy
vercel --prod
```

**Features**:
- ⚡ Serverless functions with automatic scaling
- 🌍 Global CDN for fast loading worldwide
- 🔄 Automatic HTTPS and custom domains
- 📊 Built-in analytics and monitoring
- 💰 Generous free tier

#### Prerequisites
- Vercel account ([vercel.com](https://vercel.com))
- GitHub repository

#### Environment Variables
Set these in your Vercel project settings:
```env
FLASK_ENV=production
SECRET_KEY=your-secure-random-key
GOOGLE_API_KEY=your-gemini-api-key
REDIS_URL=redis://your-redis-url (optional)
```

### 2. Streamlit Cloud Deployment

**Best for**: Quick demos and prototypes

```bash
# Run demo locally
streamlit run streamlit_app.py

# Deploy to Streamlit Cloud at share.streamlit.io
```

**Features**:
- 🎨 Beautiful UI with minimal code
- 📱 Mobile-responsive
- 🔧 Easy deployment
- 📊 Built-in data visualization

### 3. Docker Deployment

**Best for**: Self-hosted solutions, custom infrastructure

```bash
# Run with Docker Compose
docker-compose up --build

# Or manual Docker commands
docker build -t crop-disease-ai .
docker run -p 5000:5000 crop-disease-ai
```

**Features**:
- 🐳 Containerized deployment
- 🔒 Isolated environment
- ⚙️ Full control over infrastructure
- 📈 Scalable with orchestration tools

### Deployment Comparison

| Feature | Vercel | Streamlit | Docker |
|---------|--------|-----------|--------|
| Ease of Use | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Scaling | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Customization | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Cost | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Full Features | ✅ | ❌ (Demo only) | ✅ |

**Recommendation**: Use **Vercel** for production deployment with all features enabled.

## 🔒 Security

- Input validation and sanitization
- Rate limiting on API endpoints
- Secure file upload handling
- Environment variable management
- CORS configuration

## 📈 Monitoring

- Health check endpoints
- Structured logging
- Performance metrics
- Error tracking and alerting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PlantVillage dataset for training data
- TensorFlow/Keras for ML framework
- Google for Gemini API
- Open source community

## 🎯 Phase 3 Features Usage Guide

### 🌐 Multi-Language Support

The application supports English and Hindi languages:

1. **Language Switching**: Click language buttons in the header to switch between English/Hindi
2. **Persistent Settings**: Language preference is saved in session
3. **Translated Content**: All UI elements, disease names, and LLM responses are translated
4. **URL-based Switching**: Use `/lang/en` or `/lang/hi` to switch languages

### 📱 Progressive Web App (PWA)

Enable PWA features for mobile app-like experience:

1. **Installation**: Click "Install App" when prompted or use browser menu
2. **Offline Mode**: App works offline with cached models and data
3. **Background Sync**: Offline predictions sync when back online
4. **Push Notifications**: Receive disease alerts and updates

### 🩺 Treatment Recommendations

Get comprehensive treatment advice:

1. **Automatic Recommendations**: Treatment suggestions appear with predictions
2. **Detailed Analysis**: Use `/api/analyze-disease` for LLM-powered analysis
3. **Treatment Categories**:
   - **Chemical**: Fungicides, bactericides, insecticides
   - **Biological**: Natural predators, beneficial microbes
   - **Cultural**: Farming practices, prevention methods

### 🔌 Offline Inference

Run predictions without internet:

1. **Model Download**: Models are automatically cached for offline use
2. **Browser Inference**: Predictions run directly in your browser
3. **Fallback Support**: Graceful degradation when offline
4. **Performance**: Optimized TensorFlow Lite models for speed

### 🧠 Continual Learning

Help improve the AI model:

1. **Feedback Submission**: Correct predictions to teach the model
2. **Quality Validation**: Only high-quality feedback is used for training
3. **Automatic Retraining**: Model improves in the background
4. **Version Control**: Track model versions and rollback if needed

### 📊 Analytics Dashboard

Monitor system performance:

1. **Access Dashboard**: Visit `/analytics` for comprehensive metrics
2. **Real-time Monitoring**: Live system health and performance data
3. **User Analytics**: Understand user behavior and patterns
4. **Model Tracking**: Monitor prediction accuracy and trends

## 🔧 Configuration

### Environment Variables

```bash
# Phase 3 Features
BABEL_DEFAULT_LOCALE=en
BABEL_SUPPORTED_LOCALES=en,hi
BABEL_TRANSLATION_DIRECTORIES=backend/translations

# Offline Inference
TFLITE_MODEL_PATH=backend/models/model.tflite

# Continual Learning
CONTINUAL_LEARNING_ENABLED=true
MIN_SAMPLES_FOR_RETRAINING=100
RETRAINING_INTERVAL_DAYS=7

# Analytics
ANALYTICS_ENABLED=true
ANALYTICS_RETENTION_DAYS=90
```

### Model Management

```bash
# Convert models for offline use
python backend/services/convert_to_tflite.py

# Update translations
pybabel extract -F babel.cfg -o messages.pot .
pybabel update -i messages.pot -d translations
pybabel compile -d translations
```

## 📞 Support

For questions or issues:
- Create an issue on GitHub
- Email: support@cropdisease.ai
- Documentation: [Wiki](https://github.com/your-repo/wiki)

---

**Built with ❤️ for farmers and agricultural researchers**