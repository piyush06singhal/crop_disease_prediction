# Analytics Dashboard for Crop Disease Prediction System

This module provides comprehensive analytics and monitoring capabilities for the Crop Disease Prediction System, including real-time metrics, user behavior analytics, and model performance tracking.

## Features

### 🔍 Real-time System Monitoring
- CPU, memory, and disk usage tracking
- Network connections monitoring
- Active request counting
- Historical performance data

### 📊 Prediction Analytics
- Daily prediction trends
- Crop type distribution
- Disease detection patterns
- Confidence level analysis
- Processing time metrics

### 👥 User Behavior Analytics
- Page view tracking
- Session duration analysis
- Device and browser statistics
- Hourly activity patterns
- User action logging

### 🧠 Model Performance Tracking
- Accuracy, precision, recall, F1-score monitoring
- Inference time analysis
- Model comparison charts
- Performance history tracking

### 🚨 Error Monitoring
- Error type categorization
- Endpoint error distribution
- Daily error trends
- Error rate calculation

## Installation

The analytics module requires the following dependencies:

```bash
pip install flask psutil matplotlib seaborn pandas scikit-learn
```

## Usage

### 1. Initialize Analytics in Flask App

```python
from analytics import analytics_bp, log_prediction_event, log_user_action, log_model_metrics, log_error_event

# Register the analytics blueprint
app.register_blueprint(analytics_bp)

# Example: Log a prediction event
@app.route('/predict', methods=['POST'])
def predict():
    # ... prediction logic ...
    result = model.predict(image)

    # Log the prediction
    log_prediction_event(
        session_id=request.headers.get('X-Session-ID'),
        crop_type='tomato',
        disease_predicted=result['disease'],
        confidence=result['confidence'],
        processing_time=time.time() - start_time
    )

    return jsonify(result)
```

### 2. Log User Actions

```python
# In your frontend JavaScript
fetch('/analytics/log-action', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        action: 'page_view',
        page: window.location.pathname,
        duration: sessionDuration,
        device_type: getDeviceType(),
        browser: navigator.userAgent
    })
});
```

### 3. Track Model Performance

```python
# After model evaluation
log_model_metrics(
    model_name='MobileNetV2',
    accuracy=0.92,
    precision=0.89,
    recall=0.91,
    f1_score=0.90,
    inference_time=0.15
)
```

### 4. Monitor Errors

```python
# In error handlers
@app.errorhandler(500)
def internal_error(error):
    log_error_event(
        error_type='Internal Server Error',
        error_message=str(error),
        endpoint=request.path,
        stack_trace=traceback.format_exc()
    )
    return render_template('500.html'), 500
```

## API Endpoints

### Dashboard
- `GET /analytics/` - Main analytics dashboard

### Data APIs
- `GET /analytics/api/metrics` - System metrics
- `GET /analytics/api/predictions` - Prediction analytics
- `GET /analytics/api/users` - User behavior analytics
- `GET /analytics/api/models` - Model performance analytics
- `GET /analytics/api/errors` - Error analytics
- `GET /analytics/api/report` - Comprehensive performance report

### Logging APIs
- `POST /analytics/api/log-prediction` - Log prediction event
- `POST /analytics/api/log-action` - Log user action
- `POST /analytics/api/log-model` - Log model metrics
- `POST /analytics/api/log-error` - Log error event

## Database Schema

The analytics system uses SQLite with the following tables:

### system_metrics
- timestamp (REAL)
- cpu_percent (REAL)
- memory_percent (REAL)
- disk_usage (REAL)
- network_connections (INTEGER)
- active_requests (INTEGER)

### prediction_analytics
- timestamp (REAL)
- session_id (TEXT)
- user_id (TEXT)
- crop_type (TEXT)
- disease_predicted (TEXT)
- confidence (REAL)
- processing_time (REAL)
- user_agent (TEXT)
- ip_address (TEXT)
- feedback_rating (INTEGER)
- feedback_comment (TEXT)

### user_behavior
- timestamp (REAL)
- user_id (TEXT)
- action (TEXT)
- page (TEXT)
- duration (REAL)
- device_type (TEXT)
- browser (TEXT)
- referrer (TEXT)

### model_performance
- timestamp (REAL)
- model_name (TEXT)
- accuracy (REAL)
- precision (REAL)
- recall (REAL)
- f1_score (REAL)
- inference_time (REAL)
- model_version (TEXT)

### error_logs
- timestamp (REAL)
- error_type (TEXT)
- error_message (TEXT)
- stack_trace (TEXT)
- user_id (TEXT)
- endpoint (TEXT)
- request_data (TEXT)

## Configuration

### Environment Variables
- `ANALYTICS_DB_PATH` - Path to analytics database (default: `../backend/data/analytics.db`)
- `ANALYTICS_RETENTION_DAYS` - Data retention period in days (default: 90)

### Customization
The dashboard appearance can be customized by modifying the CSS in `templates/analytics/dashboard.html`.

## Performance Considerations

- The analytics system collects data every 30 seconds by default
- Database queries are optimized for real-time dashboard updates
- Charts use Chart.js for efficient rendering
- Background monitoring runs in separate threads

## Security

- All analytics data is stored locally in SQLite
- No external data transmission by default
- Consider implementing data anonymization for user tracking
- Regular database cleanup to manage storage

## Troubleshooting

### Common Issues

1. **Charts not loading**: Ensure Chart.js and Moment.js are properly loaded
2. **No data displayed**: Check database path and permissions
3. **High CPU usage**: Adjust monitoring interval or disable background monitoring
4. **Database errors**: Ensure SQLite is available and database directory is writable

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger('analytics').setLevel(logging.DEBUG)
```

## Contributing

To extend the analytics system:

1. Add new metrics to the database schema
2. Update the `AnalyticsDashboard` class with new collection methods
3. Add corresponding API endpoints
4. Update the frontend dashboard with new charts
5. Document new features in this README

## License

This analytics module is part of the Crop Disease Prediction System and follows the same licensing terms.