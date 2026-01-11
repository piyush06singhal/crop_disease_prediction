# tests/README.md - Testing Suite Documentation
# Crop Disease Prediction System - Testing Suite

This directory contains comprehensive tests for the Crop Disease Prediction System, including unit tests, API integration tests, and end-to-end tests.

## Test Structure

```
tests/
├── test_services.py      # Unit tests for backend services
├── test_api.py          # API integration tests
├── test_e2e.py          # End-to-end browser tests
├── conftest.py          # Shared test configuration and fixtures
├── run_tests.py         # Test runner script
├── requirements.txt     # Test dependencies
└── README.md           # This file
```

## Test Categories

### 1. Unit Tests (`test_services.py`)
- **PredictionService**: Tests disease prediction workflow
- **ModelService**: Tests ML model loading and inference
- **LLMService**: Tests language model interactions
- **SessionService**: Tests session management
- **ConfidenceEngine**: Tests confidence calculation
- **ImageProcessor**: Tests image preprocessing

### 2. API Integration Tests (`test_api.py`)
- **Health Endpoint**: System health checks
- **Prediction Endpoint**: Image upload and analysis
- **Answer Endpoint**: Q&A functionality
- **History Endpoint**: User history retrieval
- **Analytics Endpoint**: System analytics
- **Rate Limiting**: Request throttling
- **Error Handling**: API error responses
- **CORS**: Cross-origin resource sharing
- **Load Testing**: Concurrent request handling

### 3. End-to-End Tests (`test_e2e.py`)
- **Basic Navigation**: Page loading and tab switching
- **Image Upload**: File upload and drag-and-drop
- **Disease Analysis**: Complete analysis workflow
- **Camera Functionality**: Camera access and capture
- **PWA Features**: Progressive Web App functionality
- **Responsive Design**: Cross-device compatibility
- **Error Handling**: User-facing error scenarios
- **Accessibility**: Basic accessibility checks

## Running Tests

### Prerequisites

1. Install test dependencies:
```bash
pip install -r tests/requirements.txt
```

2. Install system dependencies for e2e tests:
```bash
# For Chrome
# Download and install ChromeDriver from https://chromedriver.chromium.org/

# For Firefox
# Download and install GeckoDriver from https://github.com/mozilla/geckodriver/releases
```

### Quick Start

Run all tests:
```bash
python tests/run_tests.py all
```

Run specific test categories:
```bash
# Unit tests only
python tests/run_tests.py unit

# API tests only
python tests/run_tests.py api

# End-to-end tests only
python tests/run_tests.py e2e

# Load tests only
python tests/run_tests.py load
```

### Advanced Options

```bash
# Run with coverage reports
python tests/run_tests.py all --coverage

# Run e2e tests with specific browser
python tests/run_tests.py e2e --browser firefox --base-url http://localhost:5000

# Run in headless mode (for CI/CD)
python tests/run_tests.py e2e --headless

# Include load tests
python tests/run_tests.py all --load

# Force include e2e tests in CI
python tests/run_tests.py all --include-e2e
```

### Using pytest Directly

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=backend --cov-report=html

# Run specific test file
pytest tests/test_services.py

# Run specific test class
pytest tests/test_services.py::TestPredictionService

# Run specific test method
pytest tests/test_services.py::TestPredictionService::test_predict_disease_success

# Run tests with markers
pytest -m "unit and not slow"
pytest -m e2e
pytest -m integration

# Run in parallel (requires pytest-xdist)
pytest -n auto

# Generate HTML report
pytest --html=tests/reports/report.html
```

## Test Configuration

### Environment Variables

- `FLASK_ENV`: Set to 'testing' for test environment
- `SECRET_KEY`: Test secret key
- `DISABLE_EXTERNAL_APIS`: Disable external API calls in tests
- `CI`: Set to 'true' in CI environment
- `BASE_URL`: Base URL for e2e tests (default: http://localhost:5000)
- `HEADLESS`: Run browser tests in headless mode
- `BROWSER`: Browser for e2e tests (chrome/firefox)

### Pytest Configuration

Configuration is defined in `pytest.ini`:
- Test discovery patterns
- Coverage settings
- Custom markers
- Warning filters

## Test Data

Test data is automatically created in `tests/test_data/`:
- `healthy_leaf.jpg`: Sample healthy leaf image
- `diseased_leaf.jpg`: Sample diseased leaf image
- `low_quality.jpg`: Low-quality test image

## Continuous Integration

For CI/CD pipelines, use:

```bash
# Setup test environment
python tests/run_tests.py setup

# Run unit and API tests (fast)
python tests/run_tests.py unit
python tests/run_tests.py api

# Run e2e tests only on main branches or with specific triggers
python tests/run_tests.py e2e --headless --include-e2e

# Generate coverage reports
python tests/run_tests.py report
```

## Writing New Tests

### Unit Test Example

```python
import pytest
from backend.services.prediction_service import PredictionService

class TestMyService:
    def test_my_function(self):
        service = PredictionService()
        result = service.my_function("input")
        assert result == "expected_output"

    @pytest.mark.parametrize("input_val,expected", [
        ("input1", "output1"),
        ("input2", "output2"),
    ])
    def test_my_function_parametrized(self, input_val, expected):
        service = PredictionService()
        result = service.my_function(input_val)
        assert result == expected
```

### API Test Example

```python
def test_my_endpoint(client):
    response = client.post('/api/my-endpoint', json={'key': 'value'})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['success'] is True
```

### E2E Test Example

```python
def test_my_ui_feature(driver, base_url):
    driver.get(base_url)
    element = driver.find_element(By.ID, 'my-element')
    element.click()
    # Assert expected behavior
```

## Troubleshooting

### Common Issues

1. **ChromeDriver not found**: Install ChromeDriver or use webdriver-manager
2. **Port conflicts**: Ensure port 5000/5001 are available for tests
3. **GPU tests failing**: Skip GPU tests with `-m "not requires_gpu"`
4. **Slow e2e tests**: Use `--headless` mode for faster execution

### Debug Mode

Run tests with detailed output:
```bash
pytest -v -s --tb=long
```

### Performance Testing

For performance benchmarking:
```bash
# Profile test execution
pytest --durations=10

# Run with performance timer fixture
pytest -k "performance"
```

## Coverage Reports

Coverage reports are generated in `tests/coverage/`:
- HTML report: `tests/coverage/index.html`
- Terminal summary: Run with `--cov-report=term-missing`

## Contributing

When adding new features:
1. Add corresponding unit tests
2. Add API integration tests if endpoints change
3. Add e2e tests for UI changes
4. Update test documentation
5. Ensure all tests pass before merging

## Test Quality Metrics

- **Unit Test Coverage**: Target > 90%
- **API Test Coverage**: All endpoints tested
- **E2E Test Coverage**: Critical user journeys
- **Performance**: Tests complete within 5 minutes
- **Reliability**: Flaky test rate < 5%