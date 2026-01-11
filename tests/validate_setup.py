#!/usr/bin/env python3
# tests/validate_setup.py - Test setup validation
"""
Simple script to validate that the testing setup is working correctly
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")

    try:
        # Test standard library imports
        import json
        import tempfile
        import unittest.mock
        print("✓ Standard library imports OK")

        # Test test framework imports
        import pytest
        print("✓ Pytest import OK")

        # Test backend imports (with path setup)
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

        # Mock external dependencies to avoid loading heavy libraries
        unittest.mock.patch.dict('sys.modules', {
            'tensorflow': unittest.mock.MagicMock(),
            'torch': unittest.mock.MagicMock(),
            'PIL': unittest.mock.MagicMock(),
            'cv2': unittest.mock.MagicMock(),
            'genai': unittest.mock.MagicMock(),
            'redis': unittest.mock.MagicMock(),
            'sqlalchemy': unittest.mock.MagicMock(),
        })

        # Test service imports
        from services.prediction_service import PredictionService
        from services.model_service import ModelService
        from services.llm_service import LLMService
        from services.session_service import SessionService
        from services.confidence_engine import ConfidenceEngine
        from services.image_processor import ImageProcessor
        print("✓ Backend service imports OK")

        return True

    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    print("Testing basic functionality...")

    try:
        # Test basic assertions
        assert 1 + 1 == 2
        assert "hello".upper() == "HELLO"
        print("✓ Basic assertions OK")

        # Test JSON operations
        import json
        data = {"test": "value"}
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed["test"] == "value"
        print("✓ JSON operations OK")

        return True

    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_file_operations():
    """Test file operations"""
    print("Testing file operations...")

    try:
        import tempfile
        import os

        # Test temporary file creation
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            temp_path = f.name

        # Test file reading
        with open(temp_path, 'r') as f:
            content = f.read()
            assert content == "test content"

        # Cleanup
        os.unlink(temp_path)
        print("✓ File operations OK")

        return True

    except Exception as e:
        print(f"✗ File operations test failed: {e}")
        return False

def main():
    """Main validation function"""
    print("=" * 50)
    print("TEST SETUP VALIDATION")
    print("=" * 50)

    tests = [
        test_imports,
        test_basic_functionality,
        test_file_operations,
    ]

    results = []
    for test in tests:
        results.append(test())
        print()

    print("=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)

    passed = sum(results)
    total = len(results)

    for i, (test, result) in enumerate(zip(tests, results)):
        status = "PASS" if result else "FAIL"
        print(f"{test.__name__}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All validation tests passed! Testing setup is ready.")
        return 0
    else:
        print("❌ Some validation tests failed. Please check the setup.")
        return 1

if __name__ == '__main__':
    sys.exit(main())